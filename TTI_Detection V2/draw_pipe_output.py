import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch


TEST_IMG_PATH = "./dataset/images/test/"
YOLO_WEIGHTS = "./runs/segment/train/weights/best.pt"
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


INSTRUMENT_NAME_TO_ID = {
    'dissector': 0,
    'scissors': 1,
    'suction': 2,
    'harmonic': 3,
    'grasper': 4,
    'grasper 2': 4,
    'grasper 3': 4,
    'bipolar': 5,
    'cautery (hook, spatula)': 6,
    'stapler': 7,
}

TTI_NAME_TO_ID = {
    'coagulation': 8,
    'other': 9,
    'retract and grab': 10,
    'retract and push': 10,
    'blunt dissection': 11,
    'energy - sharp dissection': 12,
    'staple': 13,
    'cut - sharp dissection': 14,
}



def expand_mask(mask, pixels):
    kernel_size = 2 * pixels + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return expanded

def parse_yolo_output(result) -> list[dict]:
    """Parse YOLO segmentation output into a list of dicts {class, mask}.

    Masks are returned in numpy format; conf is omitted because here we only
    care about class presence when comparing to ground truth labels.
    """
    if result is None or len(result) == 0:
        return []
    r = result[0]
    if r.masks is None or r.boxes is None or len(r.boxes.cls) == 0:
        return []

    classes = r.boxes.cls.cpu().numpy().astype(int)
    masks = r.masks.data.cpu().numpy()
    out = []
    for cls_id, m, conf in zip(classes, masks, r.boxes.conf.cpu().numpy()):
        out.append({'class': int(cls_id), 'mask': m,'score': float(conf)})
    return out




def yolo_inference(model, image) -> list[dict]:
    """
    image: np.ndarray (HWC, RGB)
    Returns: list of dicts { 'mask': HxW, 'class': int, 'score': float }
    """
    # Convert image to tensor, normalize, etc.
    # input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # with torch.no_grad():
    
    output = model.predict(image,verbose=False)

    # Parse output into masks & classes (depends on repo!)
    detections = parse_yolo_output(output) # function to be implemented
    
    return detections        

def extract_union_roi(image, tool_mask, tissue_mask, depth_map=None):
    combined_mask = (tool_mask + tissue_mask).clip(0, 1).astype('uint8') #before astype
    x, y, w, h = cv2.boundingRect(combined_mask)
    roi = image[y:y+h, x:x+w]
  

    if depth_map is not None:
        depth_roi = depth_map[y:y+h, x:x+w]
        roi = np.concatenate([roi, depth_roi[..., None]], axis=-1)  # add depth as extra channel

    merged_mask = cv2.bitwise_or(tool_mask, tissue_mask)
    merged_mask = merged_mask[y:y+h, x:x+w]
    merged_mask = np.expand_dims(merged_mask, axis=-1)
    
    if merged_mask.shape[1] != roi.shape[1] or merged_mask.shape[0] != roi.shape[0]:
        print("MISMATCH")
        return None

    roi = np.concatenate([roi, merged_mask*255], axis=-1)

    return roi

def find_tool_tissue_pairs(detections: list[dict]):
    tool_classes = list(range(0, 7))
    
    tools = [d for d in detections if d['class'] in tool_classes]
    tissues = [d for d in detections if d['class'] not in tool_classes]
    pairs = []
    for s in tools:
        for o in tissues:
            pairs.append({'tool': s, 'tissue': o,'tissue_conf': o['score']})
    return pairs

def depth_treshold(image, yolo_model):
  
    detections = yolo_inference(yolo_model, image)
    
   
    # Step 2: Pairing
    pairs = find_tool_tissue_pairs(detections)
    
    # Decommentare in caso di esecuzione di questo file
    # Commentare in caso di esecuzione di evaluate_with_inter.py
    # image = cv2.imread(image,cv2.IMREAD_COLOR)

    tti_predictions = []

    for pair in pairs:
        tool_mask = pair['tool']['mask']
        tissue_mask = pair['tissue']['mask']
    
        H_full, W_full = image.shape[:2]

        tool_mask_expanded = expand_mask(tool_mask, pixels=2)
        tissue_mask_expanded = expand_mask(tissue_mask, pixels=2)

        tool_mask_resized = cv2.resize(
                tool_mask_expanded.astype(np.uint8),
                (W_full, H_full),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        tissue_mask_resized = cv2.resize(
            tissue_mask_expanded.astype(np.uint8),
            (W_full, H_full),
            interpolation=cv2.INTER_NEAREST
            ).astype(bool)


        intersection = np.logical_and(tool_mask_resized, tissue_mask_resized).sum()

        tti = False

        if intersection > 0:
            tti = True

        if tti:
            tti_class = 1
        else:
            tti_class = 0
            
        # Save ROI result
        tti_predictions.append({
            'tool': pair['tool'],
            'tissue': pair['tissue'],
            'tti_class': tti_class,
            'tti_score': 1,
            'conf': pair['tissue_conf']
        })

    # Post-filter: per tool keep only the tissue with centroid closest to tool centroid
    best_index_by_tool = {}
    best_dist_by_tool = {}
    
    for idx, p in enumerate(tti_predictions):
        if p.get('tti_class', 0) != 1:
            continue
        
        tool_mask = p['tool']['mask']
        tissue_mask = p['tissue']['mask']
        
        # Calcola centroidi
        tool_moments = cv2.moments(tool_mask.astype(np.uint8))
        tissue_moments = cv2.moments(tissue_mask.astype(np.uint8))
        
        if tool_moments['m00'] == 0 or tissue_moments['m00'] == 0:
            continue
        
        tool_cx = tool_moments['m10'] / tool_moments['m00']
        tool_cy = tool_moments['m01'] / tool_moments['m00']
        tissue_cx = tissue_moments['m10'] / tissue_moments['m00']
        tissue_cy = tissue_moments['m01'] / tissue_moments['m00']
        
        # Distanza euclidea tra centroidi
        dist = np.sqrt((tool_cx - tissue_cx)**2 + (tool_cy - tissue_cy)**2)
        
        tool_key = id(p['tool'])
        if (tool_key not in best_dist_by_tool) or (dist < best_dist_by_tool[tool_key]):
            best_dist_by_tool[tool_key] = dist
            best_index_by_tool[tool_key] = idx

    for idx, p in enumerate(tti_predictions):
        if p.get('tti_class', 0) == 1:
            tool_key = id(p['tool'])
            if best_index_by_tool.get(tool_key) != idx:
                p['tti_class'] = 0

    return detections, tti_predictions


if __name__ == "__main__":
    yolo_model = YOLO(YOLO_WEIGHTS)
    # yolo_model.predict("./dataset/images/test/video0004_frame0000.png",show=True,save=True)
    # exit()
    for img_name in os.listdir(TEST_IMG_PATH):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image= os.path.join(TEST_IMG_PATH, img_name)

        detections, tti_predictions = depth_treshold(image, yolo_model)
        
        
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"Image: {img_name}")
        print("Detections:")
        for det in detections:
            print(f"  Class: {det['class']}")
        print("TTI Predictions:")
        for tti in tti_predictions:
            print(f"  Tool Class: {tti['tool']['class']}, Tissue Class: {tti['tissue']['class']}, TTI Class: {tti['tti_class']}")
            
        # Visualize detections with masks
        

       
        # Image with masks (keep original pixels untouched outside masks)
        img_with_masks = image.astype(np.float32).copy()
        alpha = 0.5  # transparency factor
        print(len(detections))
        for i, det in enumerate(detections):
            mask = det['mask']
            cls = det['class']
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            # Generate random color for each mask
            if cls in range(0, 7):
                color = np.array([200, 0, 0]).astype(np.float32)
            else:
                color = np.array([0, 220, 220]).astype(np.float32)
            
            # Blend only on masked pixels, leave the rest untouched
            img_with_masks[mask_resized] = (1 - alpha) * img_with_masks[mask_resized] + alpha * color

        img_with_masks = img_with_masks.astype(np.uint8)
        
        # Draw bounding boxes for TTI pairs
        img_with_boxes = img_with_masks.copy()
        for tti in tti_predictions:
            if tti['tti_class'] == 0:
                continue  # Skip non-TTI pairs
            tool_det = tti['tool']
            tissue_det = tti['tissue']
            
            # Get masks for tool and tissue
            tool_mask = tool_det['mask']
            tissue_mask = tissue_det['mask']
            
            # Resize masks to match image dimensions
            tool_mask_resized = cv2.resize(tool_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            tissue_mask_resized = cv2.resize(tissue_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Combine masks with logical OR
            combined_mask = np.logical_or(tool_mask_resized, tissue_mask_resized).astype(np.uint8)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(combined_mask)
            
            # Draw rectangle on the image
            color = (0, 220, 220)  # Green color in BGR
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
            
            # Get tissue class name and draw text above the box
            tissue_class = tissue_det['class']
            # Find the class name from the class ID
            class_name = None
            for name, class_id in TTI_NAME_TO_ID.items():
                if class_id == tissue_class:
                    class_name = name
                    break
            if class_name is None:
                class_name = f"Class {tissue_class}"
            
            # Draw text above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_color = (0, 0, 0)  # Black text
            text_thickness = 1
            
            # Get text size to draw background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, text_thickness)
            
            # Check if text fits above the bounding box
            text_y_top = y - text_height - baseline - 5
            text_y_bottom = y
            
            if text_y_top < 0:
                # Text goes out of bounds on top, put it below the box instead
                text_y_top = y + h + 5
                text_y_bottom = y + h + text_height + baseline + 10
            
            # Draw background rectangle for text
            bg_color = (0, 220, 220)  # Cyan background
            cv2.rectangle(img_with_boxes, 
                         (x, text_y_top),
                         (x + text_width + 5, text_y_bottom),
                         bg_color, -1)
            
            # Draw text on the background
            text_x = x + 2
            text_y = text_y_top + text_height + baseline - 2
            cv2.putText(img_with_boxes, class_name, (text_x, text_y), font, font_scale, text_color, text_thickness)
        
        plt.imshow(image)
        plt.imshow(img_with_masks)
        plt.imshow(img_with_boxes)
        cv2.imwrite(os.path.join("./img_output", f"{img_name.split('.')[0]}_detections.png"), cv2.cvtColor(img_with_masks, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join("./img_output", f"{img_name.split('.')[0]}_TTI.png"), cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    
        
        plt.show()