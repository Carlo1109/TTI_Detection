import torch
import cv2
import numpy as np
from ultralytics import YOLO
import torch.nn.functional as F
from PIL import Image


def load_yolo_model(model_path):
    # Load your YOLOv11-seg model (pseudo-code, depends on repo)
    model = YOLO(model_path,verbose=False)
    # model.eval()
    # print(model)
    return model

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


def found_objects(tool_list, tti_list, classes) -> list | list:
    
    tool_found = []
    tti_found = []
    
    for elem in range(len(classes)):
        if classes[elem] in tool_list:
            tool_found.append(elem)
        if classes[elem] in tti_list:
            tti_found.append(elem)
            
    return tool_found, tti_found



def parse_yolo_output(result) -> list[dict]:
    
    r = result[0]
    if len(result[0].boxes.cls) == 0:
        return []
    if len(result[0].boxes.cls) < 2 :
        return [{'class':int(r.boxes.cls[0].cpu().detach().numpy()) , 'mask': r.masks.data[0].cpu().detach().numpy()}]
    
    tool_list = list(range(0, 7))
    tti_list = list(range(8, 14))

    
    classes = r.boxes.cls
    masks = r.masks.data
    
    tool_found, tti_found = found_objects(tool_list, tti_list, classes)

    l = []
    
    if len(tool_found) == 0:
        for idx in tti_found:
            l.append({'class':int(r.boxes.cls[idx].cpu().detach().numpy()) , 'mask': r.masks.data[idx].cpu().detach().numpy()})
        return l
    
    elif len(tti_found) == 0:

        for idx in tool_found:
            l.append({'class':int(r.boxes.cls[idx].cpu().detach().numpy()) , 'mask': r.masks.data[idx].cpu().detach().numpy()})
        return l
    
    res = []
    
    for idx_tti in tti_found:
        for idx_tool in tool_found:
            # tissue_mask = masks[idx_tti].bool() & (~masks[idx_tool].bool())
            # tool_mask = masks[idx_tool].bool()
            
            tissue_mask = masks[idx_tti]
            tool_mask = masks[idx_tool]
            
            tti_dict = {'class': int(classes[idx_tti].cpu().detach().numpy()) , 'mask' : tissue_mask.int().cpu().detach().numpy()}
            tool_dict = {'class': int(classes[idx_tool].cpu().detach().numpy()) , 'mask':  tool_mask.int().cpu().detach().numpy()}
            
            
            
            
            already_has_tti = False
            for elem in res:
                if elem['class'] == tti_dict['class'] and np.array_equal(elem['mask'], tti_dict['mask']):
                    already_has_tti = True
                    break

            if not already_has_tti:
                res.append(tti_dict)

          
            already_has_tool = False
            for elem in res:
                if elem['class'] == tool_dict['class'] and np.array_equal(elem['mask'], tool_dict['mask']):
                    already_has_tool = True
                    break
            if not already_has_tool:
                res.append(tool_dict)

    return res

tool_classes = list(range(0, 7))

def find_tool_tissue_pairs(detections: list[dict]):
    
    tools = [d for d in detections if d['class'] in tool_classes]
    tissues = [d for d in detections if d['class'] not in tool_classes]
    pairs = []
    for s in tools:
        for o in tissues:
            pairs.append({'tool': s, 'tissue': o})
    return pairs


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



def end_to_end_pipeline(image, yolo_model, depth_model, tti_classifier, device, depth_map_required = True):
    # Step 1: YOLOv11-seg and depth estimation
  
    detections = yolo_inference(yolo_model, image)

    # depth_map = depth_model(image)
    # depth_map = np.array(depth_model(image)["depth"])
    
    if depth_map_required:
        depth_map = np.array(depth_model(Image.fromarray(image))["depth"])
    else:
        depth_map = None

    # Step 2: Pairing
    pairs = find_tool_tissue_pairs(detections)

    # image = cv2.imread(image,cv2.IMREAD_COLOR)
   
    
    tti_predictions = []

    for pair in pairs:
        tool_mask = pair['tool']['mask']
        tissue_mask = pair['tissue']['mask']
        roi = extract_union_roi(image, tool_mask, tissue_mask, depth_map)
        if roi is None:
            return None , None
        # Prepare input for ROI classifier
        roi_tensor = torch.from_numpy(roi).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        roi_tensor = F.interpolate(roi_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        roi_tensor = roi_tensor.to(device)
        
        # mean_rgb = [0.485, 0.456, 0.406]
        # std_rgb  = [0.229, 0.224, 0.225]
        # mean_t = torch.tensor(mean_rgb, device=device).view(1, 3, 1, 1)
        # std_t  = torch.tensor(std_rgb, device=device).view(1, 3, 1, 1)
        # roi_tensor[:, :3, :, :] = (roi_tensor[:, :3, :, :] - mean_t) / std_t
       
        tti_classifier.eval()
        with torch.no_grad():
            tti_logits = tti_classifier(roi_tensor)
            # tti_logits = torch.sigmoid(tti_classifier(roi_tensor))
            # print(tti_logits)
       
            # tti_logits = tti_logits.item()
            tti_class = torch.argmax(tti_logits, dim=1).item()
            tti_score = torch.softmax(tti_logits, dim=1).max().item()
            # if tti_logits >= 0.5:
            #     tti_class = 1
            # else:
            #     tti_class = 0
            # tti_score = tti_logits

        # Save ROI result
        tti_predictions.append({
            'tool': pair['tool'],
            'tissue': pair['tissue'],
            'tti_class': tti_class,
            'tti_score': tti_score
        })

    return detections, tti_predictions

