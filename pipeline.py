import torch
import cv2
import torchvision.models as models
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
from Model import ROIClassifier


def load_yolo_model(model_path):
    # Load your YOLOv11-seg model (pseudo-code, depends on repo)
    model = YOLO(model_path)
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
    
    output = model.predict(image)

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
    if len(result[0].boxes.cls) < 2 :
        return [{'class':int(r.boxes.cls[0].cpu().detach().numpy()) , 'mask': r.masks.data[0].cpu().detach().numpy()}]
    
    tool_list = list(range(0, 12))
    tti_list = list(range(12, 21))

    
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
            tissue_mask = masks[idx_tti].bool() & (~masks[idx_tool].bool())
            tool_mask = masks[idx_tool].bool()
            tti_dict = {'class': int(classes[idx_tti].cpu().detach().numpy()) , 'mask' : tissue_mask.int().cpu().detach().numpy()}
            tool_dict = {'class': int(classes[idx_tool].cpu().detach().numpy()) , 'mask':  tool_mask.int().cpu().detach().numpy()}
            if tti_dict not in res:
                res.append(tti_dict)
            if tool_dict not in res:
                res.append(tool_dict)
            
    return res

tool_classes = list(range(0, 12))

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

    return roi


def end_to_end_pipeline(image, yolo_model, depth_model, tti_classifier, device):
    # Step 1: YOLOv11-seg and depth estimation
    detections = yolo_inference(yolo_model, image)
    
    # depth_map = depth_model(image)
    depth_map = np.array(depth_model(image)["depth"])

    # Step 2: Pairing
    pairs = find_tool_tissue_pairs(detections)

    image = cv2.imread(image,cv2.IMREAD_COLOR)
   
    
    tti_predictions = []
    print(len(pairs))
    for pair in pairs:
        tool_mask = pair['tool']['mask']
        tissue_mask = pair['tissue']['mask']
        roi = extract_union_roi(image, tool_mask, tissue_mask, depth_map)

        # Prepare input for ROI classifier
        roi_tensor = torch.from_numpy(roi).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        roi_tensor = roi_tensor.to(device)

        with torch.no_grad():
            tti_logits = tti_classifier(roi_tensor)
            tti_class = torch.argmax(tti_logits, dim=1).item()
            tti_score = torch.softmax(tti_logits, dim=1).max().item()

        # Save ROI result
        tti_predictions.append({
            'tool': pair['tool'],
            'tissue': pair['tissue'],
            'tti_class': tti_class,
            'tti_score': tti_score
        })

    return detections, tti_predictions


if __name__ == "__main__":
    model = load_yolo_model('./runs/segment/train/weights/best.pt')
    image = './Dataset/dataset/images/test/video0114_frame0059.png'
    
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tti_class = ROIClassifier(2)
    tti_class.load_state_dict(torch.load('ROImodel.pt',map_location=device))
    tti_class.to(device)
    detection , tti_predictions  = end_to_end_pipeline(image,model,pipe,tti_class,device)
    # print(detection)
    print()
    print(tti_predictions)
    # print(find_tool_tissue_pairs(pred))