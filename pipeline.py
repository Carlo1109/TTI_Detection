import torch
import cv2
import torchvision.models as models
import torch.nn as nn
import numpy as np
from ultralytics import YOLO



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
    
    if len(result[0].boxes.cls) < 2 :
        return {'class':r.boxes.cls[0] , 'mask': r.masks.data[0]}
    
    tool_list = list(range(0, 12))
    tti_list = list(range(12, 21))

    r = result[0]
    
    classes = r.boxes.cls
    masks = r.masks.data
    
    tool_found, tti_found = found_objects(tool_list, tti_list, classes)
    print(tool_found)
    print(tti_found)
    l = []
    
    if len(tool_found) == 0:
        for idx in tti_found:
            l.append({'class':int(r.boxes.cls[idx].cpu().detach().numpy()) , 'mask': r.masks.data[idx]})
        return l
    
    elif len(tti_found) == 0:

        for idx in tool_found:
            l.append({'class':int(r.boxes.cls[idx].cpu().detach().numpy()) , 'mask': r.masks.data[idx]})
        return l
    
    res = []
    
    for idx_tti in tti_found:
        for idx_tool in tool_found:
            tissue_mask = masks[idx_tti].bool() & (~masks[idx_tool].bool())
            tool_mask = masks[idx_tool].bool()
            res.append({'class': int(classes[idx_tti].cpu().detach().numpy()) , 'mask' : tissue_mask.int()})
            res.append({'class': int(classes[idx_tool].cpu().detach().numpy()) , 'mask':  tool_mask.int()})
            
    return res



tool_classes = {
    0 : 'other', 
}

def find_tool_tissue_pairs(detections: list[dict]):
    tools = [d for d in detections if d['class'] in tool_classes]
    tissues = [d for d in detections if d['class'] not in tool_classes]
    pairs = []
    for s in tools:
        for o in tissues:
            pairs.append({'tool': s, 'tissue': o})
    return pairs


def extract_union_roi(image, tool_mask, tissue_mask, depth_map=None):
    combined_mask = (tool_mask + tissue_mask).clip(0, 1).astype('uint8')
    x, y, w, h = cv2.boundingRect(combined_mask)
    roi = image[y:y+h, x:x+w]

    if depth_map is not None:
        depth_roi = depth_map[y:y+h, x:x+w]
        roi = np.concatenate([roi, depth_roi[..., None]], axis=-1)  # add depth as extra channel

    return roi


class ROIClassifier(nn.Module):
    def __init__(self, num_hoi_classes):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True) # I would prefer efficientnet
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, num_hoi_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out
    

def end_to_end_pipeline(image, yolo_model, depth_model, tti_classifier, device):
    # Step 1: YOLOv11-seg and depth estimation
    detections = yolo_inference(yolo_model, image)
    depth_map = depth_model(image)

    # Step 2: Pairing
    pairs = find_tool_tissue_pairs(detections)

    tti_predictions = []
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
    image = './Dataset/dataset/images/test/video0014_frame0011.png'
    pred = yolo_inference(model, image)
    print(pred)
    # print(find_tool_tissue_pairs(pred))