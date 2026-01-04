import os
import re
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18
from transformers import pipeline
from ultralytics import YOLO
from TCN_model import CNN_TCN_Classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score

TEST_VIDEOS_DIR  = "./video_dataset/videos/test/"
TEST_LABELS_DIR  = "./video_dataset/labels/test/"
YOLO_WEIGHTS = "./runs/segment/train/weights/best.pt"
TCN_WEIGHTS  = "./model_TCN_V4.pt"
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', name).lower()

def _load_video(path):
    cap = cv2.VideoCapture(path)
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def _load_frame(cap, idx, rgb=False):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        raise ValueError(f"Failed to read frame {idx}")
    if rgb:
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    return fr

def load_models():
    yolo = YOLO(YOLO_WEIGHTS)
    return yolo

def expand_mask(mask, pixels):
    kernel_size = 2 * pixels + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return expanded
            
            
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
    for cls_id, m in zip(classes, masks):
        out.append({'class': int(cls_id), 'mask': m})
    return out

tool_classes = list(range(0, 8))

# Map label strings to YOLO class ids (post-remap per class_mapping.txt)
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

def _map_instrument(name: str) -> int | None:
    return INSTRUMENT_NAME_TO_ID.get(name.strip().lower()) if name is not None else None

def _map_tti(name: str) -> int | None:
    return TTI_NAME_TO_ID.get(name.strip().lower()) if name is not None else None

def _extract_gt_classes(objs) -> set:
    inst, tti = set(), set()
    for o in objs:
        if 'instrument_type' in o:
            cid = _map_instrument(o['instrument_type'])
            if cid is not None:
                inst.add(cid)
        if 'interaction_type' in o:
            cid = _map_tti(o['interaction_type'])
            if cid is not None:
                tti.add(cid)
    return inst | tti


def build_gt_pairs_dict(objs):
    """Extract ground truth (tool_id, tti_id) -> is_tti pairs from label objects.
    
    Returns dict like: {(tool_id, tti_id): 1 or 0}
    where 1 = interaction present, 0 = no interaction
    """
    d = {}
    L = len(objs)
    if L < 2:
        return d

    def get_tti_id_from_obj(o):
        tti_name = o.get('interaction_type', None)
        return _map_tti(tti_name)

    if L == 2:
        tool_name = None
        tti_id = None
        for o in objs:
            if 'is_tti' in o:
                if int(o.get('is_tti', 0)) == 1:
                    tti_id = get_tti_id_from_obj(o)
            else:
                tool_name = o.get('instrument_type')
        if tool_name is not None and tti_id is not None:
            tool_id = _map_instrument(tool_name)
            if tool_id is not None:
                d[(tool_id, tti_id)] = 1

    elif L == 3:
        tti_id = None
        inter_tool_name = None
        non_inter_tool_name = None
        inter_tool_name2 = None
        for o in objs:
            if 'is_tti' in o:
                if int(o.get('is_tti', 0)) == 1:
                    tti_id = get_tti_id_from_obj(o)
                    inter_tool_name = o.get('interaction_tool')
                else:
                    non_inter_tool_name = o.get('non_interaction_tool', None)
            else:
                inter_tool_name2 = o.get('instrument_type', None)
        if inter_tool_name is not None and tti_id is not None:
            tool_id = _map_instrument(inter_tool_name)
            if tool_id is not None:
                d[(tool_id, tti_id)] = 1
        if non_inter_tool_name is not None and tti_id is not None:
            tool_id = _map_instrument(non_inter_tool_name)
            if tool_id is not None:
                d[(tool_id, tti_id)] = 0
        if non_inter_tool_name is None and inter_tool_name2 is not None and tti_id is not None:
            if inter_tool_name2 != inter_tool_name:
                tool_id = _map_instrument(inter_tool_name2)
                if tool_id is not None:
                    d[(tool_id, tti_id)] = 0

    else:  # L > 3
        pos_tissues = []
        for o in objs:
            if 'is_tti' in o and int(o.get('is_tti', 0)) == 1:
                pos_tissues.append((o.get('interaction_tool'), get_tti_id_from_obj(o)))
        
        instruments = []
        for o in objs:
            if 'instrument_type' in o:
                instruments.append(o.get('instrument_type'))
        
        for inter_tool_name, tti_id in pos_tissues:
            if inter_tool_name is not None and tti_id is not None:
                tool_id = _map_instrument(inter_tool_name)
                if tool_id is not None:
                    d[(tool_id, tti_id)] = 1
                    
            for inst_name in instruments:
                if inst_name != inter_tool_name:
                    tool_id = _map_instrument(inst_name)
                    if tool_id is not None and tti_id is not None:
                        d[(tool_id, tti_id)] = 0

    return d

def find_tool_tissue_pairs(detections: list[dict]):
    
    tools = [d for d in detections if d['class'] in tool_classes]
    tissues = [d for d in detections if d['class'] not in tool_classes]
    pairs = []
    for s in tools:
        for o in tissues:
            pairs.append({'tool': s, 'tissue': o})
    return pairs



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
  

def depth_treshold(image, yolo_model):
    # Step 1: YOLOv11-seg and depth estimation
  
    detections = yolo_inference(yolo_model, image)
    # Step 2: Pairing
    pairs = find_tool_tissue_pairs(detections)

    # image = cv2.imread(image,cv2.IMREAD_COLOR)

    tti_predictions = []

    for pair in pairs:
        tool_mask = pair['tool']['mask']
        tissue_mask = pair['tissue']['mask']
        roi = extract_union_roi(image, tool_mask, tissue_mask)
        if roi is None:
            return [] ,[]
        

        H_full, W_full = image.shape[:2]

        tool_mask_expanded = expand_mask(tool_mask, pixels=5)
        tissue_mask_expanded = expand_mask(tissue_mask, pixels=5)

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
            'tti_score': 1
        })

    return detections, tti_predictions
         
         

def test_with_intersection():
    """Test TTI detection using depth_treshold() for predictions.
    
    Evaluates each (tool_id, tti_id) pair from GT labels:
    - Extract GT pairs from labels with is_tti values
    - Run depth_treshold() to get predicted pairs
    - Compare pair-by-pair (like evaluate.py does)
    - Report accuracy, precision, recall, F1
    """
    yolo = load_models()
    videos = [v for v in os.listdir(TEST_VIDEOS_DIR) if not v.startswith('.')]

    y_true = []
    y_pred = []
    total_frames = 0
    processed_frames = 0
    total_pairs = 0

    for vi, vid in enumerate(videos, 1):
        vpath = os.path.join(TEST_VIDEOS_DIR, vid)
        cap, fcount = _load_video(vpath)
        key = normalize(os.path.splitext(vid)[0])
        jpath = next((os.path.join(TEST_LABELS_DIR, f)
                      for f in os.listdir(TEST_LABELS_DIR)
                      if normalize(os.path.splitext(f)[0]) == key), None)
        if jpath is None:
            cap.release()
            continue

        labels = json.load(open(jpath, 'r')).get('labels', {})

        for idx_s, objs in labels.items():
            idx = int(idx_s)
            if idx < 0 or idx >= fcount:
                continue

            total_frames += 1

            # Extract GT pairs: {(tool_id, tti_id): is_tti}
            gt_pairs = build_gt_pairs_dict(objs)
            if len(gt_pairs) == 0:
                continue

            processed_frames += 1

            # Load frame
            frame_bgr = _load_frame(cap, idx, rgb=False)

            # Get predictions using depth_treshold
            try:
                detections, tti_predictions = depth_treshold(frame_bgr, yolo)
                
                # Build predicted pairs dict: {(tool_id, tti_id): tti_class}
                pred_pairs = {}
                for p in tti_predictions:
                    tool_id = p['tool']['class']
                    tti_id = p['tissue']['class']
                    tti_val = p['tti_class']
                    key = (tool_id, tti_id)
                    pred_pairs[key] = tti_val
                
                # Compare each GT pair
                for key, gt_val in gt_pairs.items():
                    total_pairs += 1
                    pred_val = pred_pairs.get(key, 0)  # Default to 0 if not predicted
                    y_true.append(gt_val)
                    y_pred.append(pred_val)

            except Exception as e:
                print(f"    [WARN] Video {vi} Frame {idx} error: {e}")
                continue

        cap.release()

    # Report results
    if len(y_true) == 0:
        print("\n[RESULT] No TTI pairs evaluated")
        return

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    print("\n" + "="*60)
    print("TTI DETECTION RESULTS (Depth-based Intersection)")
    print("="*60)
    print(f"Total frames: {total_frames} | Processed frames: {processed_frames}")
    print(f"Samples (pairs): {len(y_true)} | Total predicted pairs: {total_pairs}")
    print(f"  GT TTI=1: {sum(y_true)}")
    print(f"  GT TTI=0: {len(y_true) - sum(y_true)}")
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print("="*60)
    print(f"\nConfusion Matrix:")
    print(cm)
    print("="*60)


if __name__ == "__main__":
    test_with_intersection()