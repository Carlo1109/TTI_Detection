import os
import re
import json
import cv2
import numpy as np
import torch

from models.VIT import ROIClassifierViT
from transformers import pipeline
from ultralytics import YOLO
from draw_pipe_output import depth_treshold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from pipeline import end_to_end_pipeline
import time

import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

TEST_VIDEOS_DIR  = "./video_dataset/videos/val/"
TEST_LABELS_DIR  = "./video_dataset/labels/val/"
YOLO_WEIGHTS = "./runs/segment/train/weights/best.pt"
TCN_WEIGHTS  = "./model_TCN_V4.pt"
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', name).lower()

def _load_video(path):
    cap = cv2.VideoCapture(path)
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def _load_frame(cap, idx, rgb=True):
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
            
            




def found_objects(tool_list, tti_list, classes) -> list | list:
    
    tool_found = []
    tti_found = []
    
    for elem in range(len(classes)):
        if classes[elem] in tool_list:
            tool_found.append(elem)
        if classes[elem] in tti_list:
            tti_found.append(elem)
            
    return tool_found, tti_found



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


   
         

def test_with_intersection(with_vit : bool = False):
    """Test TTI detection using depth_treshold() for predictions.
    
    Evaluates each (tool_id, tti_id) pair from GT labels:
    - Extract GT pairs from labels with is_tti values
    - Run depth_treshold() to get predicted pairs
    - Compare pair-by-pair (like evaluate.py does)
    - Report accuracy, precision, recall, F1
    """
    yolo = load_models()
    videos = [v for v in os.listdir(TEST_VIDEOS_DIR) if not v.startswith('.')]
    
    depth = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tti_class = ROIClassifierViT(2)
    tti_class.load_state_dict(torch.load('./models/ViT_no_depth.pt',map_location=device))
    tti_class.to(device)
    
    y_true = []
    y_pred = []
    total_frames = 0
    processed_frames = 0
    total_pairs = 0
    predicted_pairs = 0
    gt_val_0 = 0
    bad_frames = []
    
    metrics = {
        'gt_tti_0': 0,
        'gt_tti_1': 0,
        'pred_tti_0_in_gt_1': 0,
        'pred_tti_1_in_gt_0': 0,
        'pred_tti_1_in_gt_1': 0,
        'pred_tti_0_in_gt_0': 0,
        'pred_tti_0_no_gt': 0,
    }
        
    
    
    # Error counters
    tti_mismatch_errors = 0      # classes match, but TTI/no-TTI differs
    class_mismatch_errors = 0    # classes mismatch (pred vs GT)

    for vi, vid in enumerate(videos, 1):
        print(f"[INFO] Processing video {vi}/{len(videos)}: {vid}")
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
            
            skip_frames = np.load('bad_frames_vit_depth.npy')
            # print(skip_frames)
            if (vid, idx) in skip_frames:
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
                if not with_vit:
                    # time_pre = time.time()
                    detections, tti_predictions = depth_treshold(frame_bgr, yolo)
                    # time_post = time.time()
                    # print(f"Depth estimation time: {time_post - time_pre:.2f} seconds")
                else: 
                
                    # time_pre = time.time()
                    detections, tti_predictions = end_to_end_pipeline(frame_bgr, yolo, depth, tti_class, device,None)
                    if tti_predictions is None and detections is None:
                        bad_frames.append((vid, idx))
                    # time_post = time.time()
                    # print(f"Depth estimation time: {time_post - time_pre:.2f} seconds")
                    
                
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
                    if gt_val == 0:
                        metrics['gt_tti_0'] += 1
                    else:
                        metrics['gt_tti_1'] += 1
                        
                    total_pairs += 1
                    if key in pred_pairs:
                        pred_val = pred_pairs[key]
                        if pred_val == 1 and gt_val == 1:
                            metrics['pred_tti_1_in_gt_1'] += 1
                        elif pred_val == 0 and gt_val == 1:
                            metrics['pred_tti_0_in_gt_1'] += 1
                        elif pred_val == 1 and gt_val == 0:
                            metrics['pred_tti_1_in_gt_0'] += 1
                        elif pred_val == 0 and gt_val == 0:
                            metrics['pred_tti_0_in_gt_0'] += 1
                            
                        
                        if pred_val != gt_val:
                            y_true.append(1)
                            y_pred.append(0)
                            tti_mismatch_errors += 1
                            # Visualizza e salva immagine con maschere in caso di mismatch TTI/no-TTI
                            try:
                                match = next((p for p in tti_predictions if p['tool']['class'] == key[0] and p['tissue']['class'] == key[1]), None)
                                if match is not None:
                                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                    H, W = frame_rgb.shape[:2]
                                    alpha = 0.5
                                    vis = frame_rgb.astype(np.float32).copy()
                                    tmask = cv2.resize(match['tool']['mask'].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                                    smask = cv2.resize(match['tissue']['mask'].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                                    tool_color = np.array([200, 0, 0], dtype=np.float32)
                                    tissue_color = np.array([0, 220, 220], dtype=np.float32)
                                    vis[tmask] = (1 - alpha) * vis[tmask] + alpha * tool_color
                                    vis[smask] = (1 - alpha) * vis[smask] + alpha * tissue_color
                                    combined = np.logical_or(tmask, smask).astype(np.uint8)
                                    if combined.sum() > 0:
                                        x, y, w, h = cv2.boundingRect(combined)
                                        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                        label = f"Tool={key[0]} TTI={key[1]} GT={gt_val} Pred={pred_val}"
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        font_scale = 0.6
                                        text_thickness = 1
                                        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                                        ty_top = y - th - baseline - 5
                                        ty_bot = y
                                        if ty_top < 0:
                                            ty_top = y + h + 5
                                            ty_bot = y + h + th + baseline + 10
                                        cv2.rectangle(vis, (x, ty_top), (x + tw + 5, ty_bot), (0, 220, 220), -1)
                                        cv2.putText(vis, label, (x + 2, ty_top + th + baseline - 2), font, font_scale, (0, 0, 0), text_thickness)
                                    out_dir = os.path.join("./img_output", "mismatch")
                                    os.makedirs(out_dir, exist_ok=True)
                                    out_name = f"{os.path.splitext(vid)[0]}_frame{idx}_tool{key[0]}_tti{key[1]}_mismatch.png"
                                    cv2.imwrite(os.path.join(out_dir, out_name), cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR))
                            except Exception:
                                pass
                    else:
                        pred_val = 0  
                        continue
                    
                    # y_true.append(gt_val)
                    # y_pred.append(pred_val)
                    y_true.append(1)
                    y_pred.append(1)
                
                
                for key, pred_val in pred_pairs.items():
                    predicted_pairs += 1
                    if key not in gt_pairs:
                        if pred_val == 0:
                            metrics['pred_tti_0_no_gt'] += 1
                        class_mismatch_errors += 1
                        y_pred.append(0)   
                        y_true.append(1)   

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
    print("TTI DETECTION RESULTS (Intersection NO Depth):")
    print("="*60)
    print(f"Total frames: {total_frames} | Processed frames: {processed_frames}")
    print(f"Samples (pairs): {len(y_true)} | Total pairs: {total_pairs}")
    print(f"Preicted pairs {predicted_pairs}")
    print(f"Errors: TTI/no-TTI mismatch = {tti_mismatch_errors}, Class mismatch = {class_mismatch_errors}")
    print("metrics:  ", metrics)
    # np.save('bad_frames_vit_depth.npy', np.array(bad_frames))
    # print(f"  GT TTI=1: {sum(y_true)}")
    # print(f"  GT TTI=0: {len(y_true) - sum(y_true)}")
    print(f"\nAccuracy:  {acc:.4f}")
    # print(f"Precision: {prec:.4f}")
    # print(f"Recall:    {rec:.4f}")
    # print(f"F1 Score:  {f1:.4f}")
    # print(f"Balanced Accuracy: {bacc:.4f}")
    # print(f"\nConfusion Matrix:")
    # print("="*60)
    # print(f"GT TTI=0: {gt_val_0}")


if __name__ == "__main__":
    test_with_intersection(with_vit=False)