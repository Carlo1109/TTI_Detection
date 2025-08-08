import re
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from Model import CNN_TCN_Classifier  


WINDOW_RAD = 3
T = WINDOW_RAD*2 + 1
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_vid_re = re.compile(r"video(\d+)_frame(\d+)\.png")

def parse_img_name(fname):
    m = _vid_re.match(fname)
    if not m: 
        return None, None
    return int(m.group(1)), int(m.group(2))

def polygon_from_mask_or_box(result, i, W, H):
    if getattr(result, "masks", None) is not None and result.masks is not None and len(result.masks.xy) > i:
        poly = np.array(result.masks.xy[i], dtype=np.float32)  
        poly[:, 0] = np.clip(poly[:, 0], 0, W-1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H-1)
        return poly.astype(np.int32)
    xyxy = result.boxes.xyxy[i].cpu().numpy().astype(int)  
    x1, y1, x2, y2 = xyxy
    x1 = int(np.clip(x1, 0, W-1)); x2 = int(np.clip(x2, 0, W-1))
    y1 = int(np.clip(y1, 0, H-1)); y2 = int(np.clip(y2, 0, H-1))
    rect = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32)
    return rect

def select_best_instance(result, want_id, is_tool, to_tool_id, to_tti_id):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None, None
    names = result.names if hasattr(result, "names") else result.model.names
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    best_i, best_conf, best_name = None, -1, None
    for i, c in enumerate(cls):
        name = names[int(c)]
        mapped = to_tool_id(name) if is_tool else to_tti_id(name)
        if mapped == want_id and conf[i] > best_conf:
            best_conf = conf[i]; best_i = i; best_name = name
    return best_i, best_name

def extract_union_roi(image, tool_mask, tissue_mask, depth_map=None):
    H, W = image.shape[:2]
    bin_tool = np.zeros((H, W), np.uint8)
    if tool_mask is not None and tool_mask.size:
        cv2.fillPoly(bin_tool, [tool_mask], 1)
    bin_tissue = np.zeros((H, W), np.uint8)
    if tissue_mask is not None and tissue_mask.size:
        cv2.fillPoly(bin_tissue, [tissue_mask], 1)
    combined = np.clip(bin_tool + bin_tissue, 0, 1).astype(np.uint8)
    if combined.max() == 0:
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = cv2.boundingRect(combined)
    roi = image[y:y+h, x:x+w]
    if depth_map is not None:
        d = depth_map[y:y+h, x:x+w]
        roi = np.concatenate([roi, d[..., None]], axis=-1)
    mask_crop = (bin_tool[y:y+h, x:x+w] | bin_tissue[y:y+h, x:x+w])[..., None] * 255
    roi = np.concatenate([roi, mask_crop.astype(np.uint8)], axis=-1)
    return roi

def build_seq(video_path, center_idx, tool_poly, tissue_poly, depth_pipe):
    cap, _ = _load_video(video_path)
    frame_c = _load_frame(cap, center_idx)
    depth_map = np.array(depth_pipe(frame_c)["depth"])
    seq = []
    for t in range(center_idx - WINDOW_RAD, center_idx + WINDOW_RAD + 1):
        f = _load_frame(cap, t)
        img = np.array(f)
        roi = extract_union_roi(img, tool_poly, tissue_poly, depth_map)
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        seq.append(np.transpose(roi, (2,0,1)).astype(np.float32) / 255.0)
    cap.release()
    x = np.stack(seq)  
    return torch.from_numpy(x).unsqueeze(0)  

def predict_for_image(img_fname, model_tcn, yolo, depth_pipe, to_tool_id, to_tti_id):
    vid_idx, frm_idx = parse_img_name(img_fname)
    if vid_idx is None:
        return [], []

    videos = os.listdir(VIDEOS_PATH)
    video_path = os.path.join(VIDEOS_PATH, videos[vid_idx])

    lab_fname = img_fname.replace('.png', '.txt')
    gt_pairs = {}  
    with open(os.path.join(LABELS_TEST, lab_fname), 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s: 
                continue
            is_tti = int(s[0])
            tool_id = int(s[2])
            tti_id = int(s[4:])
            gt_pairs[(tool_id, tti_id)] = is_tti

    cap, _ = _load_video(video_path)
    center_frame = _load_frame(cap, frm_idx)
    img_cv = cv2.cvtColor(np.array(center_frame), cv2.COLOR_RGB2BGR)
    res = yolo(img_cv, verbose=False)[0]
    H, W = img_cv.shape[:2]

    y_true, y_pred = [], []
    model_tcn.eval()
    with torch.no_grad():
        for (tool_id, tti_id), gt in gt_pairs.items():
            i_tool, _ = select_best_instance(res, tool_id, is_tool=True,  to_tool_id=to_tool_id, to_tti_id=to_tti_id)
            i_tti,  _ = select_best_instance(res, tti_id,  is_tool=False, to_tool_id=to_tool_id, to_tti_id=to_tti_id)
            if i_tool is None or i_tti is None:
                continue
            tool_poly   = polygon_from_mask_or_box(res, i_tool, W, H)
            tissue_poly = polygon_from_mask_or_box(res, i_tti,  W, H)

            x = build_seq(video_path, frm_idx, tool_poly, tissue_poly, depth_pipe).to(DEVICE)  # (1,T,5,224,224)

            p = model_tcn(x).squeeze(1).sigmoid().item()  # (1,)
            y_pred.append(1 if p >= 0.5 else 0)
            y_true.append(int(gt))

    cap.release()
    return y_true, y_pred

def generate_predictions_temporal(model_tcn, yolo, depth_pipe, to_tool_id, to_tti_id):
    images = [f for f in os.listdir(IMAGES_TEST) if f.endswith('.png')]
    images.sort()
    all_true, all_pred = [], []
    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img}")
        yt, yp = predict_for_image(img, model_tcn, yolo, depth_pipe, to_tool_id, to_tti_id)
        all_true.extend(yt)
        all_pred.extend(yp)
    return all_true, all_pred


if __name__ == "__main__":
    yolo = YOLO('./runs_OLD_DATASET/segment/train/weights/best.pt')
    depth = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    model_tcn = CNN_TCN_Classifier(tcn_channels=[256,128], sequence_length=T, num_classes=1)
    model_tcn.load_state_dict(torch.load('model_TCN.pt', map_location=DEVICE))
    model_tcn.to(DEVICE)

    y_true, y_pred = generate_predictions_temporal(model_tcn, yolo, depth, to_tool_id, to_tti_id)

    print("N =", len(y_true))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 macro:", f1_score(y_true, y_pred, average='macro'))
    print("F1 weighted:", f1_score(y_true, y_pred, average='weighted'))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
