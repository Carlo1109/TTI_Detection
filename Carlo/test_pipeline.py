import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from torchvision.models import resnet18
from Model import CNN_TCN_Classifier

WINDOW_RAD = 3
T          = WINDOW_RAD*2 + 1          # 7
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS_PATH  = '../Dataset/video_dataset/labels/test/'
VIDEOS_PATH  = '../Dataset/video_dataset/videos/test/'
IMAGES_TEST  = '../Dataset/evaluation/images/'
LABELS_TEST  = '../Dataset/evaluation/labels/'

YOLO_WEIGHTS = '../Common Code/runs_OLD_DATASET/segment/train/weights/best.pt'
TCN_WEIGHTS  = 'model_TCN.pt'          

VIDEOS_LIST = sorted(os.listdir(VIDEOS_PATH))

def to_tool_id(name):
    if name is None:
        return 0
    name_to_id = {
        'unknown_tool': 0,
        'dissector': 1,
        'scissors': 2,
        'suction': 3,
        'grasper 3': 4,
        'harmonic': 5,
        'grasper': 6,
        'bipolar': 7,
        'grasper 2': 8,
        'cautery (hook, spatula)': 9,
        'ligasure': 10,
        'stapler': 11,
    }
    name = name.lower()
    return name_to_id.get(name, 0)

def to_tti_id(name):
    if name is None:
        return 12
    name_to_id = {
        'unknown_tti': 12,
        'coagulation': 13,
        'other': 14,
        'retract and grab': 15,
        'blunt dissection': 16,
        'energy - sharp dissection': 17,
        'staple': 18,
        'retract and push': 19,
        'cut - sharp dissection': 20,
    }
    name = name.lower()
    return name_to_id.get(name, 12)

_vid_re = re.compile(r"video(\d+)_frame(\d+)\.png")

def parse_img_name(fname):
    m = _vid_re.match(fname)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def _load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_count

def _load_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Failed to read frame {frame_idx}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

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
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32)

def select_best_instance(result, want_id, is_tool):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None, None
    names = result.names if hasattr(result, "names") else result.model.names
    cls  = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    best_i, best_conf, best_name = None, -1.0, None
    for i, c in enumerate(cls):
        name = names[int(c)]
        mapped = to_tool_id(name) if is_tool else to_tti_id(name)
        if mapped == want_id and conf[i] > best_conf:
            best_conf, best_i, best_name = conf[i], i, name
    return best_i, best_name

def build_seq(video_path, center_idx, tool_poly, tissue_poly, depth_pipe):
    cap, frame_count = _load_video(video_path)

    center_idx = int(np.clip(center_idx, 0, frame_count-1))
    frame_c = _load_frame(cap, center_idx)
    depth_map = np.array(depth_pipe(frame_c)["depth"])

    start = max(0, center_idx - WINDOW_RAD)
    end   = min(frame_count - 1, center_idx + WINDOW_RAD)

    frames = []
    for t in range(start, end + 1):
        f = _load_frame(cap, t)
        img = np.array(f)
        roi = extract_union_roi(img, tool_poly, tissue_poly, depth_map)
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        frames.append(np.transpose(roi, (2,0,1)).astype(np.float32) / 255.0)

    while len(frames) < T:
        if start > 0 and len(frames) < (center_idx - start) + 1 + WINDOW_RAD:
            frames.insert(0, frames[0])
        else:
            frames.append(frames[-1])

    cap.release()
    x = np.stack(frames[:T])
    return torch.from_numpy(x).unsqueeze(0)  



def predict_for_image(img_fname, model_tcn, yolo, depth_pipe):
    vid_idx, frm_idx = parse_img_name(img_fname)
    if vid_idx is None:
        return [], []

    if vid_idx < 0 or vid_idx >= len(VIDEOS_LIST):
        return [], []
    video_path = os.path.join(VIDEOS_PATH, VIDEOS_LIST[vid_idx])

    lab_fname = img_fname.replace('.png', '.txt')
    gt_pairs = {}  
    with open(os.path.join(LABELS_TEST, lab_fname), 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            is_tti = int(s[0])
            tool_id = int(s[2])
            tti_id  = int(s[4:])
            gt_pairs[(tool_id, tti_id)] = is_tti

    cap_tmp, frame_count = _load_video(video_path)
    cap_tmp.release()
    if frm_idx >= frame_count:
        return [], []

    cap, _ = _load_video(video_path)
    center_frame = _load_frame(cap, frm_idx)
    img_cv = cv2.cvtColor(np.array(center_frame), cv2.COLOR_RGB2BGR)
    res = yolo(img_cv, verbose=False)[0]
    H, W = img_cv.shape[:2]
    cap.release()

    y_true, y_pred = [], []
    model_tcn.eval()
    with torch.no_grad():
        for (tool_id, tti_id), gt in gt_pairs.items():
            i_tool, _ = select_best_instance(res, tool_id, is_tool=True)
            i_tti,  _ = select_best_instance(res, tti_id,  is_tool=False)
            if i_tool is None or i_tti is None:
                continue

            tool_poly   = polygon_from_mask_or_box(res, i_tool, W, H)
            tissue_poly = polygon_from_mask_or_box(res, i_tti,  W, H)

            x = build_seq(video_path, frm_idx, tool_poly, tissue_poly, depth_pipe).to(DEVICE)  

            p = model_tcn(x).squeeze(1).item()  
            y_pred.append(1 if p >= 0.5 else 0)
            y_true.append(int(gt))

    return y_true, y_pred

def generate_predictions_temporal(model_tcn, yolo, depth_pipe):
    images = sorted([f for f in os.listdir(IMAGES_TEST) if f.endswith('.png')])
    all_true, all_pred = [], []
    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img}")
        yt, yp = predict_for_image(img, model_tcn, yolo, depth_pipe)
        all_true.extend(yt)
        all_pred.extend(yp)
    return all_true, all_pred

if __name__ == "__main__":
    yolo = YOLO(YOLO_WEIGHTS)  
    depth = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if torch.cuda.is_available() else -1
    )

    model_tcn = CNN_TCN_Classifier(tcn_channels=[256,128], sequence_length=T, num_classes=1, pretrained=False)
    model_tcn.load_state_dict(torch.load(TCN_WEIGHTS, map_location=DEVICE))
    model_tcn.to(DEVICE)

    y_true, y_pred = generate_predictions_temporal(model_tcn, yolo, depth)

    print("N =", len(y_true))
    if len(y_true) == 0:
        print("Nessuna coppia (tool, tti) valutabile trovata. Controlla mapping classi o detection YOLO.")
    else:
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("F1 macro:", f1_score(y_true, y_pred, average='macro'))
        print("F1 weighted:", f1_score(y_true, y_pred, average='weighted'))
        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
        print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))