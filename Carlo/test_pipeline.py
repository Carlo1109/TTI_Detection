
VIDEOS_PATH  = "../Dataset/video_dataset/videos/test"        
LABELS_PATH  = "../Dataset/video_dataset/videos/test"        
TCN_WEIGHTS  = "model_TCN.pt"  


IMG_SIZE     = 224
SEQ_LEN      = 5
BATCH_SIZE   = 16
THRESHOLD    = 0.5
USE_CUDA     = True
DEPTH_DEV    = None   
MAX_FRAME    = None   
PRINT_EVERY  = 50     

import os, re, json
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix, classification_report
)

class CNN_TCN_Classifier(nn.Module):
    def __init__(self, tcn_channels=[256, 128], sequence_length=SEQ_LEN, num_classes=1, pretrained=True):
        super().__init__()
        backbone = resnet18(pretrained=pretrained)
        backbone.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            bias=False
        )
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))

        tcn_layers = []
        num_inputs = 512
        for i, out_ch in enumerate(tcn_channels):
            dilation = 2 ** i
            tcn_layers += [
                nn.Conv1d(
                    in_channels=num_inputs if i == 0 else tcn_channels[i-1],
                    out_channels=out_ch,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation
                ),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]
        self.tcn = nn.Sequential(*tcn_layers)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(tcn_channels[-1], num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)
        x = self.pool2d(x)          
        x = x.view(B, T, 512)      
        x = x.permute(0, 2, 1)    
        x = self.tcn(x)           
        x = self.pool1d(x)        
        out = self.classifier(x)  
        return out.squeeze(1)     

def normalize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', name).lower()

def _load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {path}")
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def _load_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Impossibile leggere frame {idx}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def parse_mask_string(mask_str, H, W):
    if not mask_str or not mask_str.strip():
        return np.zeros((0,2), np.int32)
    tokens = mask_str.strip().split()
    coords = []
    for i in range(0, (len(tokens)//2)*2, 2):
        try:
            x = int(float(tokens[i]) * W)
            y = int(float(tokens[i+1]) * H)
        except:
            continue
        coords.append((max(0, min(W-1, x)), max(0, min(H-1, y))))
    return np.array(coords, np.int32) if coords else np.zeros((0,2), np.int32)

def get_polygon(poly_json):
    s = ''
    for v in poly_json.values():
        s += f" {v['x']} {v['y']}"
    return s

def extract_union_roi(img_rgb, tool_poly_pts, tissue_poly_pts, depth_map):
    H, W = img_rgb.shape[:2]
    bin_tool = np.zeros((H, W), np.uint8)
    if tool_poly_pts.size:
        cv2.fillPoly(bin_tool, [tool_poly_pts], 1)
    bin_tissue = np.zeros((H, W), np.uint8)
    if tissue_poly_pts.size:
        cv2.fillPoly(bin_tissue, [tissue_poly_pts], 1)
    union = np.clip(bin_tool + bin_tissue, 0, 1).astype(np.uint8)
    if union.max() == 0:
        return None
    x, y, w, h = cv2.boundingRect(union)
    roi_rgb = img_rgb[y:y+h, x:x+w]
    d = depth_map[y:y+h, x:x+w]
    if d.ndim == 2:
        d = d[..., None]
    merged_mask = (union[y:y+h, x:x+w][..., None] * 255).astype(np.uint8)
    roi = np.concatenate([roi_rgb, d, merged_mask], axis=-1) 
    return roi

def resize_norm_5ch(roi, img_size):
    roi = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    rgb = roi[..., :3] / 255.0
    depth = roi[..., 3:4]          
    mask = roi[..., 4:5] / 255.0
    roi5 = np.concatenate([rgb, depth, mask], axis=-1)            
    roi5 = np.transpose(roi5, (2, 0, 1))                          
    return roi5

def build_pairs_for_idx(objs, H, W):

    pairs = []
    len_dict = len(objs)

    if len_dict == 2:
        tissue_poly = None
        tool_poly = None
        for j in range(len_dict):
            if not bool(objs[j]):
                continue
            if 'is_tti' in objs[j]:
                if objs[j]['is_tti'] == 1:
                    tissue_poly = parse_mask_string(get_polygon(objs[j]['tti_polygon']), H, W)
                else:
                    continue
            else:
                tool_poly = parse_mask_string(get_polygon(objs[j]['instrument_polygon']), H, W)
        if tissue_poly is not None and tool_poly is not None:
            pairs.append((1, tissue_poly, tool_poly))

    elif len_dict == 3:
        tissue_poly = None
        tool_poly = None
        non_tool_poly = None
        for j in range(len_dict):
            if not bool(objs[j]):
                continue
            if 'is_tti' in objs[j]:
                if objs[j]['is_tti'] == 1:
                    tissue_poly = parse_mask_string(get_polygon(objs[j]['tti_polygon']), H, W)
                else:
                    non_tool_poly = parse_mask_string(get_polygon(objs[j]['instrument_polygon']), H, W)
            else:
                tool_poly = parse_mask_string(get_polygon(objs[j]['instrument_polygon']), H, W)

        if tissue_poly is not None and tool_poly is not None:
            pairs.append((1, tissue_poly, tool_poly))
        if tissue_poly is not None and non_tool_poly is not None:
            pairs.append((0, tissue_poly, non_tool_poly))

    else:
        
        pair_list = []  
        inter_list = []
        for j in range(len_dict):
            if not bool(objs[j]):
                continue
            if 'is_tti' in objs[j] and objs[j]['is_tti'] == 1:
                tissue_poly = parse_mask_string(get_polygon(objs[j]['tti_polygon']), H, W)
                interaction_tool_name = objs[j]['interaction_tool']
                inter_list.append((tissue_poly, interaction_tool_name))

        for tissue_poly, interaction_tool_name in inter_list:
            for j in range(len_dict):
                if not bool(objs[j]):
                    continue
                if 'is_tti' in objs[j]:
                    continue
                tool_poly = parse_mask_string(get_polygon(objs[j]['instrument_polygon']), H, W)
                tool_name_here = objs[j]['instrument_type']
                if tissue_poly is None or tool_poly is None:
                    continue
                x = 1 if (tool_name_here == interaction_tool_name) else 0
                pairs.append((x, tissue_poly, tool_poly))

    return pairs

def make_depth_pipeline():
    use_cuda = USE_CUDA and torch.cuda.is_available()
    if DEPTH_DEV is None:
        depth_dev = 0 if use_cuda else -1
    else:
        depth_dev = DEPTH_DEV if use_cuda else -1
    print(f"[Depth] Inizializzo Depth-Anything V2 Small su device {depth_dev}…")
    return pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=depth_dev
    )


def build_sequence(cap, idx, tissue_poly_pts, tool_poly_pts, depth_map):
    start = idx - (SEQ_LEN - 1)
    end = idx
    seq = []
    for t in range(start, end + 1):
        fr = _load_frame(cap, t)
        np_fr = np.array(fr)  # RGB
        roi = extract_union_roi(np_fr, tool_poly_pts, tissue_poly_pts, depth_map)
        if roi is None:
            return None
        roi5 = resize_norm_5ch(roi, IMG_SIZE)
        seq.append(roi5)
    return np.stack(seq, axis=0).astype(np.float32)  # [T,5,H,W]

def evaluate_on_the_fly():
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Device modello: {device}")

    model = CNN_TCN_Classifier(sequence_length=SEQ_LEN, pretrained=False).to(device)
    state = torch.load(TCN_WEIGHTS, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    depth_pipe = make_depth_pipeline()

    videos = sorted([v for v in os.listdir(VIDEOS_PATH) if not v.startswith(".")])
    total_pairs = 0
    skipped_pairs = 0

    y_true, y_pred, y_score = [], [], []

    xb_buf, yb_buf = [], []

    def flush_batch():
        nonlocal xb_buf, yb_buf, y_true, y_pred, y_score
        if not xb_buf:
            return
        xb = torch.from_numpy(np.stack(xb_buf, axis=0)).to(device)  
        yb = torch.tensor(yb_buf, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = model(xb).clamp(0.0, 1.0)  
        preds = (probs >= THRESHOLD).long().cpu().numpy().tolist()
        y_true.extend(yb.cpu().numpy().astype(int).tolist())
        y_pred.extend(preds)
        y_score.extend(probs.cpu().numpy().tolist())
        xb_buf, yb_buf = [], []

    processed_videos = 0
    for v in videos:
        video_path = os.path.join(VIDEOS_PATH, v)
        if not os.path.isfile(video_path):
            continue

        key_video = normalize(os.path.splitext(v)[0])
        matched_json = None
        for fname in os.listdir(LABELS_PATH):
            if normalize(os.path.splitext(fname)[0]) == key_video:
                matched_json = os.path.join(LABELS_PATH, fname)
                break
        if not matched_json:
            print(f"[WARNING] Nessun JSON per «{v}». Salto.")
            continue

        cap, fcount = _load_video(video_path)
        data = json.load(open(matched_json, 'r', encoding='utf-8'))
        labels = data['labels']

        frame_indices = [int(k) for k in labels.keys() if len(labels[k]) != 0]
        frame_indices.sort()

        for i_idx, idx in enumerate(frame_indices, 1):
            if MAX_FRAME is not None and idx > MAX_FRAME:
                continue
            if idx - (SEQ_LEN - 1) < 0 or idx >= fcount:
                continue

            objs = labels[str(idx)]
            len_dict = len(objs)
            if len_dict < 2:
                continue

            frame_center = _load_frame(cap, idx)
            W, H = frame_center.size
            depth_map = np.array(depth_pipe(frame_center)['depth']).astype(np.float32)

            pairs = build_pairs_for_idx(objs, H, W)

            for (x, tissue_poly, tool_poly) in pairs:
                total_pairs += 1
                seq = build_sequence(cap, idx, tissue_poly, tool_poly, depth_map)
                if seq is None:
                    skipped_pairs += 1
                    continue
                xb_buf.append(seq)
                yb_buf.append(int(x))

                if len(xb_buf) >= BATCH_SIZE:
                    flush_batch()

            if total_pairs % PRINT_EVERY == 0:
                print(f"[{processed_videos+1}/{len(videos)}] frame {i_idx}/{len(frame_indices)} — coppie finora: {total_pairs}, saltate: {skipped_pairs}")

        cap.release()
        processed_videos += 1

    flush_batch()

    if len(y_true) == 0:
        raise RuntimeError("Nessuna predizione valida prodotta.")

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    print("\n=== RISULTATI ON-THE-FLY (oracle GT) ===")
    print(f"Coppie totali processate:   {total_pairs}")
    print(f"Coppie saltate (ROI vuota): {skipped_pairs}")
    print(f"Accuracy:                   {acc:.4f}")
    print(f"F1 macro:                   {f1_mac:.4f}")
    print(f"F1 weighted:                {f1_w:.4f}")
    print(f"Precision:                  {prec:.4f}")
    print(f"Recall:                     {rec:.4f}")
    print(f"Balanced accuracy:          {bal_acc:.4f}")
    print("Confusion matrix [righe=gt 0,1; colonne=pred 0,1]:")
    print(cm)
    print("\nClassification report:")
    print(report)

if __name__ == "__main__":
    evaluate_on_the_fly()
