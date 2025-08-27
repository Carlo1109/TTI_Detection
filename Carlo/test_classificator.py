import os, re, json, cv2, torch, numpy as np
from PIL import Image
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from transformers import pipeline

TEST_VIDEOS_DIR = '../Dataset/video_dataset/videos/test/'   
TEST_LABELS_DIR = '../Dataset/video_dataset/labels/test/'   
IMG_SIZE        = 224
SEQ_LEN         = 5
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

YOLO_WEIGHTS    = '../Common Code/runs_OLD_DATASET/segment/train/weights/best.pt'  
TCN_WEIGHTS     = 'model_TCN_new.pt'                              
USE_DEPTH    = True   
VERBOSE      = True

def to_tool_id(name):
    if name is None:
        return 0
    name_to_id = {
        'unknown_tool': 0, 'dissector': 1, 'scissors': 2, 'suction': 3,
        'grasper 3': 4, 'harmonic': 5, 'grasper': 6, 'bipolar': 7,
        'grasper 2': 8, 'cautery (hook, spatula)': 9, 'ligasure': 10, 'stapler': 11,
    }
    key = str(name).lower().strip()
    return name_to_id.get(key, 0)

def to_tti_id(name):
    if name is None:
        return 12
    name_to_id = {
        'unknown_tti': 12, 'coagulation': 13, 'other': 14,
        'retract and grab': 15, 'blunt dissection': 16,
        'energy - sharp dissection': 17, 'staple': 18,
        'retract and push': 19, 'cut - sharp dissection': 20,
    }
    key = str(name).lower().strip()
    return name_to_id.get(key, 12)

def normalize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', name).lower()

def _load_video(path):
    cap = cv2.VideoCapture(path)
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def _load_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Impossibile leggere frame {idx}")
    return frame  # BGR

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

def poly_to_mask(poly_str, H, W):
    pts = parse_mask_string(poly_str, H, W)
    mask = np.zeros((H, W), np.uint8)
    if pts.size:
        cv2.fillPoly(mask, [pts], 1)
    return mask

def build_gt_pairs_and_masks(objs, H, W):
    tools = []
    tissues = []
    for o in objs:
        if 'instrument_polygon' in o:
            tid = to_tool_id(o.get('instrument_type'))
            tpoly = get_polygon(o['instrument_polygon'])
            tmask = poly_to_mask(tpoly, H, W)
            tools.append((tid, tmask))
        if 'tti_polygon' in o:
            uid = to_tti_id(o.get('tti_class'))
            is_tti = int(o.get('is_tti', 0))
            inter_tool = to_tool_id(o.get('interaction_tool'))
            upoly = get_polygon(o['tti_polygon'])
            umask = poly_to_mask(upoly, H, W)
            tissues.append((uid, is_tti, inter_tool, umask))
    pairs = []
    for (tid, tmask) in tools:
        for (uid, is_tti, inter_tid, umask) in tissues:
            label = 1 if (is_tti == 1 and inter_tid == tid) else 0
            pairs.append((tid, uid, label, tmask, umask))
    return pairs

def extract_union_bbox(tool_mask, tissue_mask):
    comb = (tool_mask.astype(np.uint8) | tissue_mask.astype(np.uint8))
    if comb.max() == 0:
        return None
    x, y, w, h = cv2.boundingRect(comb)
    return x, y, w, h

def make_clip(cap, idx_center, bbox, depth_map_center, tool_mask_center, tissue_mask_center):
    x, y, w, h = bbox
    seq = []
    for t in range(idx_center-(SEQ_LEN-1), idx_center+1):
        fr = _load_frame(cap, t)  # BGR
        patch_rgb = cv2.cvtColor(fr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        Hc, Wc = patch_rgb.shape[:2]
        if USE_DEPTH and depth_map_center is not None:
            dpatch = depth_map_center[y:y+h, x:x+w]
        else:
            dpatch = np.zeros((Hc, Wc), np.float32)
        mpatch = (tool_mask_center[y:y+h, x:x+w] | tissue_mask_center[y:y+h, x:x+w]).astype(np.uint8)*255
        roi = np.concatenate([patch_rgb, dpatch[...,None], mpatch[...,None]], axis=-1)  # H,W,5
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        roi = roi.astype(np.float32) / 255.0
        seq.append(np.transpose(roi, (2,0,1)))  # C,H,W
    seq_np = np.stack(seq, axis=0)  # T,C,H,W
    return seq_np

class CNN_TCN_Classifier(nn.Module):
    def __init__(self, tcn_channels=[256, 128], sequence_length=SEQ_LEN, num_classes=1, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18
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
                nn.Conv1d(in_channels=num_inputs if i == 0 else tcn_channels[i-1],
                          out_channels=out_ch, kernel_size=3, padding=dilation, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]
        self.tcn = nn.Sequential(*tcn_layers)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(tcn_channels[-1], num_classes), nn.Sigmoid())

    def forward(self, x):  # x: [B,T,C,H,W]
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)
        x = self.pool2d(x)
        x = x.view(B, T, 512)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.pool1d(x)
        out = self.classifier(x)  # [B,1] sigmoid
        return out

def evaluate_oracle():
    depth_pipe = pipeline(task="depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Small-hf",
                          device=0 if DEVICE.type == 'cuda' else -1)
    model = CNN_TCN_Classifier(sequence_length=SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load(TCN_WEIGHTS, map_location=DEVICE))
    model.eval()

    y_true, y_prob = [], []

    videos = [v for v in os.listdir(TEST_VIDEOS_DIR) if not v.startswith('.')]
    for i, vid in enumerate(videos, 1):
        if VERBOSE: print(f"[{i}/{len(videos)}] Video: {vid}")
        vpath = os.path.join(TEST_VIDEOS_DIR, vid)
        cap, fcount = _load_video(vpath)

        key = normalize(os.path.splitext(vid)[0])
        jpath = next((os.path.join(TEST_LABELS_DIR, f)
                      for f in os.listdir(TEST_LABELS_DIR)
                      if normalize(os.path.splitext(f)[0]) == key), None)
        if jpath is None:
            if VERBOSE: print("  Nessun JSON, salto.")
            cap.release()
            continue

        labels = json.load(open(jpath, 'r'))['labels']

        for idx_s, objs in labels.items():
            idx = int(idx_s)
            if idx - (SEQ_LEN - 1) < 0 or idx >= fcount:
                continue

            frame_c = _load_frame(cap, idx)  # BGR
            H, W = frame_c.shape[:2]
            pairs = build_gt_pairs_and_masks(objs, H, W)
            if len(pairs) == 0:
                continue

            depth_map = None
            if USE_DEPTH:
                depth_map = np.array(
                    depth_pipe(Image.fromarray(cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)))['depth'],
                    dtype=np.float32
                )

            for (tool_id, tti_id, label, tool_mask, tti_mask) in pairs:
                bbox = extract_union_bbox(tool_mask, tti_mask)
                if bbox is None:
                    continue
                seq_np = make_clip(cap, idx, bbox, depth_map, tool_mask, tti_mask)
                xb = torch.from_numpy(seq_np).unsqueeze(0).to(DEVICE)  # 1,T,C,H,W
                with torch.no_grad():
                    prob = model(xb).squeeze(1).item()
                y_true.append(label)
                y_prob.append(prob)

        cap.release()

    if len(y_true) == 0:
        print("Nessun sample raccolto.")
        return

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    ba   = balanced_accuracy_score(y_true, y_pred)
    pos  = int(np.sum(y_true))

    print("\n===== RISULTATI ORACLE (solo classificatore) =====")
    print(f"Samples: {len(y_true)}  |  Positivi GT: {pos}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 macro: {f1m:.4f}  |  F1 weighted: {f1w:.4f}")
    print(f"Precision: {prec:.4f}  |  Recall: {rec:.4f}")
    print("Confusion matrix:\n", cm)
    print(f"Balanced accuracy: {ba:.4f}")

    best_f1, best_thr = -1, 0.5
    for thr in np.linspace(0.05, 0.95, 19):
        y_hat = [1 if p >= thr else 0 for p in y_prob]
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"Miglior F1 in sweep: {best_f1:.4f} a soglia {best_thr:.2f}")

if __name__ == "__main__":
    evaluate_oracle()
