import os, json, re, cv2, torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from torchvision.models import resnet18

TEST_VIDEOS_DIR = '../Dataset/video_dataset/videos/test/'   
TEST_LABELS_DIR = '../Dataset/video_dataset/labels/test/'   
IMG_SIZE        = 224
SEQ_LEN         = 5
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

YOLO_WEIGHTS    = '../Common Code/runs_OLD_DATASET/segment/train/weights/best.pt'  
TCN_WEIGHTS     = 'model_TCN_V3.pt'  


IMG_SIZE   = 224
SEQ_LEN    = 5
CONF_THR   = 0.45     
IOU_DUP_THR= 0.90     
D_MAX      = 30       
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', name).lower()

def _load_video(path):
    cap = cv2.VideoCapture(path)
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def _load_frame(cap, idx, rgb=False):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        raise ValueError(f"Impossibile leggere frame {idx}")
    if rgb:
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    return fr  

def parse_mask_string(mask_str, H, W):
    if not mask_str or not mask_str.strip():
        return np.zeros((0,2), np.int32)
    tokens = mask_str.strip().split()
    coords = []
    for i in range(0, (len(tokens)//2)*2, 2):
        try:
            x = int(float(tokens[i]) * W); y = int(float(tokens[i+1]) * H)
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
        fr_bgr = _load_frame(cap, t, rgb=False)
        rgb = cv2.cvtColor(fr_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        Hc, Wc = rgb.shape[:2]
        dpatch = depth_map_center[y:y+h, x:x+w] if depth_map_center is not None else np.zeros((Hc, Wc), np.float32)
        mpatch = (tool_mask_center[y:y+h, x:x+w] | tissue_mask_center[y:y+h, x:x+w]).astype(np.uint8)*255
        roi = np.concatenate([rgb, dpatch[...,None], mpatch[...,None]], axis=-1)  # [H,W,5]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        roi = roi.astype(np.float32) / 255.0
        seq.append(np.transpose(roi, (2,0,1)))  # C,H,W
    return np.stack(seq, axis=0)  # T,C,H,W

def to_tool_id(name):
    if name is None: return 0
    m = {'unknown_tool':0,'dissector':1,'scissors':2,'suction':3,'grasper 3':4,'harmonic':5,'grasper':6,'bipolar':7,'grasper 2':8,'cautery (hook, spatula)':9,'ligasure':10,'stapler':11}
    return m.get(str(name).lower().strip(), 0)

def to_tti_id(name):
    if name is None: return 12
    m = {'unknown_tti':12,'coagulation':13,'other':14,'retract and grab':15,'blunt dissection':16,'energy - sharp dissection':17,'staple':18,'retract and push':19,'cut - sharp dissection':20}
    return m.get(str(name).lower().strip(), 12)

class CNN_TCN_Classifier(nn.Module):
    def __init__(self, tcn_channels=[256, 128], sequence_length=SEQ_LEN, num_classes=1, pretrained=True):
        super().__init__()
        backbone = resnet18(pretrained=pretrained)
        old_w = backbone.conv1.weight.data.clone()  # [64,3,7,7]
        backbone.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            bias=False
        )
        nn.init.kaiming_normal_(backbone.conv1.weight, nonlinearity='relu')
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_w
            backbone.conv1.weight[:, 3:5] = old_w.mean(dim=1, keepdim=True)
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
        self.classifier = nn.Sequential(  # <- stesso nome del training
            nn.Flatten(),
            nn.Linear(tcn_channels[-1], num_classes)
        )

    def forward(self, x):  # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)
        x = self.pool2d(x)
        x = x.view(B, T, 512)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.pool1d(x)
        out = self.classifier(x)  # logit
        return out

def filter_by_conf_and_dedup(result, conf_thr=0.45, iou_thr=0.9):
    r = result[0]
    if r.masks is None or r.boxes is None or len(r.boxes.cls) == 0:
        return []
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs   = r.boxes.conf.cpu().numpy()
    masks   = r.masks.data.cpu().numpy()  # [N,h,w]
    keep = confs >= conf_thr
    classes, masks, confs = classes[keep], masks[keep], confs[keep]
    H, W = r.masks.orig_shape
    masks = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) for m in masks], axis=0)

    out = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        sel = []
        for i in idx:
            m = masks[i]; dup = False
            for j in sel:
                inter = np.logical_and(m, masks[j]).sum()
                uni   = np.logical_or (m, masks[j]).sum() + 1e-9
                if inter / uni >= iou_thr:
                    dup = True; break
            if not dup:
                sel.append(i)
        for i in sel:
            out.append({'class': int(classes[i]), 'mask': masks[i], 'conf': float(confs[i])})
    return out

TOOL_CLASSES = list(range(0, 12))

def centroid(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None, None
    return xs.mean(), ys.mean()

def dilate(mask, k=3):
    return cv2.dilate(mask.astype(np.uint8), np.ones((k,k), np.uint8), iterations=1)

def pair_by_nearest(tools, tissues, d_max=30):
    pairs = []
    for t in tools:
        cx_t, cy_t = centroid(t['mask'])
        if cx_t is None: continue
        best, bestd = None, 1e9
        for u in tissues:
            cx_u, cy_u = centroid(u['mask'])
            if cx_u is None: continue
            d = np.hypot(cx_t - cx_u, cy_t - cy_u)
            if d < bestd:
                bestd = d; best = u
        if best is not None and bestd <= d_max:
            overlap = cv2.countNonZero(cv2.bitwise_and(dilate(t['mask'], 3), best['mask']))
            if overlap > 0:
                pairs.append({'tool': t, 'tissue': best})
    return pairs

def extract_union_roi_e2e(frame_bgr, tool_mask, tissue_mask, depth_map=None):
    combined = (tool_mask.astype(np.uint8) | tissue_mask.astype(np.uint8))
    if combined.max() == 0:
        return None, None
    x, y, w, h = cv2.boundingRect(combined)
    roi_img = frame_bgr[y:y+h, x:x+w, :]
    rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    if depth_map is not None:
        depth_roi = depth_map[y:y+h, x:x+w]
        roi = np.concatenate([rgb, depth_roi[..., None]], axis=-1)
    else:
        roi = rgb
    merged_mask = combined[y:y+h, x:x+w][..., None] * 255
    roi = np.concatenate([roi, merged_mask], axis=-1)  # [H,W,5]
    return roi, (x, y, w, h)

def build_gt_pairs_from_json(objs):
    tools = set(); tissues = []
    for o in objs:
        if 'instrument_polygon' in o:
            tools.add(to_tool_id(o.get('instrument_type')))
        if 'tti_polygon' in o:
            uid = to_tti_id(o.get('tti_class')); is_tti = int(o.get('is_tti', 0))
            inter_tid = to_tool_id(o.get('interaction_tool'))
            tissues.append((uid, is_tti, inter_tid))
    gt = {}
    for t in tools:
        for (u, is_tti, inter_tid) in tissues:
            gt[(t, u)] = 1 if (is_tti == 1 and inter_tid == t) else 0
    return gt  # {(tool,tissue): is_tti}

def test_end_to_end(depth_pipe, yolo, model, thr_best):
    model.eval()
    y_true, y_pred = [], []
    wrong_pairing, missed_pos, total_pairs = 0, 0, 0

    videos = [v for v in os.listdir(TEST_VIDEOS_DIR) if not v.startswith('.')]
    i = 1
    for vid in videos:
        print(f"Processing video {i}/102...")
        i+=1
        vpath = os.path.join(TEST_VIDEOS_DIR, vid)
        cap, fcount = _load_video(vpath)
        key = normalize(os.path.splitext(vid)[0])
        jpath = next((os.path.join(TEST_LABELS_DIR, f)
                      for f in os.listdir(TEST_LABELS_DIR)
                      if normalize(os.path.splitext(f)[0]) == key), None)
        if jpath is None:
            cap.release(); continue
        labels = json.load(open(jpath, 'r'))['labels']

        for idx_s, objs in labels.items():
            idx = int(idx_s)
            if idx - (SEQ_LEN - 1) < 0 or idx >= fcount:
                continue

            gt_pairs = build_gt_pairs_from_json(objs)
            if not gt_pairs:
                continue

            frame_c = _load_frame(cap, idx, rgb=False)
            dets = yolo.predict(frame_c, verbose=False)
            dets = filter_by_conf_and_dedup(dets, conf_thr=CONF_THR, iou_thr=IOU_DUP_THR)
            tools   = [d for d in dets if d['class'] in TOOL_CLASSES]
            tissues = [d for d in dets if d['class'] not in TOOL_CLASSES]
            pairs = pair_by_nearest(tools, tissues, d_max=D_MAX)

            depth_map = np.array(depth_pipe(Image.fromarray(cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)))['depth'],
                                 dtype=np.float32)

            matched_keys = set()

            for pair in pairs:
                t_class = int(pair['tool']['class'])
                u_class = int(pair['tissue']['class'])

                roi0, bbox = extract_union_roi_e2e(frame_c, pair['tool']['mask'], pair['tissue']['mask'], depth_map=depth_map)
                if roi0 is None:
                    continue

                x, y, w, h = bbox
                seq = []
                for t in range(idx-(SEQ_LEN-1), idx+1):
                    fr = _load_frame(cap, t, rgb=False)
                    patch_rgb = cv2.cvtColor(fr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    dpatch = depth_map[y:y+h, x:x+w]
                    mpatch = (pair['tool']['mask'][y:y+h, x:x+w] | pair['tissue']['mask'][y:y+h, x:x+w]).astype(np.uint8)*255
                    roi = np.concatenate([patch_rgb, dpatch[...,None], mpatch[...,None]], axis=-1)
                    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                    roi = roi.astype(np.float32) / 255.0
                    seq.append(np.transpose(roi, (2,0,1)))
                seq = np.stack(seq, axis=0)  # T,C,H,W
                xb = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logit = model(xb).squeeze(1)
                    prob  = torch.sigmoid(logit).item()
                    pred  = 1 if prob >= thr_best else 0

                y_pred.append(pred)
                y_t = gt_pairs.get((t_class, u_class), 0)
                y_true.append(y_t)
                if (t_class, u_class) in gt_pairs:
                    matched_keys.add((t_class, u_class))
                else:
                    wrong_pairing += 1
                total_pairs += 1

            for (k, v) in gt_pairs.items():
                if v == 1 and k not in matched_keys:
                    y_true.append(1)
                    y_pred.append(0)
                    missed_pos += 1
                    total_pairs += 1

        cap.release()

    if not y_true:
        print("E2E: nessun sample prodotto.")
        return

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    ba   = balanced_accuracy_score(y_true, y_pred)

    print("\n===== RISULTATI E2E (pairing+classificazione) =====")
    print(f"Samples: {len(y_true)}  |  Total predicted pairs: {total_pairs}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 macro: {f1m:.4f}  |  F1 weighted: {f1w:.4f}")
    print(f"Precision: {prec:.4f}  |  Recall: {rec:.4f}")
    print("Confusion matrix:\n", cm)
    print(f"Balanced accuracy: {ba:.4f}")
    print(f"Wrong pairing (FP coppie): {wrong_pairing}")
    print(f"Missed positives (FN coppie GT positive non predette): {missed_pos}")


if __name__ == '__main__':
    print("Device:", DEVICE)

    depth_pipe = pipeline(task="depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Small-hf",
                          device=0 if DEVICE.type=='cuda' else -1)
    yolo = YOLO(YOLO_WEIGHTS)
    model = CNN_TCN_Classifier(sequence_length=SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load(TCN_WEIGHTS, map_location=DEVICE), strict=True)

    test_end_to_end(depth_pipe, yolo, model, thr_best=0.05)