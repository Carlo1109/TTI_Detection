import os, json, re, cv2, torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score

TEST_VIDEOS_DIR = '../Dataset/video_dataset/videos/test/'   
TEST_LABELS_DIR = '../Dataset/video_dataset/labels/test/'   
IMG_SIZE        = 224
SEQ_LEN         = 5
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

YOLO_WEIGHTS    = '../Common Code/runs_OLD_DATASET/segment/train/weights/best.pt'  
TCN_WEIGHTS     = 'model_TCN_new.pt'        

def to_tool_id(name):
    if name == None :
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
    if name not in name_to_id.keys():
      return 0
    return name_to_id[name]


def to_tti_id(name):
    if name == None:
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
    if name not in name_to_id.keys():
        return 12
    
    return name_to_id[name]

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

def yolo_inference(model: YOLO, image_bgr: np.ndarray) -> list[dict]:
    res = model.predict(image_bgr, verbose=False)
    r = res[0]
    if r.masks is None or r.boxes is None or len(r.boxes.cls) == 0:
        return []
    classes = r.boxes.cls.cpu().numpy().astype(int)
    masks   = r.masks.data.cpu().numpy()  
    H, W = r.masks.orig_shape
    out = []
    for c, m in zip(classes, masks):
        m_full = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        out.append({'class': int(c), 'mask': m_full})
    return out

TOOL_CLASSES = list(range(0, 12))
def find_tool_tissue_pairs(detections: list[dict]) -> list[dict]:
    tools   = [d for d in detections if d['class'] in TOOL_CLASSES]
    tissues = [d for d in detections if d['class'] not in TOOL_CLASSES]
    pairs = []
    for t in tools:
        for u in tissues:
            pairs.append({'tool': t, 'tissue': u})
    return pairs

def extract_union_roi(frame_bgr, tool_mask, tissue_mask, depth_map=None):
    combined = (tool_mask.astype(np.uint8) | tissue_mask.astype(np.uint8))
    if combined.max() == 0:
        return None, None  
    x, y, w, h = cv2.boundingRect(combined)
    roi_img = frame_bgr[y:y+h, x:x+w, :]                      
    if depth_map is not None:
        depth_roi = depth_map[y:y+h, x:x+w]                   
        roi = np.concatenate([cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB),  
                              depth_roi[..., None]], axis=-1)
    else:
        roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    merged_mask = combined[y:y+h, x:x+w][..., None] * 255
    roi = np.concatenate([roi, merged_mask], axis=-1)         
    return roi, (x, y, w, h)


def build_gt_pairs_from_json(objs: list[dict]) -> dict[tuple[int,int], int]:
    tool_ids = set()
    tissue_items = []  

    for o in objs:
        # tool
        if 'instrument_polygon' in o:
            cand = o.get('instrument_type') or o.get('class') or None
            tid = to_tool_id(cand)
            tool_ids.add(tid)

        # tissue
        if 'tti_polygon' in o:
            cand_tissue = o.get('tti_class') or o.get('class') or None
            uid = to_tti_id(cand_tissue)
            is_tti = int(o.get('is_tti', 0))

            inter_tool_cand = o.get('interaction_tool', None)
            inter_tid = to_tool_id(inter_tool_cand) if inter_tool_cand else None

            tissue_items.append((uid, is_tti, inter_tid))

    gt = {}
    for t_id in tool_ids:
        for (u_id, is_tti, inter_tid) in tissue_items:
            label = 1 if (is_tti == 1 and inter_tid == t_id) else 0
            gt[(t_id, u_id)] = label

    return gt


class CNN_TCN_Classifier(nn.Module):
    def __init__(self, tcn_channels=[256, 128], sequence_length=SEQ_LEN, num_classes=1, pretrained=True, in_channels=5):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=pretrained)
        backbone.conv1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=backbone.conv1.out_channels,
                                   kernel_size=backbone.conv1.kernel_size,
                                   stride=backbone.conv1.stride,
                                   padding=backbone.conv1.padding,
                                   bias=False)
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

def test_end_to_end():
    yolo = YOLO(YOLO_WEIGHTS)
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if DEVICE.type=='cuda' else -1)
    model = CNN_TCN_Classifier(sequence_length=SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load(TCN_WEIGHTS, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []
    wrong_pairing, missed_pos, total_pairs = 0, 0, 0

    videos = [v for v in os.listdir(TEST_VIDEOS_DIR) if not v.startswith('.')]
    for i, vid in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] Video: {vid}")
        vpath = os.path.join(TEST_VIDEOS_DIR, vid)
        cap, fcount = _load_video(vpath)

        key = normalize(os.path.splitext(vid)[0])
        jpath = next((os.path.join(TEST_LABELS_DIR, f)
                      for f in os.listdir(TEST_LABELS_DIR)
                      if normalize(os.path.splitext(f)[0]) == key), None)
        if jpath is None:
            print("  Nessun JSON di label trovato, salto.")
            cap.release()
            continue

        labels = json.load(open(jpath, 'r'))['labels']

        for idx_s, objs in labels.items():
            idx = int(idx_s)
            if idx - (SEQ_LEN - 1) < 0 or idx >= fcount:
                continue

            gt_pairs = build_gt_pairs_from_json(objs)  # {(tool_class,tissue_class): is_tti}
            if len(gt_pairs) == 0:
                continue

            
            frame_c = _load_frame(cap, idx)  # BGR
            
            dets = yolo_inference(yolo, frame_c)
            pairs = find_tool_tissue_pairs(dets)

            
            depth_map = np.array(depth_pipe(Image.fromarray(cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)))['depth']).astype(np.float32)

            matched_keys = set()

            for pair in pairs:
                t_class = int(pair['tool']['class'])
                u_class = int(pair['tissue']['class'])

                roi0, bbox = extract_union_roi(frame_c,
                                               pair['tool']['mask'].astype(np.uint8),
                                               pair['tissue']['mask'].astype(np.uint8),
                                               depth_map=depth_map)
                if roi0 is None:
                    continue

                seq_tensors = []
                x, y, w, h = bbox
                for t in range(idx-(SEQ_LEN-1), idx+1):
                    fr = _load_frame(cap, t)  # BGR
                    patch_rgb = cv2.cvtColor(fr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    patch_depth = depth_map[y:y+h, x:x+w]
                    merged_mask = (pair['tool']['mask'][y:y+h, x:x+w] | pair['tissue']['mask'][y:y+h, x:x+w]).astype(np.uint8)*255
                    roi = np.concatenate([patch_rgb, patch_depth[...,None], merged_mask[...,None]], axis=-1)
                    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                    roi = roi.astype(np.float32) / 255.0
                    seq_tensors.append(np.transpose(roi, (2,0,1)))  # C,H,W

                seq_np = np.stack(seq_tensors, axis=0)            # T,C,H,W
                xb = torch.from_numpy(seq_np).unsqueeze(0).to(DEVICE)  # 1,T,C,H,W

                with torch.no_grad():
                    prob = model(xb).squeeze(1).item()  
                pred = 1 if prob >= 0.5 else 0
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

    if len(y_true) == 0:
        print("Nessun dato di test prodotto.")
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
    print(f"Wrong pairing (FP per coppie inesistenti): {wrong_pairing}")
    print(f"Missed positives (FN su coppie GT positive non predette): {missed_pos}")

if __name__ == '__main__':
    test_end_to_end()
