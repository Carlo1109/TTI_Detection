import os, re, json, cv2, torch, numpy as np
from PIL import Image
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from ultralytics import YOLO
from transformers import pipeline
from torchvision.models import resnet18
from Model import CNN_TCN_Classifier

TEST_VIDEOS_DIR = '../Dataset/video_dataset/videos/val/'
TEST_LABELS_DIR = '../Dataset/video_dataset/labels/val/'

YOLO_WEIGHTS = '../Common Code/runs_OLD_DATASET/segment/train/weights/best.pt'
TCN_WEIGHTS  = 'model_TCN_V4.pt'   
THR_TCN      = 0.10

IMG_SIZE     = 224
SEQ_LEN      = 5
CONF_THR     = 0.35     
IOU_DUP_THR  = 0.95     
K_NEAREST    = 3        
D_MAX        = None     
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_tool_id(name):
    if name is None: return 0
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
    key = str(name).strip().lower()
    return name_to_id.get(key, 0)

def to_tti_id(name):
    if name is None: return 12
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
    key = str(name).strip().lower()
    return name_to_id.get(key, 12)

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

def centroid(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None, None
    return xs.mean(), ys.mean()

def extract_union_bbox(tool_mask, tissue_mask):
    combined = (tool_mask.astype(np.uint8) | tissue_mask.astype(np.uint8))
    if combined.max() == 0:
        return None
    x, y, w, h = cv2.boundingRect(combined)
    return (x, y, w, h)

def make_clip_from_bbox(cap, idx_center, bbox, depth_map_center, tool_mask_center, tissue_mask_center,
                        IMG_SIZE=224, SEQ_LEN=5):
    x, y, w, h = bbox
    seq = []
    merged_center = (tool_mask_center | tissue_mask_center).astype(np.uint8) * 255
    for t in range(idx_center-(SEQ_LEN-1), idx_center+1):
        fr_bgr = _load_frame(cap, t, rgb=False)
        rgb = cv2.cvtColor(fr_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        dpatch = depth_map_center[y:y+h, x:x+w]
        mpatch = merged_center[y:y+h, x:x+w][..., None]
        roi = np.concatenate([rgb, dpatch[...,None], mpatch], axis=-1)  # [H,W,5]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        roi = roi.astype(np.float32) / 255.0
        seq.append(np.transpose(roi, (2,0,1)))  # C,H,W
    return np.stack(seq, axis=0)  # T,C,H,W




def filter_by_conf_and_dedup(result, conf_thr=0.35, iou_thr=0.95):
    r = result[0]
    if r.masks is None or r.boxes is None or len(r.boxes.cls) == 0:
        return []
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs   = r.boxes.conf.cpu().numpy()
    masks   = r.masks.data.cpu().numpy()  # [N,h,w]
    H, W = r.masks.orig_shape

    keep = confs >= conf_thr
    classes, masks, confs = classes[keep], masks[keep], confs[keep]
    if len(classes) == 0:
        return []

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

def pair_by_k_nearest(tools, tissues, k=3, d_max=None):
    pairs = []
    tool_cent = []
    for t in tools:
        cx, cy = centroid(t['mask'])
        if cx is None: continue
        tool_cent.append((t, cx, cy))
    tissue_cent = []
    for u in tissues:
        cx, cy = centroid(u['mask'])
        if cx is None: continue
        tissue_cent.append((u, cx, cy))

    for t, cx_t, cy_t in tool_cent:
        dists = []
        for u, cx_u, cy_u in tissue_cent:
            d = np.hypot(cx_t - cx_u, cy_t - cy_u)
            dists.append((d, u))
        dists.sort(key=lambda x: x[0])
        picked = dists[:k] if k is not None else dists
        for d, u in picked:
            if d_max is not None and d > d_max:
                continue
            pairs.append({'tool': t, 'tissue': u})
    return pairs

def build_gt_pairs_dict(objs):
    d = {}
    L = len(objs)
    if L < 2: return d

    def get_tti_id_from_obj(o):
        tti_name = o.get('interaction_type', None)
        return to_tti_id(tti_name)

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
            d[(to_tool_id(tool_name), tti_id)] = 1

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
            d[(to_tool_id(inter_tool_name), tti_id)] = 1
        if non_inter_tool_name is not None and tti_id is not None:
            d[(to_tool_id(non_inter_tool_name), tti_id)] = 0
        if non_inter_tool_name is None and inter_tool_name2 is not None and tti_id is not None:
            if inter_tool_name2 != inter_tool_name:
                d[(to_tool_id(inter_tool_name2), tti_id)] = 0

    else:  
        pos_tissues = []
        instruments = []
        for o in objs:
            if 'is_tti' in o and int(o.get('is_tti', 0)) == 1:
                pos_tissues.append((o.get('interaction_tool'), get_tti_id_from_obj(o)))
        for o in objs:
            if 'is_tti' not in o and 'instrument_type' in o:
                instruments.append(o.get('instrument_type'))
        for (inter_tool_name, tti_id) in pos_tissues:
            for instr_name in instruments:
                if inter_tool_name is None or tti_id is None or instr_name is None:
                    continue
                if inter_tool_name == instr_name:
                    d[(to_tool_id(instr_name), tti_id)] = 1
                else:
                    d[(to_tool_id(instr_name), tti_id)] = 0

    return d  
def test_end_to_end():
    print("Device:", DEVICE)
    depth_pipe = pipeline(task="depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Small-hf",
                          device=0 if DEVICE.type=='cuda' else -1)
    yolo = YOLO(YOLO_WEIGHTS)
    names = yolo.model.names if hasattr(yolo, 'model') else yolo.names

    model = CNN_TCN_Classifier(sequence_length=SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load(TCN_WEIGHTS, map_location=DEVICE), strict=True)
    model.eval()

    y_true, y_pred = [], []
    wrong_pairing, missed_pos, total_pairs = 0, 0, 0

    videos = [v for v in os.listdir(TEST_VIDEOS_DIR) if not v.startswith('.')]
    for vi, vid in enumerate(videos, 1):
        vpath = os.path.join(TEST_VIDEOS_DIR, vid)
        cap, fcount = _load_video(vpath)
        key = normalize(os.path.splitext(vid)[0])
        jpath = next((os.path.join(TEST_LABELS_DIR, f)
                      for f in os.listdir(TEST_LABELS_DIR)
                      if normalize(os.path.splitext(f)[0]) == key), None)
        if jpath is None:
            cap.release(); continue
        labels = json.load(open(jpath, 'r'))['labels']
        print(f"[E2E] Video ({vi}/{len(videos)}): {vid}  ({fcount} frames)")

        for idx_s, objs in labels.items():
            idx = int(idx_s)
            if idx - (SEQ_LEN - 1) < 0 or idx >= fcount:
                continue

            gt_pairs = build_gt_pairs_dict(objs)  # {(tool_id, tti_id): is_tti}
            if len(gt_pairs) == 0:
                continue

            frame_c_bgr = _load_frame(cap, idx, rgb=False)
            H, W = frame_c_bgr.shape[:2]
            depth_map = np.array(
                depth_pipe(Image.fromarray(cv2.cvtColor(frame_c_bgr, cv2.COLOR_BGR2RGB)))['depth'],
                dtype=np.float32
            )

            r = yolo.predict(frame_c_bgr, verbose=False)
            dets = filter_by_conf_and_dedup(r, conf_thr=CONF_THR, iou_thr=IOU_DUP_THR)
            tools, tissues = [], []
            for d in dets:
                cls = d['class']
                nm  = names[cls] if isinstance(names, dict) else names[cls]
                tid = to_tool_id(nm)
                xid = to_tti_id(nm)
                if tid != 0:
                    d['tool_id'] = tid
                    tools.append(d)
                elif xid != 12:
                    d['tti_id'] = xid
                    tissues.append(d)

            pairs = pair_by_k_nearest(tools, tissues, k=K_NEAREST, d_max=D_MAX)
            total_pairs += len(pairs)

            pred_pos_keys_this_frame = set()

            for pair in pairs:
                tool_mask   = pair['tool']['mask']
                tissue_mask = pair['tissue']['mask']
                bbox = extract_union_bbox(tool_mask, tissue_mask)
                if bbox is None:
                    continue
                seq = make_clip_from_bbox(cap, idx, bbox, depth_map, tool_mask, tissue_mask,
                                          IMG_SIZE=IMG_SIZE, SEQ_LEN=SEQ_LEN)
                xb = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    prob = torch.sigmoid(model(xb).squeeze(1)).item()
                pred = 1 if prob >= THR_TCN else 0

                tool_id = pair['tool'].get('tool_id', 0)
                tti_id  = pair['tissue'].get('tti_id', 12)
                key = (tool_id, tti_id)

                gt_tools_in_frame = {k[0] for k in gt_pairs.keys()}
                if tool_id not in gt_tools_in_frame:
                    wrong_pairing += 1

                if key in gt_pairs:
                    y_true.append(int(gt_pairs[key]))
                    y_pred.append(int(pred))
                    if pred == 1:
                        pred_pos_keys_this_frame.add(key)

            for k, v in gt_pairs.items():
                if v == 1 and k not in pred_pos_keys_this_frame:
                    missed_pos += 1

        cap.release()

    if len(y_true) == 0:
        print("Nessuna coppia valutata in E2E.")
        return

    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1w  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    ba   = balanced_accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)

    print("\n===== RISULTATI E2E (pairing + classificazione) =====")
    print(f"Samples: {len(y_true)}  |  Total predicted pairs: {total_pairs}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 macro: {f1m:.4f}  |  F1 weighted: {f1w:.4f}")
    print(f"Precision: {prec:.4f}  |  Recall: {rec:.4f}")
    print("Confusion matrix:\n", cm)
    print(f"Balanced accuracy: {ba:.4f}")
    print(f"Wrong pairing (tool non in GT): {wrong_pairing}")
    print(f"Missed positives (GT non coperti da predizioni positive): {missed_pos}")

if __name__ == '__main__':
    test_end_to_end()
