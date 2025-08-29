import os, re, json, cv2, torch, numpy as np
from PIL import Image
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from ultralytics import YOLO
from transformers import pipeline
from torchvision.models import resnet18
from Model import CNN_TCN_Classifier


IMAGES_TEST = '../Dataset/evaluation/images/'
LABELS_TEST = '../Dataset/evaluation/labels/'

YOLO_WEIGHTS = '../Common Code/runs_OLD_DATASET/segment/train/weights/best.pt'
TCN_WEIGHTS  = 'model_TCN_V4.pt'   
USE_TCN      = True        
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
        'unknown_tool': 0, 'dissector': 1, 'scissors': 2, 'suction': 3,
        'grasper 3': 4, 'harmonic': 5, 'grasper': 6, 'bipolar': 7,
        'grasper 2': 8, 'cautery (hook, spatula)': 9, 'ligasure': 10, 'stapler': 11
    }
    return name_to_id.get(str(name).strip().lower(), 0)

def to_tti_id(name):
    if name is None: return 12
    name_to_id = {
        'unknown_tti': 12, 'coagulation': 13, 'other': 14, 'retract and grab': 15,
        'blunt dissection': 16, 'energy - sharp dissection': 17, 'staple': 18,
        'retract and push': 19, 'cut - sharp dissection': 20
    }
    return name_to_id.get(str(name).strip().lower(), 12)

def filter_by_conf_and_dedup(result, conf_thr=0.35, iou_thr=0.95):
    r = result[0]
    if r.masks is None or r.boxes is None or len(r.boxes.cls) == 0:
        return []
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs   = r.boxes.conf.cpu().numpy()
    masks   = r.masks.data.cpu().numpy()
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

def centroid(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None, None
    return xs.mean(), ys.mean()

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

def extract_roi_5ch(image_bgr, depth_map, tool_mask, tissue_mask, img_size=224):
    H, W = image_bgr.shape[:2]
    merged = (tool_mask.astype(np.uint8) | tissue_mask.astype(np.uint8)) * 255
    x, y, w, h = cv2.boundingRect(merged)
    if w == 0 or h == 0: return None
    rgb = cv2.cvtColor(image_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    d   = depth_map[y:y+h, x:x+w]
    m   = merged[y:y+h, x:x+w][..., None]
    roi = np.concatenate([rgb, d[...,None], m], axis=-1).astype(np.float32) / 255.0
    roi = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    roi = np.transpose(roi, (2,0,1))  # C,H,W
    return roi  # [5,224,224]


def predict_pairs_on_image(image_path, yolo, depth_pipe, model, names, use_tcn=True):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return []

    depth_map = np.array(depth_pipe(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))['depth'], dtype=np.float32)

    r = yolo.predict(bgr, verbose=False)
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
    preds = []

    for pair in pairs:
        roi = extract_roi_5ch(bgr, depth_map, pair['tool']['mask'], pair['tissue']['mask'], IMG_SIZE)
        if roi is None:
            continue

        if use_tcn:
            seq = np.stack([roi for _ in range(SEQ_LEN)], axis=0)  # [T=SEQ_LEN,5,224,224]
            xb = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)     # [1,T,C,H,W]
            with torch.no_grad():
                prob = torch.sigmoid(model(xb).squeeze(1)).item()
            pred = 1 if prob >= THR_TCN else 0
        else:
            x1 = torch.from_numpy(roi).unsqueeze(0).to(DEVICE)     # [1,5,224,224]
            with torch.no_grad():
                logit = model(x1)            # adattare al tuo modello a frame singolo
                prob  = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0   # soglia placeholder

        preds.append({
            'tool_id': pair['tool']['tool_id'],
            'tti_id' : pair['tissue']['tti_id'],
            'pred'   : pred,
            'prob'   : prob
        })

    return preds

def main():
    print("Device:", DEVICE)
    depth_pipe = pipeline(task="depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Small-hf",
                          device=0 if DEVICE.type=='cuda' else -1)
    yolo = YOLO(YOLO_WEIGHTS)
    names = yolo.model.names if hasattr(yolo, 'model') else yolo.names

    if USE_TCN:
        clf = CNN_TCN_Classifier(sequence_length=SEQ_LEN).to(DEVICE)
        clf.load_state_dict(torch.load(TCN_WEIGHTS, map_location=DEVICE), strict=True)
        clf.eval()
    else:
        raise NotImplementedError("Imposta USE_TCN=True oppure fornisci un modello a frame singolo.")

    imgs = sorted([f for f in os.listdir(IMAGES_TEST) if f.lower().endswith('.png')])
    y_true, y_pred = [], []
    wrong_class, no_pred = 0, 0

    for i, img in enumerate(imgs, 1):
        print(f"Predicting image {i}/{len(imgs)}: {img}")
        label_file = os.path.join(LABELS_TEST, img.replace('.png', '.txt'))

        # carica GT {(tool_id, tti_id): is_tti}
        gt = {}
        if os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                    is_tti, tool_id, tti_id = map(int, parts[:3])
                    gt[(tool_id, tti_id)] = is_tti

        preds = predict_pairs_on_image(os.path.join(IMAGES_TEST, img), yolo, depth_pipe, clf, names, use_tcn=USE_TCN)

        if len(preds) == 0 and len(gt) > 0:
            # nessuna coppia predetta ma GT presente → conta 1 FN "macro"
            no_pred += 1
            # (opzionale) considerare tutti i GT come FN: qui manteniamo lo stile del tuo script (una penalità)
            continue

        # confronta solo su chiavi presenti in GT (come facevi tu)
        found_any = False
        for p in preds:
            key = (p['tool_id'], p['tti_id'])
            if key in gt:
                found_any = True
                y_pred.append(p['pred'])
                y_true.append(gt[key])

        if not found_any and len(gt) > 0:
            wrong_class += 1  # coppie predette su chiavi non presenti in GT

    print("Wrong classes (pred keys non presenti in GT):", wrong_class)
    print("No preds (immagini con GT ma 0 coppie predette):", no_pred)

    if len(y_true) == 0:
        print("Nessun match GT↔pred trovato. Controlla mapping ID o cartelle.")
        return

    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1w  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    ba   = balanced_accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)

    print("\n===== RISULTATI (stile ViT, su cartelle images/labels) =====")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 macro: {f1m:.4f}  |  F1 weighted: {f1w:.4f}")
    print(f"Precision: {prec:.4f}  |  Recall: {rec:.4f}")
    print("Confusion matrix:\n", cm)
    print(f"Balanced accuracy: {ba:.4f}")

if __name__ == '__main__':
    main()
