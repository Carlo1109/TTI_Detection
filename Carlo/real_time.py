import os, cv2, re, json, numpy as np, torch, torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from torchvision.models import resnet18
from Model import CNN_TCN_Classifier

VIDEO_PATH   = "../Dataset/video_dataset/videos/test/Adnanset-Lc 119-003.mp4"       
OUTPUT_PATH  = "output_overlay.mp4"    

YOLO_WEIGHTS = "../Common Code/runs_OLD_DATASET/segment/train/weights/best.pt"
TCN_WEIGHTS  = "model_TCN_V4.pt"

SEQ_LEN      = 5
IMG_SIZE     = 224
THR_TCN      = 0.10

CONF_THR     = 0.35
IOU_DUP_THR  = 0.95
K_NEAREST    = 3
D_MAX        = None  

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA_TOOL   = 0.50
ALPHA_TTI    = 0.50
COLOR_TOOL   = (255, 140, 60)   
COLOR_TTI    = (60, 220, 60)    
COLOR_BOX_OK = (40, 200, 40)    
FONT         = cv2.FONT_HERSHEY_SIMPLEX

def to_tool_id(name):
    if name is None: return 0
    d = {
        'unknown_tool': 0, 'dissector': 1, 'scissors': 2, 'suction': 3,
        'grasper 3': 4, 'harmonic': 5, 'grasper': 6, 'bipolar': 7,
        'grasper 2': 8, 'cautery (hook, spatula)': 9, 'ligasure': 10, 'stapler': 11
    }
    return d.get(str(name).strip().lower(), 0)

def to_tti_id(name):
    if name is None: return 12
    d = {
        'unknown_tti': 12, 'coagulation': 13, 'other': 14, 'retract and grab': 15,
        'blunt dissection': 16, 'energy - sharp dissection': 17, 'staple': 18,
        'retract and push': 19, 'cut - sharp dissection': 20
    }
    return d.get(str(name).strip().lower(), 12)

def filter_by_conf_and_dedup(result, conf_thr=0.35, iou_thr=0.95):
    r = result[0]
    if r.masks is None or r.boxes is None or len(r.boxes.cls)==0:
        return []
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs   = r.boxes.conf.cpu().numpy()
    masks   = r.masks.data.cpu().numpy()  # [N,h,w]
    H, W = r.masks.orig_shape

    keep = confs >= conf_thr
    classes, masks, confs = classes[keep], masks[keep], confs[keep]
    if len(classes)==0: return []

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
                if inter/uni >= iou_thr:
                    dup = True; break
            if not dup: sel.append(i)
        for i in sel:
            out.append({'class': int(classes[i]), 'mask': masks[i], 'conf': float(confs[i])})
    return out

def centroid(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None, None
    return int(xs.mean()), int(ys.mean())

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
            if d_max is not None and d > d_max: continue
            pairs.append({'tool': t, 'tissue': u})
    return pairs

def union_bbox(mask_a, mask_b):
    comb = (mask_a.astype(np.uint8) | mask_b.astype(np.uint8))
    if comb.max()==0: return None
    x, y, w, h = cv2.boundingRect(comb)
    if w==0 or h==0: return None
    return (x, y, w, h)

def build_clip_from_buffer(frame_buffer, bbox, depth_center, tool_mask_center, tissue_mask_center,
                           seq_len=5, img_size=224):
    x, y, w, h = bbox
    merged = (tool_mask_center | tissue_mask_center).astype(np.uint8)*255
    seq = []
    for fr_bgr in frame_buffer:
        rgb = cv2.cvtColor(fr_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        d   = depth_center[y:y+h, x:x+w]
        m   = merged[y:y+h, x:x+w][..., None]
        roi = np.concatenate([rgb, d[...,None], m], axis=-1)
        roi = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        roi = (roi.astype(np.float32)/255.0).transpose(2,0,1)  # [5,H,W]
        seq.append(roi)
    return np.stack(seq, axis=0)  # [T,5,224,224]

def fill_mask_poly(img_bgr, mask, color, alpha=0.5, label=None):
    poly = mask.astype(np.uint8)
    if poly.max()==0: return img_bgr
    overlay = img_bgr.copy()
    overlay[poly>0] = (overlay[poly>0]*(1-alpha) + np.array(color, dtype=np.float32)*alpha).astype(np.uint8)
    out = overlay
    if label:
        cx, cy = centroid(mask)
        if cx is not None:
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
            cv2.rectangle(out, (cx-3, cy-th-8), (cx+tw+3, cy+4), (0,0,0), -1)
            cv2.putText(out, label, (cx, cy), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return out

def draw_box_with_text(img_bgr, box, text, color=(40,200,40)):
    x, y, w, h = box
    cv2.rectangle(img_bgr, (x,y), (x+w,y+h), color, 2)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    cv2.rectangle(img_bgr, (x, y-th-8), (x+tw+8, y+4), color, -1)
    cv2.putText(img_bgr, text, (x+4, y-2), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire: {VIDEO_PATH}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (W, H))

    yolo = YOLO(YOLO_WEIGHTS)
    names = yolo.model.names if hasattr(yolo, 'model') else yolo.names

    depth_pipe = pipeline(task="depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Small-hf",
                          device=0 if DEVICE.type=='cuda' else -1)

    tcn = CNN_TCN_Classifier(sequence_length=SEQ_LEN).to(DEVICE)
    tcn.load_state_dict(torch.load(TCN_WEIGHTS, map_location=DEVICE), strict=True)
    tcn.eval()

    frame_buffer = []  
    while True:
        ok, fr_bgr = cap.read()
        if not ok: break
        frame_buffer.append(fr_bgr.copy())
        if len(frame_buffer) > SEQ_LEN:
            frame_buffer.pop(0)

        r = yolo.predict(fr_bgr, verbose=False)
        dets = filter_by_conf_and_dedup(r, CONF_THR, IOU_DUP_THR)

        tools, tissues = [], []
        for d in dets:
            cls = d['class']
            nm  = names[cls] if isinstance(names, dict) else names[cls]
            tid = to_tool_id(nm)
            xid = to_tti_id(nm)
            if tid != 0:
                d['name'] = nm
                tools.append(d)
            elif xid != 12:
                d['name'] = nm
                tissues.append(d)

        viz = fr_bgr.copy()
        for t in tools:
            viz = fill_mask_poly(viz, t['mask'], COLOR_TOOL, ALPHA_TOOL, label=t['name'])
        for u in tissues:
            viz = fill_mask_poly(viz, u['mask'], COLOR_TTI,  ALPHA_TTI,  label=u['name'])

        if len(frame_buffer) == SEQ_LEN and tools and tissues:
            depth_map = np.array(depth_pipe(Image.fromarray(cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)))['depth'],
                                 dtype=np.float32)
            pairs = pair_by_k_nearest(tools, tissues, K_NEAREST, D_MAX)

            for pr in pairs:
                tm, um = pr['tool']['mask'], pr['tissue']['mask']
                box = union_bbox(tm, um)
                if box is None: continue

                seq = build_clip_from_buffer(frame_buffer, box, depth_map, tm, um,
                                             seq_len=SEQ_LEN, img_size=IMG_SIZE)
                xb  = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    prob = torch.sigmoid(tcn(xb).squeeze(1)).item()
                pred = 1 if prob >= THR_TCN else 0

                if pred == 1:
                    draw_box_with_text(viz, box, f"TTI=1  p={prob:.2f}", COLOR_BOX_OK)

        out.write(viz)

    cap.release()
    out.release()
    print(f"[OK] Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
