import os, re, json, cv2, torch, numpy as np
from PIL import Image
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from transformers import pipeline
from torchvision.models import resnet18

TEST_VIDEOS_DIR = '../Dataset/video_dataset/videos/test/'   
TEST_LABELS_DIR = '../Dataset/video_dataset/labels/test/'   
IMG_SIZE        = 224
SEQ_LEN         = 5
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

YOLO_WEIGHTS    = '../Common Code/runs_OLD_DATASET/segment/train/weights/best.pt'  
TCN_WEIGHTS     = 'model_TCN_V3.pt'                              
USE_DEPTH    = True   
VERBOSE      = True

# Parametri modello/pipeline
IMG_SIZE   = 224
SEQ_LEN    = 5
CONF_THR   = 0.45     # filtro confidenza YOLO
IOU_DUP_THR= 0.90     # dedup maschere sovrapposte
D_MAX      = 30       # px max per pairing "nearest"
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# UTILS COMUNI
# =========================
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
    return fr  # BGR o RGB

def get_polygon(poly_json):
    s = ''
    for v in poly_json.values():
        s += f" {v['x']} {v['y']}"
    return s

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

def extract_union_bbox_from_poly(tool_poly_pts, tissue_poly_pts, H, W):
    bin_t = np.zeros((H, W), np.uint8)
    if tool_poly_pts is not None and tool_poly_pts.size:
        cv2.fillPoly(bin_t, [tool_poly_pts], 1)
    bin_u = np.zeros((H, W), np.uint8)
    if tissue_poly_pts is not None and tissue_poly_pts.size:
        cv2.fillPoly(bin_u, [tissue_poly_pts], 1)
    comb = (bin_t | bin_u).astype(np.uint8)
    if comb.max() == 0:
        return None, None, None
    x, y, w, h = cv2.boundingRect(comb)
    return (x, y, w, h), bin_t, bin_u

def make_clip_from_bbox(cap, idx_center, bbox, depth_map_center, tool_bin_center, tissue_bin_center,
                        IMG_SIZE=224, SEQ_LEN=5):
    x, y, w, h = bbox
    seq = []
    for t in range(idx_center-(SEQ_LEN-1), idx_center+1):
        fr_bgr = _load_frame(cap, t, rgb=False)
        rgb = cv2.cvtColor(fr_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        dpatch = depth_map_center[y:y+h, x:x+w]
        mpatch = (tool_bin_center[y:y+h, x:x+w] | tissue_bin_center[y:y+h, x:x+w]).astype(np.uint8)*255
        roi = np.concatenate([rgb, dpatch[...,None], mpatch[...,None]], axis=-1)  # [H,W,5]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        roi = roi.astype(np.float32) / 255.0
        seq.append(np.transpose(roi, (2,0,1)))  # C,H,W
    return np.stack(seq, axis=0)  # T,C,H,W


# =========================
# MODELLO (IDENTICO AL TRAINING)
# =========================
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
            backbone.conv1.weight[:, :3] = old_w                      # copia RGB
            backbone.conv1.weight[:, 3:5] = old_w.mean(dim=1, keepdim=True)  # media per depth+mask

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
            nn.Linear(tcn_channels[-1], num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool2d(x)
        x = x.view(B, T, 512)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.pool1d(x)
        out = self.classifier(x)  # logit
        return out


# =========================
# COSTRUZIONE COPPIE/LABEL — COMPATIBILE CON IL TUO create_train()
# =========================
def build_pairs_from_labels(objs, H, W):
    """
    Replica la tua logica:
      - len==2: un instrument + un tissue con is_tti==1  -> positivo (1)
      - len==3: tissue is_tti==1 + instrument interagente (1) + instrument non-interagente (0)
      - len>=4: per ogni instrument x ogni tissue positivo -> label 1 se instrument_type == interaction_tool, altrimenti 0
    Ritorna: lista di tuple (tool_poly_pts, tissue_poly_pts, label, tool_name, interaction_tool_name)
    """
    L = len(objs)
    pairs = []
    if L < 2:
        return pairs

    def mask_from_instr(o):
        if 'instrument_polygon' not in o: return None
        return parse_mask_string(get_polygon(o['instrument_polygon']), H, W)

    def mask_from_tti(o):
        if 'tti_polygon' not in o: return None
        return parse_mask_string(get_polygon(o['tti_polygon']), H, W)

    if L == 2:
        tool_poly = tissue_poly = None
        tool_name = inter_name = None
        for o in objs:
            if 'is_tti' in o:
                if int(o.get('is_tti',0)) == 1:
                    tissue_poly = mask_from_tti(o)
                    inter_name  = o.get('interaction_tool')
            else:
                tool_poly = mask_from_instr(o)
                tool_name = o.get('instrument_type')
        if tool_poly is not None and tissue_poly is not None:
            pairs.append((tool_poly, tissue_poly, 1, tool_name, inter_name))

    elif L == 3:
        tool_poly = tissue_poly = non_tool_poly = None
        tool_name = inter_name = non_tool_name = None
        for o in objs:
            if 'is_tti' in o:
                if int(o.get('is_tti',0)) == 1:
                    tissue_poly = mask_from_tti(o)
                    inter_name  = o.get('interaction_tool')
                else:
                    # trattato come instrument non-interagente
                    if 'instrument_polygon' in o:
                        non_tool_poly = mask_from_instr(o)
                        non_tool_name = o.get('instrument_type')
            else:
                tool_poly = mask_from_instr(o)
                tool_name = o.get('instrument_type')

        if tool_poly is not None and tissue_poly is not None:
            pairs.append((tool_poly, tissue_poly, 1, tool_name, inter_name))
        if non_tool_poly is not None and tissue_poly is not None:
            pairs.append((non_tool_poly, tissue_poly, 0, non_tool_name, inter_name))

    else:  # L >= 4
        pos_tissues = []
        instruments = []
        for o in objs:
            if 'is_tti' in o and int(o.get('is_tti',0)) == 1:
                tp = mask_from_tti(o)
                inter_name = o.get('interaction_tool')
                if tp is not None:
                    pos_tissues.append((tp, inter_name))
        for o in objs:
            if 'is_tti' not in o and 'instrument_polygon' in o:
                ip = mask_from_instr(o)
                name = o.get('instrument_type')
                if ip is not None:
                    instruments.append((ip, name))
        for (tp, inter_name) in pos_tissues:
            for (ip, name) in instruments:
                lab = 1 if (inter_name == name) else 0
                pairs.append((ip, tp, lab, name, inter_name))

    return pairs  # [(tool_poly, tissue_poly, label, tool_name, inter_name)]


# =========================
# ORACLE (SOLO CLASSIFICATORE) — COMPATIBILE CON LA LOGICA SOPRA
# =========================
def evaluate_oracle(depth_pipe, model, labels_dir, videos_dir, IMG_SIZE=224, SEQ_LEN=5):
    model.eval()
    y_true, y_prob = [], []

    videos = [v for v in os.listdir(videos_dir) if not v.startswith('.')]
    for vid in videos:
        vpath = os.path.join(videos_dir, vid)
        cap, fcount = _load_video(vpath)
        key = normalize(os.path.splitext(vid)[0])
        jpath = next((os.path.join(labels_dir, f)
                      for f in os.listdir(labels_dir)
                      if normalize(os.path.splitext(f)[0]) == key), None)
        if jpath is None:
            cap.release(); continue
        labels = json.load(open(jpath, 'r'))['labels']

        print(f"[ORACLE] Video: {vid}  ({fcount} frames)")

        for idx_s, objs in labels.items():
            idx = int(idx_s)
            if idx - (SEQ_LEN - 1) < 0 or idx >= fcount:
                continue

            frame_c_bgr = _load_frame(cap, idx, rgb=False)
            H, W = frame_c_bgr.shape[:2]

            # depth del frame centrale UNA volta
            depth_map = np.array(
                depth_pipe(Image.fromarray(cv2.cvtColor(frame_c_bgr, cv2.COLOR_BGR2RGB)))['depth'],
                dtype=np.float32
            )

            pairs = build_pairs_from_labels(objs, H, W)
            if not pairs:
                continue

            print(f"   Frame {idx}/{fcount} — coppie GT generate: {len(pairs)}")

            for (tool_poly, tissue_poly, label, tool_name, inter_name) in pairs:
                bbox, tool_bin, tissue_bin = extract_union_bbox_from_poly(tool_poly, tissue_poly, H, W)
                if bbox is None:
                    continue
                seq = make_clip_from_bbox(cap, idx, bbox, depth_map, tool_bin, tissue_bin,
                                          IMG_SIZE=IMG_SIZE, SEQ_LEN=SEQ_LEN)
                xb = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # [1,T,C,H,W]
                with torch.no_grad():
                    logit = model(xb).squeeze(1)
                    prob  = torch.sigmoid(logit).item()
                y_true.append(label)
                y_prob.append(prob)

        cap.release()

    if not y_true:
        print("ORACLE: nessun sample.")
        return 0.5, {}

    # sweep soglia per best F1
    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = [1 if p >= thr else 0 for p in y_prob]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    yp50 = [1 if p >= 0.5 else 0 for p in y_prob]
    metrics_50 = dict(
        acc=accuracy_score(y_true, yp50),
        f1=f1_score(y_true, yp50, zero_division=0),
        prec=precision_score(y_true, yp50, zero_division=0),
        rec=recall_score(y_true, yp50, zero_division=0),
        ba=balanced_accuracy_score(y_true, yp50),
        cm=confusion_matrix(y_true, yp50).tolist()
    )
    ypB = [1 if p >= best_thr else 0 for p in y_prob]
    metrics_best = dict(
        acc=accuracy_score(y_true, ypB),
        f1=f1_score(y_true, ypB, zero_division=0),
        prec=precision_score(y_true, ypB, zero_division=0),
        rec=recall_score(y_true, ypB, zero_division=0),
        ba=balanced_accuracy_score(y_true, ypB),
        cm=confusion_matrix(y_true, ypB).tolist()
    )

    print("\n===== ORACLE (solo classificatore, logica compatibile) =====")
    print(f"samples={len(y_true)}  pos={int(np.sum(y_true))}")
    print(f"@0.5  acc={metrics_50['acc']:.3f}  f1={metrics_50['f1']:.3f}  prec={metrics_50['prec']:.3f}  rec={metrics_50['rec']:.3f}  ba={metrics_50['ba']:.3f}")
    print(f"@best acc={metrics_best['acc']:.3f}  f1={metrics_best['f1']:.3f}  prec={metrics_best['prec']:.3f}  rec={metrics_best['rec']:.3f}  ba={metrics_best['ba']:.3f}  thr≈{best_thr:.2f}")
    print("CM @best:", metrics_best['cm'])

    return float(best_thr), metrics_best

if __name__ == '__main__':
    depth_pipe = pipeline(task="depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Small-hf",
                          device=0 if DEVICE.type=='cuda' else -1)
    model = CNN_TCN_Classifier(sequence_length=SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load(TCN_WEIGHTS, map_location=DEVICE), strict=True)

    # 2) ORACLE: ricava soglia ottimale
    best_thr, _ = evaluate_oracle(depth_pipe, model, TEST_LABELS_DIR, TEST_VIDEOS_DIR)
    print(f"\nSoglia TCN suggerita dall'ORACLE: {best_thr:.2f}")