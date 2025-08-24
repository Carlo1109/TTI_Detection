#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================
# CONFIGURAZIONE PERCORSI
# =======================
IMAGES_DIR   = "../Dataset/evaluation/images"              
LABELS_DIR   = "../Dataset/evaluation/labels"              
YOLO_WEIGHTS = "../Common Code/runs_OLD_DATASET/segment/train/weights/best.pt"      # pesi YOLO-seg
TCN_WEIGHTS  = "model_TCN_new.pt"              # pesi modello CNN+TCN allenato

# =============
# ALTRI PARAMETRI
# =============
IMG_SIZE   = 224
SEQ_LEN    = 5
THRESHOLD  = 0.5
USE_CUDA   = True        # forza CPU mettendo False
DEPTH_DEV  = None        # None = 0 su GPU, -1 su CPU; oppure imposta un indice GPU intero

import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
from ultralytics import YOLO
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, balanced_accuracy_score, classification_report
)

# ---------------------------
# Modello CNN + TCN
# ---------------------------
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
        # x: B x T x C x H x W (C=5)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)
        x = self.pool2d(x)              # (B*T, 512, 1, 1)
        x = x.view(B, T, 512)           # (B, T, 512)
        x = x.permute(0, 2, 1)          # (B, 512, T)
        x = self.tcn(x)                 # (B, ch, T)
        x = self.pool1d(x)              # (B, ch, 1)
        out = self.classifier(x)        # (B, 1)
        return out.squeeze(1)           # (B,)

# ---------------------------
# Utilità YOLO e ROI
# ---------------------------
def load_yolo(model_path, device_index=None):
    m = YOLO(model_path, verbose=False)
    if device_index is not None and torch.cuda.is_available():
        m.to(device_index)
    return m

def yolo_detect(yolo_model, image_bgr):
    res = yolo_model.predict(image_bgr, verbose=False)[0]
    if res.masks is None or res.boxes is None or len(res.boxes.cls) == 0:
        return [], []
    classes = res.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    masks = res.masks.data.detach().cpu().numpy().astype(np.uint8)  # [N,h,w] 0/1
    return classes, masks

def find_mask_for_class(classes, masks, target_cls):
    for i, c in enumerate(classes):
        if c == target_cls:
            return masks[i]
    return None

def extract_union_roi(image_bgr, tool_mask, tissue_mask, depth_map):
    H, W = image_bgr.shape[:2]
    if tool_mask is None or tissue_mask is None:
        return None
    tool_mask = tool_mask.astype(np.uint8)
    tissue_mask = tissue_mask.astype(np.uint8)
    union = np.clip(tool_mask + tissue_mask, 0, 1).astype(np.uint8)
    if union.max() == 0:
        return None
    x, y, w, h = cv2.boundingRect(union)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)[y:y+h, x:x+w]  # HxWx3
    d = depth_map[y:y+h, x:x+w]
    if d.ndim == 2:
        d = d[..., None]
    merged_mask = (union[y:y+h, x:x+w][..., None] * 255).astype(np.uint8)
    roi = np.concatenate([rgb, d, merged_mask], axis=-1)  # HxWx5
    return roi

def build_sample_from_image(image_path, y_cls, z_cls, yolo_model, depth_pipe, img_size, seq_len):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    # YOLO
    classes, masks = yolo_detect(yolo_model, image_bgr)
    tool_mask = find_mask_for_class(classes, masks, y_cls)
    tissue_mask = find_mask_for_class(classes, masks, z_cls)
    if tool_mask is None or tissue_mask is None:
        return None
    # Depth
    pil_img = Image.open(image_path).convert("RGB")
    depth_out = depth_pipe(pil_img)
    depth_map = np.array(depth_out["depth"]).astype(np.float32)  # ~[0,1]
    # ROI 5 canali
    roi = extract_union_roi(image_bgr, tool_mask, tissue_mask, depth_map)
    if roi is None:
        return None
    # resize + normalizzazione
    roi = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    rgb = roi[..., :3] / 255.0
    depth = roi[..., 3:4]          # già ~[0,1]
    mask = roi[..., 4:5] / 255.0
    roi5 = np.concatenate([rgb, depth, mask], axis=-1)            # HxWx5
    roi5 = np.transpose(roi5, (2, 0, 1))                           # 5xHxW
    # replica temporale
    seq = np.stack([roi5 for _ in range(seq_len)], axis=0)         # T x 5 x H x W
    return torch.from_numpy(seq).float()

def read_label_file(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    for ch in [",", ";", "\t"]:
        line = line.replace(ch, " ")
    parts = [p for p in line.split() if p]
    if len(parts) < 3:
        raise ValueError(f"Label malformata: {label_path} -> '{line}'")
    x = int(parts[0])   # 0/1
    y = int(parts[1])   # classe tool
    z = int(parts[2])   # classe tissue
    return x, y, z

def main():
    # device
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Device modello: {device}")

    # YOLO
    yolo_device_index = 0 if use_cuda else None
    print("Carico YOLO…")
    yolo_model = load_yolo(YOLO_WEIGHTS, device_index=yolo_device_index)

    # Depth pipeline
    if DEPTH_DEV is None:
        depth_dev = 0 if use_cuda else -1
    else:
        depth_dev = DEPTH_DEV if use_cuda else -1
    print(f"Inizializzo depth-anything su device {depth_dev}…")
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=depth_dev
    )

    # Modello TCN
    print("Carico pesi TCN…")
    model = CNN_TCN_Classifier(sequence_length=SEQ_LEN, pretrained=False).to(device)
    state = torch.load(TCN_WEIGHTS, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    # lista immagini
    img_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [f for f in os.listdir(IMAGES_DIR) if os.path.splitext(f)[1].lower() in img_ext]
    images.sort()
    if not images:
        raise RuntimeError("Nessuna immagine trovata nella cartella IMAGES_DIR.")

    y_true, y_pred, y_score = [], [], []
    miss_any = 0

    for i, img_name in enumerate(images, 1):
        img_path = os.path.join(IMAGES_DIR, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_name)

        try:
            x, y_tool, z_tissue = read_label_file(label_path)
        except Exception as e:
            print(f"[{i}/{len(images)}] label non valida per {img_name}: {e}")
            continue

        sample = build_sample_from_image(
            img_path, y_tool, z_tissue,
            yolo_model, depth_pipe,
            IMG_SIZE, SEQ_LEN
        )

        if sample is None:
            # classi non trovate o ROI degenerata
            y_true.append(x)
            y_pred.append(0)
            y_score.append(0.0)
            miss_any += 1
            continue

        sample = sample.unsqueeze(0).to(device)  # 1 x T x 5 x H x W
        with torch.no_grad():
            prob = model(sample).clamp(0.0, 1.0).item()
        pred = 1 if prob >= THRESHOLD else 0

        y_true.append(x)
        y_pred.append(pred)
        y_score.append(prob)

        if i % 25 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] prob={prob:.3f} pred={pred} gt={x}")

    # metriche
    if len(y_true) == 0:
        raise RuntimeError("Nessuna predizione valida prodotta.")

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    print("\n=== RISULTATI ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"F1 macro:          {f1_mac:.4f}")
    print(f"F1 weighted:       {f1_w:.4f}")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall:            {rec:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print("Confusion matrix [righe=gt 0,1; colonne=pred 0,1]:")
    print(cm)
    print("\nClassification report:")
    print(report)
    print(f"\nCampioni senza ROI valida (classi mancanti o union vuota): {miss_any}")

if __name__ == "__main__":
    main()
