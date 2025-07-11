import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIGURAZIONE ─────────────────────────────────────────
IMAGE_PATH = "../../Dataset/evaluation/images/video0404_frame0082.png"
MODEL_CFG   = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT  = "../sam2/checkpoints/sam2.1_hiera_large.pt"
DEVICE      = "cuda"  # o "cpu"



img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── BUILD MODEL ─────────────────────────────────
sam = build_sam2(MODEL_CFG, CHECKPOINT).to(DEVICE).eval()

# ── AUTOMATIC MASK GENERATOR ────────────────────────────────
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam,
    points_per_side=32,     # risoluzione della griglia di punti
    pred_iou_thresh=0.75,   # soglia di confidenza minima
    stability_score_thresh=0.75, 
    box_nms_thresh=0.7
)

# ── GENERA TUTTE LE MASCHERE ZERO‑SHOT ──────────────────────
masks_info = mask_generator.generate(img)  
# masks_info è lista di dict, ogni dict contiene almeno 'segmentation' (H×W bool mask) e 'predicted_iou'

# ── VISUALIZZAZIONE DI OGNI MASCHERA ────────────────────────
alpha  = 0.5
colors = plt.cm.get_cmap("tab20", len(masks_info))

for i, info in enumerate(masks_info):
    mask  = info["segmentation"]  # array booleano H×W
    score = info["predicted_iou"]

    # overlay colore
    overlay = img.copy().astype(float) / 255.0
    color   = np.array(colors(i)[:3])  # RGB da colormap

    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    # visualizza
    plt.figure(figsize=(6,6))
    plt.imshow((overlay * 255).astype(np.uint8))
    plt.axis("off")
    plt.title(f"Maschera {i+1} (IoU_pred={score:.2f})")
    plt.show()