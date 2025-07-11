import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIGURAZIONE ─────────────────────────────────────────
IMAGE_PATH = "../../Dataset/evaluation/images/video0036_frame0032.png"
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
    points_per_side=32,    
    pred_iou_thresh=0.75,   
    stability_score_thresh=0.75, 
    box_nms_thresh=0.7
)

# ── GENERA TUTTE LE MASCHERE ZERO‑SHOT ──────────────────────
masks_info = mask_generator.generate(img)  

# ── VISUALIZZAZIONE DI OGNI MASCHERA ────────────────────────
# Parametri
alpha = 0.5
n_masks = len(masks_info)
cmap = plt.cm.get_cmap("tab20", n_masks)

# Partiamo dall’immagine originale normalizzata
base = img.copy().astype(float) / 255.0
overlay = base.copy()

# Per ciascuna maschera, sovrapponila all’overlay
for i, info in enumerate(masks_info):
    mask = info["segmentation"]        # boolean array H×W
    score = info["predicted_iou"]
    color = np.array(cmap(i)[:3])      # colore RGB

    # Applichiamo la trasparenza solo dove mask==True
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

# Visualizziamo il risultato unico
plt.figure(figsize=(8, 8))
plt.imshow((overlay * 255).astype(np.uint8))
plt.axis("off")
plt.title("All detected masks")
plt.show()