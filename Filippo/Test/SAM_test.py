import torch
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIGURAZIONE ─────────────────────────────────────────
IMAGE_PATH = "../../Dataset/evaluation/images/video0000_frame0051.png"
# MODEL_CFG   = "../sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT  = "../MedSAM/work_dir/MedSAM/smedsam_vit_b.pth"
DEVICE      = "cuda"  # o "cpu"



img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── BUILD MODEL ─────────────────────────────────
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(DEVICE).eval()
predictor = SamPredictor(sam)

# ── PROMPT E PREDICT ────────────────────────────────────────
predictor.set_image(img)
points = np.array([
    [37, 201],   
    [170, 445],  
    [624, 210],  
])
labels = np.array([1, 1, 1])  

masks, scores, _ = predictor.predict(
    point_coords=points,
    point_labels=labels,
    multimask_output=True
)

alpha = 0.5

for i, mask in enumerate(masks):
    overlay = img.copy().astype(float) / 255.0  # normalize
    
    color = np.array([0.0, 0.0, 1.0])  # rosso

    overlay[mask == 1] = (1 - alpha) * overlay[mask == 1] + alpha * color

    overlay_uint8 = (overlay * 255).astype(np.uint8)

    
    plt.figure(figsize=(6,6))
    plt.imshow(overlay_uint8)
    plt.axis('off')
    plt.title(f"Maschera {i+1} sovrapposta (score={scores[i]:.2f})")
    plt.show()