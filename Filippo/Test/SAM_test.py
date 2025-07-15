import torch
import clip
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ── CONFIGURAZIONE ─────────────────────────────────────────
IMAGE_PATH = "../../Dataset/evaluation/images/video0000_frame0051.png"
MODEL_CFG   = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT  = "../sam2/checkpoints/sam2.1_hiera_large.pt"
DEVICE      = "cuda"  # o "cpu"



img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

model, preprocess = clip.load("ViT-B/32", device=DEVICE )
image = preprocess(Image.fromarray(img)).unsqueeze(0).to(DEVICE )
text = clip.tokenize(["surgical instrument", "biological tissue"]).to(DEVICE )



# ── BUILD MODEL ─────────────────────────────────
sam = build_sam2(MODEL_CFG, CHECKPOINT).to(DEVICE).eval()

# ── AUTOMATIC MASK GENERATOR ────────────────────────────────
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam,
    points_per_side=32,    
    pred_iou_thresh=0.8,   
    stability_score_thresh=0.8, 
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
    
    with torch.no_grad():
        crop = img.copy(); crop[~mask] = 0
        pil_crop    = Image.fromarray(crop)
        img_input   = preprocess(pil_crop).unsqueeze(0).to(DEVICE)
        image_features = model.encode_image(img_input)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(img_input , text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)

    score = info["predicted_iou"]
    color = np.array(cmap(i)[:3])      # colore RGB

    # Applichiamo la trasparenza solo dove mask==True
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    # Visualizziamo il risultato unico
    plt.figure(figsize=(8, 8))
    # plt.imshow((overlay * 255).astype(np.uint8))
    plt.imshow(pil_crop)
    plt.axis("off")
    plt.title("All detected masks")
    plt.show()