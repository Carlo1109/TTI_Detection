import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base", use_safetensors=True).to(device)
processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base", use_safetensors=True)

raw_image = '../Full Dataset/train/video0590_frame72.jpg'
raw_image = Image.open(raw_image).convert("RGB")
W, H = raw_image.size

# --- CREAZIONE GRIGLIA DI PUNTI (ridotta per sicurezza) ---
num_points_x = 5
num_points_y = 5
margin_x = W * 0.07  # 5% del lato orizzontale
margin_y = H * 0.07  # 5% del lato verticale

xs = np.linspace(margin_x, W - margin_x, num_points_x)
ys = np.linspace(margin_y, H - margin_y, num_points_y)
input_points = np.array([[x, y] for y in ys for x in xs])
input_labels = np.ones(len(input_points))  # tutti positivi

# --- FUNZIONI DI VISUALIZZAZIONE ---
def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([251/255, 252/255, 30/255, 0.9])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(points, ax):
    ax.scatter(points[:, 0], points[:, 1], color='red', s=20)

# --- LOOP PER MASCHERE SINGOLE ---
for i, point in enumerate(input_points):
    # Prepara il singolo punto
    point_inputs = processor(
        raw_image,
        input_points=[[point.tolist()]],
        input_labels=[[1]],
        return_tensors="pt"
    ).to(device)
    
    # Inferenza
    with torch.no_grad():
        point_outputs = model(**point_inputs, multimask_output=False)
    
    # Post-process maschera
    mask = processor.image_processor.post_process_masks(
        point_outputs.pred_masks.sigmoid().cpu(),
        point_inputs["original_sizes"].cpu(),
        point_inputs["reshaped_input_sizes"].cpu(),
        binarize=False
    )[0]
    
    # Visualizza singola maschera
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.array(raw_image))
    color = np.concatenate([np.random.random(3), np.array([0.9])], axis=0)
    show_mask(mask, ax, color=color)
    show_points(np.array([point]), ax)
    ax.set_title(f"Maschera punto {i+1}/{len(input_points)}")
    ax.axis("off")
    plt.show()
    
    # Libera VRAM per il prossimo punto
    del point_inputs, point_outputs, mask
    torch.cuda.empty_cache()
