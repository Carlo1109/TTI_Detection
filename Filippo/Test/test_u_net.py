import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from ultralytics import YOLO
from typing import List, Tuple
from transformers import pipeline
from PIL import Image
import torchvision.transforms as T

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = (256, 256)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT  = "teacher_cycle_1.pt"
YOLO_WEIGHTS= "./runs_OLD_DATASET/segment/train/weights/best.pt"  # il tuo file di pesi YOLOv11-seg
NUM_CLASSES = 22                    # 21 vere classi + 1 background
BG_IDX      = NUM_CLASSES - 1       # indice della classe background (da ignorare)

# ── Nomi delle classi (0–20 vere classi; 21 = background) ───────────────────────
class_names: List[str] = [
    # 0–11 strumenti
    "unknown_tool", "dissector", "scissors", "suction", "grasper 3", "harmonic",
    "grasper", "bipolar", "grasper 2", "cautery (hook, spatula)", "ligasure", "stapler",
    # 12–20 TTI id
    "unknown_tti", "coagulation", "other", "retract and grab", "blunt dissection",
    "energy - sharp dissection", "staple", "retract and push", "cut - sharp dissection",
    # 21 background
    "background"
]
# assert len(class_names) == NUM_CLASSES

# ── Funzioni UNet ────────────────────────────────────────────────────────────────
def load_unet(ckpt_path: str) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=DEVICE)
    sd = state.get("model_state_dict", state)
    model = smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    )
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()

def preprocess_image(img_path: str) -> torch.Tensor:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    # depth_map = np.array(pipe(Image.fromarray(img))["depth"])
    # img = np.concatenate([img, depth_map[..., None]], axis=-1) 
    
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)
    return tensor.unsqueeze(0).to(DEVICE)

def infer_mask_unet(model: torch.nn.Module, input: torch.Tensor) -> np.ndarray:
    
    img = cv2.imread(input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    tf_img = T.Compose([
            T.ToPILImage(),
            T.Resize(IMG_SIZE,  interpolation=Image.BILINEAR),
            T.ToTensor(),  # [0,1] e C×H×W
            T.Normalize(mean=(0.485,0.456,0.406),
                        std=(0.229,0.224,0.225)),
        ])
    inp = tf_img(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        preds = model(inp)
        probs = torch.softmax(model(inp), dim=1)  # [1, C, H_out, W_out]
    conf, preds_1 = probs.max(1)      
    pred_mask = preds_1.squeeze(0).cpu().numpy().astype(np.int8)  # H_out×W_out, valori 0…C-1
        
    mask = torch.nn.functional.interpolate(
        preds,size = img.shape[:2], mode='bilinear',align_corners=False
        
    )
    conf_map  = conf.squeeze(0).cpu().numpy() 
    mask_filtered = np.full_like(pred_mask, fill_value=2, dtype=np.int8)

    H,W,_ = img.shape
    for c in np.unique(pred_mask):
        if c == 2: 
            continue  
        
        class_pixels = (pred_mask == c)
        mean_conf = conf_map[class_pixels].mean()
        if mean_conf >= 0.8:
            mask_filtered[class_pixels] = c
        
    
    mask_np = cv2.resize(mask_filtered, (W, H), interpolation=cv2.INTER_NEAREST)

    
    # return mask_np
    return mask[0].argmax(0).cpu().numpy()

# ── Funzioni YOLOv11-seg ─────────────────────────────────────────────────────────
def load_yolo(weights_path: str):
    """Carica il tuo YOLOv11-seg addestrato via ultralytics."""
    return YOLO(weights_path)

def infer_mask_yolo(model, img_path: str) -> np.ndarray:
    """
    Inferisce maschere da YOLOv11-seg e restituisce una singola mappa H×W di classi.
    """
    results = model(img_path)[0]
    # masks.data: [N, H, W] boolean; boxes.cls: [N] class id per istanza
    masks = results.masks.data.cpu().numpy()      # [N, H, W]
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    # Combina maschere in un'unica mappa
    H, W = masks.shape[1], masks.shape[2]
    mask_comb = np.full((H, W), BG_IDX, dtype=np.uint8)
    for i, cid in enumerate(cls_ids):
        mask_comb[masks[i].astype(bool)] = cid
    return mask_comb

# ── Utility color & overlay ─────────────────────────────────────────────────────
def generate_colors(n: int) -> List[Tuple[int,int,int]]:
    cmap = plt.get_cmap("tab20", n)
    return [(int(r*255), int(g*255), int(b*255)) for r,g,b,_ in cmap(np.arange(n))]

def colorize_mask(mask: np.ndarray, colors: List[Tuple[int,int,int]]) -> np.ndarray:
    h, w = mask.shape
    col = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, c in enumerate(colors):
        if idx == BG_IDX:
            continue
        col[mask == idx] = c
    return col

def make_overlay(orig_bgr: np.ndarray, col_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Crea overlay tra immagine RGB e mask colorata."""
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = orig_rgb.shape[:2]
    col_resized = cv2.resize(col_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(orig_rgb, 1-alpha, col_resized, alpha, 0)

# ── Main: confronto fianco a fianco ───────────────────────────────────────────
if __name__ == "__main__":
    img_fp = "../../Dataset/evaluation/images/video0001_frame0150.png"
    orig_bgr = cv2.imread(img_fp)

    # UNet inference
    unet = load_unet(CHECKPOINT)
    
    mask_unet = infer_mask_unet(unet,img_fp)

    # YOLOv11-seg inference
    yolo = load_yolo(YOLO_WEIGHTS)
    mask_yolo = infer_mask_yolo(yolo, img_fp)

    # Palette e overlay
    palette = generate_colors(NUM_CLASSES)
    col_unet = colorize_mask(mask_unet, palette)
    col_yolo = colorize_mask(mask_yolo, palette)
    overlay_unet = make_overlay(orig_bgr, col_unet)
    overlay_yolo = make_overlay(orig_bgr, col_yolo)

    # Visualizza
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(mask_unet); axes[0].set_title("UNet Segmentation"); axes[0].axis("off")
    axes[1].imshow(overlay_yolo); axes[1].set_title("YOLOv11-seg Segmentation"); axes[1].axis("off")
    plt.tight_layout()
    plt.show()
