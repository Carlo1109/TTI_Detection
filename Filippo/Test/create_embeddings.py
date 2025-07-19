from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as T
from PIL import Image

MODEL_CFG   = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT  = "../sam2/checkpoints/sam2.1_hiera_large.pt"
IMAGES_FOLDER = '../Full Dataset/train/'
DEVICE  = "cuda"

def segment(sam_model,image):
    img = cv2.imread(IMAGES_FOLDER + image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks_info = sam_model.generate(img)
    return masks_info , img

def build_sam_mask_generator():
    sam = build_sam2(MODEL_CFG, CHECKPOINT).to(DEVICE).eval()
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam,
        points_per_side=48,    
        pred_iou_thresh=0.74,   
        stability_score_thresh=0.74, 
        box_nms_thresh=0.7
    )   
    return mask_generator


def create_embeddings():
    sam_model = build_sam_mask_generator()
    images = os.listdir(IMAGES_FOLDER)
    dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(DEVICE)
    transform = T.Compose([
                            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ])
            
    k = 0
    for image in images:
        print(f'processing image {k}/{len(images)} ----- {image}')
        masks_info , img = segment(sam_model,image)
        for i, info in enumerate(masks_info):
            mask = info["segmentation"]  
            crop = img.copy(); crop[~mask] = 0
            
            pil_crop = Image.fromarray(crop)
            img_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                feats = dinov2_vits14_reg(img_tensor)     
                # global_emb = torch.nn.functional.adaptive_avg_pool2d(feats, (1,1)).squeeze().cpu().numpy()
                
            print(feats)
            # print(global_emb)
        exit()
        
        k+=1
    
    




if __name__ == "__main__":

    create_embeddings()