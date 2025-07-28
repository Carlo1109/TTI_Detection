from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import random

MODEL_CFG   = "../sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT  = "../sam2/checkpoints/sam2.1_hiera_small.pt"
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
        pred_iou_thresh=0.88,   
        stability_score_thresh=0.88, 
        box_nms_thresh=0.85
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
    masks = 0
    random.shuffle(images)
    all_embeddings = []
    for image in images:
        print(f'processing image {k}/{len(images)} ----- {image}')
        masks_info , img = segment(sam_model,image)
        for i, info in enumerate(masks_info):
            mask = info["segmentation"]  
            crop = img.copy(); crop[~mask] = 0
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

            crop = crop[y:y+h, x:x+w]
        
            pil_crop = Image.fromarray(crop)
            img_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                feats = dinov2_vits14_reg(img_tensor).squeeze(0).cpu().numpy()
                # print(feats.shape)
                plt.imshow(crop)
                plt.show()
                lab = int(input("1 for tool, 0 for tissue "))
                if lab > 1:
                    continue
                compl_feat = np.append(feats,lab)
                # print(compl_feat.shape)
                # print(compl_feat)
                
       
            all_embeddings.append(compl_feat)
            masks += 1
            print("mask n: ",masks)
            
        if masks >= 300:
            all_embeddings = np.array(all_embeddings)
            np.save("embedding_class.npy",all_embeddings)
            print(all_embeddings.shape)
            exit()
            # print(feats)
        
        
        k+=1
    return np.array(all_embeddings)
    
    




if __name__ == "__main__":

    emb = create_embeddings()
    np.save('embeddings.npy',emb)