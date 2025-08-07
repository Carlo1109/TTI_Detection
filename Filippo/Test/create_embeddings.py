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
from transformers import pipeline
from pycocotools import mask as mask_utils
import json
from sklearn import svm

MODEL_CFG   = "../sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT  = "../sam2/checkpoints/sam2.1_hiera_base_plus.pt"
IMAGES_FOLDER = '../Full Dataset/train/'
DEVICE  = "cuda"

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0  

    return intersection / union



def is_glare(mask, image, glare_thresh=230):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    vals = gray[mask]
    bright_count = np.count_nonzero(vals > glare_thresh)
    frac = bright_count / vals.size
    if vals.size == 0:
        return True
    
    if frac > 0.17:
        return True
    return False

def is_dark(mask, image, dark_thresh=35):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    vals = gray[mask]
    dark_count = np.count_nonzero(vals < dark_thresh)
    frac = dark_count / vals.size
    if vals.size == 0:
        return True
 
    if frac > 0.85:
        return True
    return False




def segment(sam_model,image):
    img = cv2.imread(IMAGES_FOLDER + image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks_info = sam_model.generate(img)

    masks_to_return = []
    used = []
    print("MASK NUMBER: " , len(masks_info))
 
    print(img.shape)
    
    img_area = img.shape[0]*img.shape[1]
    for i, info in enumerate(masks_info):
        if i in used:
            continue
        mask = info["segmentation"]  
        
        area = mask.sum()

        glare = is_glare(mask,img)
        dark = is_dark(mask,img)

        if area < 1800 or area >= img_area/2  or glare or dark:
            continue
        
        found = False
        for k in range(i+1,len(masks_info)):
            if k in used:
                continue
                
            mask2 = masks_info[k]['segmentation']
            iou = compute_iou(mask, mask2)
            
            if iou > 0.9 and not found:
                combined_mask = np.logical_or(mask, mask2)
                found = True
                used.append(k)
            elif iou > 0.9 and found:
                combined_mask = np.logical_or(mask, combined_mask)
                used.append(k)
        
        if found:
            masks_to_return.append(combined_mask)

        elif i not in used:
            masks_to_return.append(mask)
            
        used.append(i)
    print("RETURNED MASKS: ", len(masks_to_return))
    return masks_to_return , img




def build_sam_mask_generator():
    sam = build_sam2(MODEL_CFG, CHECKPOINT).to(DEVICE).eval()
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam,
        points_per_side=36,    
        pred_iou_thresh=0.75,   
        stability_score_thresh=0.75, 
        box_nms_thresh=0.75
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
            mask = info
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
    
    
def train_svm():
    

    svm_dataset =  np.load('embedding_classification.npy')

    X = []
    y = []
    for e in svm_dataset:
        X.append(e[:-1])
        y.append(e[-1])

    clf = svm.SVC()
    clf.fit(X, y)
    
    return clf


MAX_IMAGES = 20

def test():
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

    emb = []

    clf = train_svm()
    random.shuffle(images)
    os.makedirs('train_masks',exist_ok=True)
    os.makedirs('images',exist_ok=True)
    for image in images:
        print(f'processing image {k}/{len(images)} ----- {image}')
        if k >= MAX_IMAGES:
            break
        masks_info , img = segment(sam_model,image)
        H,W,_ = img.shape

        annotations = []
        for i, info in enumerate(masks_info):
            mask = info  
            crop = img.copy()
            crop[~mask] = 200
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

            crop = crop[y:y+h, x:x+w]

            pil_crop = Image.fromarray(crop)
            img_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                feats = dinov2_vits14_reg(img_tensor).squeeze(0).cpu().numpy()   
            cluster = clf.predict(feats.reshape(1, -1))[0]

            mask = mask.astype(np.uint8)

            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            annotations.append({
                "boxes": [x, y, x+w, y+h],
                "labels": int(cluster)+1,
                "mask": {
                        "height": mask.shape[0],
                        "width":  mask.shape[1],
                        "rle":    rle
                    }
            })
            

        # cv2.imwrite(f'./train_masks/{image.replace(".jpg", ".png")}', empty_mask.astype(np.uint8))
        output_path = f'./train_masks/{image.replace(".jpg", ".json")}'
        with open(output_path, "w") as f:
            json.dump(annotations, f)
        cv2.imwrite(f'./images/{image.replace(".jpg", ".png")}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

         #        plt.imshow(crop)
         #        plt.title(f"Cluster: {cluster}")
         #        plt.axis("off")
         #        plt.show()

        k+=1










if __name__ == "__main__":

    # emb = create_embeddings()
    # np.save('embeddings.npy',emb)
    test()