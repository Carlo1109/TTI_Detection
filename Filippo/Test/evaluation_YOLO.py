import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image

import os
import random
from ultralytics import YOLO


IMG_SIZE    = (256, 256)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT  = './runs_sampled_dataset/segment/train/weights/best.pt'





def expand_mask(mask, pixels):
    kernel_size = 2 * pixels + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return expanded


def vis(model,im,depth_model):

    preds = model.predict(im,verbose=False)
    img = cv2.imread(im)
    print(img.shape)
    
    tool_list = []
    tissue_list = []
    
   
    for result in preds:
        # get array results
        if result.masks is None:
            return
        masks = result.masks.data
        boxes = result.boxes.data
        # extract classes
        clss = boxes[:, 5]
        
        for i in range(masks.shape[0]):
            mask_resized = cv2.resize(masks[i].cpu().numpy().astype('uint8'), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            print(mask_resized.shape)
            if clss[i] == 1:
                tool_list.append(mask_resized)
            else:
                tissue_list.append(mask_resized)
        
           
    pairs = []
    for tool in tool_list:
        print("tool found")
        tool_mask = tool
        depth_map = np.array(depth_model(im)["depth"]) / 255
        tool_depth = depth_map[tool_mask.astype(bool)]
        
        med_depth = np.percentile(tool_depth,15)
        
        t = np.logical_and(tool_mask , depth_map < med_depth)
        
        tool_mask = t


        for tissue in tissue_list:
            tissue_mask = tissue
            # plt.imshow(tissue_mask)
            # plt.show()
            # plt.imshow(expand_mask(tissue_mask,10))
            # plt.show()
            
            intersection_mask = np.logical_and(expand_mask(tool_mask,10),expand_mask(tissue_mask,10))
            inter_numb = intersection_mask.sum()
            
            plt.imshow(img)
            plt.imshow(expand_mask(np.ma.masked_where(tool_mask == 0, tool_mask)  ,10),alpha=0.7)
            plt.imshow(expand_mask(np.ma.masked_where(tissue_mask == 0, tissue_mask),10),alpha=0.7)
            plt.show()
            
            if inter_numb > 15:
                plt.imshow(img)
                plt.imshow(np.ma.masked_where(intersection_mask == 0, intersection_mask)  ,alpha=0.8)
                plt.title("PRE EXPAND")
                plt.show()
                
                intersection_mask = expand_mask(intersection_mask,20)
                
                plt.imshow(img)
                plt.imshow(np.ma.masked_where(intersection_mask == 0, intersection_mask),alpha=0.8)
                plt.show()
                
                plt.imshow(depth_map)
                plt.show()
                
                tool_int = np.logical_and(intersection_mask.astype(bool),tool_mask.astype(bool))
                tissue_int = np.logical_and(intersection_mask.astype(bool),tissue_mask.astype(bool))
                
                if not np.any(tool_int) or not np.any(tissue_int):
                    continue
  
                # plt.imshow(tool_int)
                # plt.show()
                # plt.imshow(tissue_int)
                # plt.show()
                
                depth_tool_int = depth_map[tool_int.astype(bool)]
                depth_tissue_int = depth_map[tissue_int.astype(bool)]
                
                #depth median
                med_tool = np.mean(depth_tool_int)
                med_tissue = np.mean(depth_tissue_int)
                
                print(med_tool)
                print(med_tissue)
                
                if np.isnan(med_tool) or np.isnan(med_tissue):
                    continue

                
                tolerance = 0.06
    
                
                if np.abs(med_tool - med_tissue) <= tolerance:
                    pairs.append((tool_mask,tissue_mask))
                
        
        
    if len(pairs) == 0:
        print("NO TTI FOUND")
    for pair in pairs:  
        plt.imshow(img)
        to = np.ma.masked_where(pair[0] == 0, pair[0])  
        ti = np.ma.masked_where(pair[1] == 0, pair[1])  
        plt.imshow(to,alpha=0.5, cmap='jet')
        plt.imshow(ti,alpha=0.5,cmap='Greens')
        
        # plt.gca().add_patch(rect)

        plt.title(f" TTI ")
        plt.axis("off")
        plt.show()
        
    


if __name__ == '__main__':
    model = YOLO(CHECKPOINT)
    img_fp = "../Full Dataset/val/video0194_frame104.jpg"
    sd = torch.load(CHECKPOINT, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    depth = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    
    images = os.listdir("../Full Dataset/val/")
    random.shuffle(images)
    
    for img_fp in images:
        img_fp = os.path.join('../Full Dataset/val/',img_fp)
        vis(model,img_fp,depth)