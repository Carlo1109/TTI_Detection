import cv2
import os
import random
import numpy as np
import torch
from ultralytics import YOLO

VIDEO_FOLDER = '../../Dataset/video_dataset/videos/train'
OUTPUT_FOLDER = './medSAM2_dataset/images/'
MODEL_PATH = './runs/segment/train/weights/best.pt'
MODEL_PATH_TOOLS = './runs_fine_tuned/segment/train/weights/best.pt'
MASK_PATH = './medSAM2_dataset/masks/'
MAX_VIDEO = 400


def _load_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_count


def save_mask(model,model_tool,img,output_dir):
    img_path = img
    res = model.predict(img_path,verbose=False)
    
    res_tool = model_tool.predict(img_path,verbose=False)
    
    first_net_empty = False
    if res is None or len(res_tool) == 0:
        first_net_empty = True
    
    for result in res:
        # get array results
        if result.masks is None:
            first_net_empty = True
            break
        masks = result.masks.data
        boxes = result.boxes.data
        # extract classes
        clss = boxes[:, 5]
        out_mask = torch.zeros_like(masks[0], dtype=torch.uint8)
        
        next_class = 0
        for i in range(masks.shape[0]):
            if clss[i] == 1:
                mask_i = masks[i] > 0  
                out_mask[mask_i] = 1
            else:
                mask_i = masks[i] > 0  
                out_mask[mask_i] = i + 2
                next_class = i + 2
    
    
    
    result_tool = res_tool[0]
    if result_tool is None or len(result_tool) == 0:
        if first_net_empty:
            return
        cv2.imwrite(f'{output_dir}/prompt_mask.png', out_mask.cpu().numpy())
        return
    masks_tool = result_tool.masks.data
    boxes_tool = result_tool.boxes.data
    if first_net_empty:
        out_mask = torch.zeros_like(masks_tool[0], dtype=torch.uint8)
        next_class = 2
    clss_tool = boxes_tool[:, 5]

    for i in range(masks_tool.shape[0]):
        if clss_tool[i] == 1:
            mask_i = masks_tool[i] > 0  
            out_mask[mask_i] = 1
        else:
            mask_i = masks_tool[i] > 0          
            if not torch.all(out_mask[mask_i] != 0):
                out_mask[mask_i] = i + next_class
    
    
    
    cv2.imwrite(f'{output_dir}/prompt_mask.png', out_mask.cpu().numpy())   





def create_datatset():
    model = YOLO(MODEL_PATH)
    model_tool = YOLO(MODEL_PATH_TOOLS)
    videos = os.listdir(VIDEO_FOLDER)
    random.shuffle(videos)
    
    i = 1
    
    # done = os.listdir('./medSAM2_dataset/images/')
    done = []
    
    for video in videos:
        if i == MAX_VIDEO:
            break
        if video in done:
            continue
        
        full_path = os.path.join(OUTPUT_FOLDER,video.replace('.mp4',''))
        os.mkdir(full_path)
        mask_path = os.path.join(MASK_PATH,video.replace('.mp4',''))
        os.mkdir(mask_path)
        
        
        print(f'Processing Video {i}/{len(videos)}')
        
        
        cap, frame_count = _load_video(os.path.join(VIDEO_FOLDER, video))
        
        count = 0
        success = True
        while success:
            success,image = cap.read()
            if success:
                cv2.imwrite(full_path +'/' + f'frame{count:03d}.jpg',image)
                count+=1
        
        i+=1
        
        save_mask(model,model_tool,os.path.join(full_path,'frame000.jpg'),mask_path + '/')
        
        
       

    


if __name__ == "__main__":
    create_datatset()