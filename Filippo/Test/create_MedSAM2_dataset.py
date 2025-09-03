import cv2
import os
import random
import torch
from ultralytics import YOLO

VIDEO_FOLDER = '../../Dataset/video_dataset/videos/train'
OUTPUT_FOLDER = './medSAM2_dataset/images/'
MODEL_PATH = './runs/segment/train/weights/best.pt'
MASK_PATH = './medSAM2_dataset/masks/'
MAX_VIDEO = 100


def _load_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_count


def save_mask(model,img,output_dir):
    img_path = img
    res = model.predict(img_path,verbose=False)
    if res is None or len(res) == 0:
        return
    
    for result in res:
        # get array results
        if result.masks is None:
            return
        masks = result.masks.data
        boxes = result.boxes.data
        # extract classes
        clss = boxes[:, 5]
        out_mask = torch.zeros_like(masks[0], dtype=torch.uint8)
        
        for i in range(masks.shape[0]):
            if clss[i] == 1:
                mask_i = masks[i] > 0  
                out_mask[mask_i] = 1
            else:
                mask_i = masks[i] > 0  
                out_mask[mask_i] = i + 2
    
        cv2.imwrite(f'{output_dir}/prompt_mask.png', out_mask.cpu().numpy())   





def create_datatset():
    model = YOLO(MODEL_PATH)
    videos = os.listdir(VIDEO_FOLDER)
    random.shuffle(videos)
    i = 1
    
    for video in videos:
        if i == MAX_VIDEO:
            break
        
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
        
        save_mask(model,os.path.join(full_path,'frame000.jpg'),mask_path + '/')
        
       

    


if __name__ == "__main__":
    create_datatset()