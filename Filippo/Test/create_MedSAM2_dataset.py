import cv2
import os
import random

VIDEO_FOLDER = '../../Dataset/video_dataset/videos/train'
OUTPUT_FOLDER = './medSAM2_dataset/images/'


def _load_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_count



def create_datatset():
    
    videos = os.listdir(VIDEO_FOLDER)
    random.shuffle(videos)
    i = 1
    
    for video in videos:
        if i == 50:
            break
        
        full_path = os.path.join(OUTPUT_FOLDER,video)
        os.mkdir(full_path)
        
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
       

    


if __name__ == "__main__":
    create_datatset()