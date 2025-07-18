import cv2
import os


VIDEO_FOLDER = '../../Dataset/video_dataset/videos/train/'
OUTPUT_FOLDER = '../Full Dataset/train/'


def _load_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_count



def create_datatset():
    
    videos = os.listdir(VIDEO_FOLDER)
    i = 1
    
    for video in videos:
        print(f'Processing Video {i}/{len(videos)}')
        
        cap, frame_count = _load_video(os.path.join(VIDEO_FOLDER, video))
        
        count = 0
        success = True
        while success:
            success,image = cap.read()

            if count%8 == 0 and count < frame_count :
                print(f"Frame {count}/{frame_count} ----> {video}")
                cv2.imwrite(OUTPUT_FOLDER + f'video{i:04d}_frame%d.jpg'%count,image)
            count+=1
        i+=1
       

    


if __name__ == "__main__":
    create_datatset()