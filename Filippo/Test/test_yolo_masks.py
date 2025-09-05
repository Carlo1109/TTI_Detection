from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import cv2
import random
import torch

MODEL_PATH = './runs_compl_dataset/segment/train/weights/best.pt'
IMAGES_PATH = '../Full Dataset/val/'



if __name__ == '__main__':
    model = YOLO(MODEL_PATH)
    
    imgs = os.listdir(IMAGES_PATH)
    random.shuffle(imgs)

    for img in imgs:
        img_path = os.path.join(IMAGES_PATH,img)
        res = model.predict(img_path)
        ann = res[0].plot()                    
        ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)  
        
        plt.imshow(ann_rgb)
        plt.axis('off')
        plt.show()