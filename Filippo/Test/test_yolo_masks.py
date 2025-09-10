from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import cv2
import random
import torch

MODEL_PATH_ORIGINAL = './runs_fine_tuned/segment/train/weights/best.pt'
MODEL_PATH = './runs/segment/train/weights/best.pt'
IMAGES_PATH = '../Full Dataset/val/'



if __name__ == '__main__':
    model = YOLO(MODEL_PATH)
    model_orig = YOLO(MODEL_PATH_ORIGINAL)
    imgs = os.listdir(IMAGES_PATH)
    random.shuffle(imgs)

    for img in imgs:
        img_path = os.path.join(IMAGES_PATH,img)
        res = model.predict(img_path)
        ann = res[0].plot()                    
        ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB) 
        
         
        res_orig = model_orig.predict(img_path)
        ann_orig = res_orig[0].plot()                    
        ann_rgb_orig = cv2.cvtColor(ann_orig, cv2.COLOR_BGR2RGB)  
        
        
        
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(ann_rgb)
        plt.title("Model")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(ann_rgb_orig)
        plt.title("Model Orig")
        plt.axis('off')

        plt.tight_layout()
        plt.show()