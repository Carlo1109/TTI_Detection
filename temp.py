import torch
import cv2
import torchvision.models as models
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
from Model import ROIClassifier ,AutoEncoder
import matplotlib.pyplot as plt
import time
import os


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    auto_enc = AutoEncoder()
    auto_enc.load_state_dict(torch.load('Auto_enc.pt',map_location=device))
    auto_enc.to(device)
    
    enc = auto_enc.encoder
    enc.eval()
    
    depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    IMAGES = './Dataset/yolo_dataset/images/train/'
    OUT = './Dataset/encoder_output/images/train/'
    
    images = os.listdir(IMAGES)
    c=1
    for image in images:
        print(f"Processing {c}/{len(images)}/{image}")
        c+=1
        
        depth_map = np.array(depth_model(IMAGES + image)["depth"])

        
        img = cv2.imread(IMAGES + image,cv2.IMREAD_COLOR)


        img = np.concatenate([img, depth_map[..., None]], axis=-1)
        img = torch.tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = enc(img)
      
        pred = pred.squeeze(0).detach().cpu()  # (C, H, W)

        
        pred = pred.permute(1, 2, 0).numpy()
        pred = (pred * 255).astype(np.uint8)

        cv2.imwrite(OUT+image,pred)
    
    
    
    