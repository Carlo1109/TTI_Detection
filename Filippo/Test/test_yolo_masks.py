from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import cv2
import random
import torch

MODEL_PATH = './runs/segment/train/weights/best.pt'
IMAGES_PATH = '../Full Dataset/val/'



if __name__ == '__main__':
    model = YOLO(MODEL_PATH)
    
    # imgs = os.listdir(IMAGES_PATH)
    # random.shuffle(imgs)
    imgs = ['./medSAM2_dataset/images/Adnanset-Lc 125-010/frame000.jpg']
    for img in imgs:
        img_path = os.path.join(IMAGES_PATH,img)
        img_path = img
        res = model.predict(img_path)
        ann = res[0].plot()                    
        ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)  
        
        for result in res:
            # get array results
            masks = result.masks.data
            boxes = result.boxes.data
            # extract classes
            clss = boxes[:, 5]
            out_mask = torch.zeros_like(masks[0], dtype=torch.uint8)

            # classe 0 → valore 1
            inds0 = torch.where(clss == 0)[0]
            if len(inds0) > 0:
                mask0 = torch.any(masks[inds0], dim=0)
                out_mask[mask0] = 1

            # classe 1 → valore 2
            inds1 = torch.where(clss == 1)[0]
            if len(inds1) > 0:
                mask1 = torch.any(masks[inds1], dim=0)
                out_mask[mask1] = 2

            # save to file
            cv2.imwrite('binary_mask.png', out_mask.cpu().numpy())          
        plt.imshow(ann_rgb)
        plt.axis('off')
        plt.show()