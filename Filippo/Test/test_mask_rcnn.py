import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


IMG_SIZE    = (256, 256)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT  = "teacher_cycle_0.pt"

def get_maskrcnn(num_classes=3):
 
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    

    if num_classes != -1:

        # Get number of input features for the classifier.
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained box predictor head with a new one.
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Now get the number of input features for the mask classifier.
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        hidden_layer = 256
        # Replace the pre-trained mask predictor head with a new one.
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

    return model

def vis(model,im,score_thresh=0.5, mask_thresh=0.5):
    img = cv2.imread(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    tf_img = T.Compose([
            T.ToPILImage(),
            T.Resize(IMG_SIZE,  interpolation=Image.BILINEAR),
            T.ToTensor(),  # [0,1] e C×H×W
        ])
    inp = tf_img(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        preds = model(inp)[0]
    # print(preds)
    # masks  = preds["masks"].squeeze(1).cpu().numpy()      # (N, H, W)
    # labels = preds["labels"].cpu().numpy().astype(int)    # (N,)
    # scores = preds["scores"].cpu().numpy()                # (N,)
    
    for mask, label, score in zip(preds["masks"], preds["labels"], preds["scores"]):
        score = score.cpu().numpy()   
        label = label.cpu().numpy()   
        if score < score_thresh:
            continue
        mask_bin = mask.squeeze(0).cpu().numpy() >= mask_thresh
        plt.imshow(mask_bin)
        plt.title(f"Class {label.item()} | Score {score:.2f}")
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    model = get_maskrcnn()
    model.to(DEVICE)
    img_fp = "../../Dataset/evaluation/images/video0000_frame0051.png"
    vis(model,img_fp)