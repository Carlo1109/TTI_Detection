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

def vis(model,im,score_thresh=0.4, mask_thresh=0.4):
    img = cv2.imread(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    tf_img = T.Compose([
            T.ToPILImage(),
            T.Resize(IMG_SIZE,  interpolation=Image.BILINEAR),
            T.ToTensor(),  # [0,1] e C×H×W
        ])
    inp = tf_img(img).to(DEVICE)
    model.eval()
    with torch.no_grad():
        preds = model([inp])[0]
    # print(preds)
    # masks  = preds["masks"].squeeze(1).cpu().numpy()      # (N, H, W)
    # labels = preds["labels"].cpu().numpy().astype(int)    # (N,)
    # scores = preds["scores"].cpu().numpy()                # (N,)
    
    H,W,_ = img.shape
    scale_x = W / IMG_SIZE[0]
    scale_y = H / IMG_SIZE[1]
    for mask, label, score, box in zip(preds["masks"], preds["labels"], preds["scores"], preds["boxes"]):
        score = score.cpu().numpy()   
        label = label.cpu().numpy()
        box = box.cpu().numpy()   
        if score < score_thresh:
            continue
        
        mask_bin = mask.squeeze(0).cpu().numpy() >= mask_thresh
        mask_bin = cv2.resize(mask_bin.astype(np.uint8),(W,H))
        mask_bin = np.ma.masked_where(mask_bin == 0, mask_bin)        
        plt.imshow(img)
        plt.imshow(mask_bin,alpha=0.5, cmap='jet')
        x1, y1, x2, y2 = box
        x1 *= scale_x ; x2 *= scale_x ;y1 *= scale_y ; y2 *= scale_y
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                            fill=False, color='green', linewidth=1.5)
        plt.gca().add_patch(rect)

        plt.title(f"Class {label.item()} | Score {score:.2f}")
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    model = get_maskrcnn()
    model.to(DEVICE)
    img_fp = "../../Dataset/evaluation/images/video0000_frame0051.png"
    sd = torch.load(CHECKPOINT, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    vis(model,img_fp)