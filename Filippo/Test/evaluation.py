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
CHECKPOINT  = "student_model_best.pt"

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



def expand_mask(mask, pixels):
    if pixels <= 0:
        return mask.astype(bool).copy()
    ks = 2 * int(pixels) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)

def vis(model,im,depth_model,score_thresh=0.3, mask_thresh=0.3):
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

    
    H,W,_ = img.shape
    scale_x = W / IMG_SIZE[0]
    scale_y = H / IMG_SIZE[1]
    
    tool_list = []
    tissue_list = []
    
    for mask, label, score, box in zip(preds["masks"], preds["labels"], preds["scores"], preds["boxes"]):
        score = score.cpu().numpy()   
        label = label.cpu().numpy()
        box = box.cpu().numpy()   
        if score < score_thresh:
            continue
        
        mask_bin = mask.squeeze(0).cpu().numpy() >= mask_thresh
        mask_bin = cv2.resize(mask_bin.astype(np.uint8),(W,H),cv2.INTER_NEAREST)
        
        x1, y1, x2, y2 = box
        x1 *= scale_x ; x2 *= scale_x ;y1 *= scale_y ; y2 *= scale_y
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                            fill=False, color='green', linewidth=1.5)
        
        if label == 1:
            tissue_list.append((mask_bin,rect,score))
        else:
            tool_list.append(((mask_bin,rect,score)))
           
    pairs = []
    for tool in tool_list:
        print("tool found")
        tool_mask = tool[0]
        tool_bbox = tool[1]
        for tissue in tissue_list:
            tissue_mask = tissue[0]
            tissue_bbox = tissue[1]
            intersection_mask = np.logical_and(expand_mask(tool_mask,2),tissue_mask)
            inter_numb = intersection_mask.sum()
            # plt.imshow(tissue_mask)
            # plt.show()
            if inter_numb > 0:
                depth_map = np.array(depth_model(im)["depth"]) / 255
                
                tool_int = np.logical_and(intersection_mask,tool_mask)
                tissue_int = np.logical_and(intersection_mask,tissue_mask)
                # plt.imshow(tool_int)
                # plt.show()
                # plt.imshow(tissue_int)
                # plt.show()
                
                depth_tool_int = depth_map[tool_int]
                depth_tissue_int = depth_map[tissue_int]
                
                #depth median
                med_tool = np.median(depth_tool_int)
                med_tissue = np.median(depth_tissue_int)
   
                print(med_tool)
                print(med_tissue)
                
                
                
                tolerance = 0.03
                
                if abs(med_tool - med_tissue) <= tolerance:
                    pairs.append((tool_mask,tissue_mask))
                
        
        
    plt.imshow(img)
    plt.show()
    for pair in pairs:  
        plt.imshow(img)
        to = np.ma.masked_where(pair[0] == 0, pair[0])  
        ti = np.ma.masked_where(pair[1] == 0, pair[1])  
        plt.imshow(to,alpha=0.5, cmap='jet')
        plt.imshow(ti,alpha=0.5,cmap='Greens')
        
        # plt.gca().add_patch(rect)

        plt.title(f" TTI ")
        plt.axis("off")
        plt.show()
        
    


if __name__ == '__main__':
    model = get_maskrcnn()
    model.to(DEVICE)
    img_fp = "../Full Dataset/val/video0273_frame120.jpg"
    sd = torch.load(CHECKPOINT, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    depth = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    vis(model,img_fp,depth)