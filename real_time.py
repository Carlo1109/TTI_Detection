import torch
import cv2
from transformers import pipeline
from Model import ROIClassifier
from pipeline import *
from PIL import  Image
import time


def real(video,yolo_model,depth,tti_classifier,device):
    vidcap = cv2.VideoCapture(video)
    
    success,image = vidcap.read()
    H_full, W_full = image.shape[:2]
    while success:
        
        detection , tti_predictions  = end_to_end_pipeline(image,yolo_model,depth,tti_classifier,device)
        combined_tool_mask = np.zeros((H_full, W_full), dtype=np.uint8)
        combined_tissue_mask = np.zeros((H_full, W_full), dtype=np.uint8)
        for elem in tti_predictions:
            tool_mask = elem['tool']['mask']
            tissue_mask = elem['tissue']['mask']
            is_tti = elem['tti_class']
            
            tool_class = elem['tool']['class']
            tissue_class = elem['tissue']['class']

            tool_mask = tool_mask.astype(np.uint8)
            
            tool_mask_resized = cv2.resize(
                tool_mask.astype(np.uint8),
                (W_full, H_full),
                interpolation=cv2.INTER_NEAREST
            )
            tissue_mask_resized = cv2.resize(
                tissue_mask.astype(np.uint8),
                (W_full, H_full),
                interpolation=cv2.INTER_NEAREST
            )
 

            combined_tool_mask = np.maximum(combined_tool_mask, tool_mask_resized)
            combined_tissue_mask = np.maximum(combined_tissue_mask, tissue_mask_resized)


            overlay_image_tool = show_mask_overlay_from_binary_mask(image, combined_tool_mask, alpha=0.5, mask_color=(1.0, 0.0, 0.0))
            overlay_image_tissue = show_mask_overlay_from_binary_mask(image, combined_tissue_mask, alpha=0.5, mask_color=(0.0, 1.0, 0.0))
            overlay_mask = cv2.bitwise_or(overlay_image_tool,overlay_image_tissue)
            
            
        cv2.imshow("video", overlay_mask )
        cv2.waitKey(1)
      
        success,image = vidcap.read()
        if image is None:
            break
  
        


    vidcap.release()
    cv2.destroyAllWindows()
        





if __name__ == "__main__":
    model = load_yolo_model('./runs/segment/train/weights/best.pt')
    video = './Dataset/Video/test/Adnanset-Lc 11-010.mp4'
    
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tti_class = ROIClassifier(2)
    tti_class.load_state_dict(torch.load('ROImodel.pt',map_location=device))
    tti_class.to(device)
    
    real(video,model,pipe,tti_class,device)
    
    
    
    