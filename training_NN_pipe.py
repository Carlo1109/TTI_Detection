import cv2
import numpy as np
import torch.nn as nn
from transformers import pipeline
from PIL import Image
from train_NN import train_model
from Model import ROIClassifier
import torch
from transformers import pipeline
import os


LABELS_PATH = './Dataset/dataset/labels/train/'
IMAGES_PATH = './Dataset/dataset/images/train/'


def parse_mask_string(mask_str, H, W):
    
    if mask_str is None or mask_str.strip() == "":
        return np.zeros((0, 2), dtype=np.int32)

    tokens = mask_str.strip().split()
    if len(tokens) < 2:
        return np.zeros((0, 2), dtype=np.int32)

    n_vals = len(tokens) // 2 * 2
    coords = []
    for i in range(0, n_vals, 2):
        try:
            x_norm = float(tokens[i])
            y_norm = float(tokens[i + 1])
        except ValueError:
            continue

        x_pix = int(x_norm * W)
        y_pix = int(y_norm * H)

        x_pix = max(0, min(W - 1, x_pix))
        y_pix = max(0, min(H - 1, y_pix))

        coords.append((x_pix, y_pix))

    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    return np.array(coords, dtype=np.int32)



def extract_union_roi(image, tool_mask, tissue_mask, depth_map=None):
    H, W = image.shape[:2]
    tool_mask = parse_mask_string(tool_mask,H,W)
    tissue_mask = parse_mask_string(tissue_mask,H,W)
    
    polygons = [tool_mask]
    tool_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(tool_mask, polygons, color=1)
    
    polygons = [tissue_mask]
    tissue_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(tissue_mask, polygons, color=1)

    
    combined_mask = (tool_mask + tissue_mask).clip(0, 1).astype('uint8') #before astype
    x, y, w, h = cv2.boundingRect(combined_mask)

    roi = image[y:y+h, x:x+w]

    if depth_map is not None:
        depth_roi = depth_map[y:y+h, x:x+w]
        roi = np.concatenate([roi, depth_roi[..., None]], axis=-1)  # add depth as extra channel

    return roi


def create_train():
    
    images = os.listdir(IMAGES_PATH)
    labels = os.listdir(LABELS_PATH)
    X_train = []
    y_train = []
    
    count = 0
    for img in images:
        print(f"PROCESSING IMAGE {count}/{len(images)}")
        count+=1
        image_path = os.path.join(IMAGES_PATH,img)
        label_name = img.replace('png','txt')
        label_path = os.path.join(LABELS_PATH,label_name)
        tool_classes = list(range(0, 12))
        
        
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        if len(lines) < 2 or len(lines) == 4:
            continue
        
        l = []
            
        
        for line in lines:
            is_tti = line[0]
            cls = line[2:4]
            d = dict()
            
            if is_tti == '1':
                d['is_tti'] = 1
                if int(cls) in tool_classes:
                    d['class'] = 'tool'
                    tool_mask = line[4:]
                    d['mask'] = tool_mask
                else:
                    d['class'] = 'tissue'
                    tti_mask = line[4:]
                    d['mask'] = tti_mask
            else:
                d['is_tti'] = 0
                d['class'] = 'tool'
                non_tool_mask = line[4:]
                d['mask'] = non_tool_mask
                
            l.append(d)
        
        
        if len(l) == 2:
            for dictionary in l:
                if dictionary['class'] == 'tool':
                    tool_mask = dictionary['mask']
                elif dictionary['class'] == 'tissue':
                    tissue_mask = dictionary['mask']
            X_train.append(get_x_train(image_path,tool_mask,tissue_mask))
            y_train.append(1)
        
        temp_dict = {}
        temp_dict['1'] = []
        
        if len(l) == 3:
            for dictionary in l:
                if dictionary['is_tti'] == 1:
                    temp_dict['1'].append(dictionary)
                else:
                    temp_dict['0'] = dictionary
            
        
        for key in temp_dict.keys():
            if key == 1:
                for elem in temp_dict[key]:
                    if elem['class'] == 'tool':
                        tool_mask = elem['mask']
                    else:
                        tissue_mask = elem['mask']
                X_train.append(get_x_train(image_path,tool_mask,tissue_mask))
                y_train.append(1)

            if key == 0:
                for elem in temp_dict['1']:
                    if elem['class'] == 'tissue':
                        tissue_mask = elem['mask']
                        
                tool_mask = temp_dict['0']['mask']
                X_train.append(get_x_train(image_path,tool_mask,tissue_mask))
                y_train.append(0)
    
    return X_train , y_train

depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf") 

def get_x_train(image_path,tool_mask,tissue_mask):
     
    image_roi = cv2.imread(image_path,cv2.IMREAD_COLOR)
    depth_map = np.array(depth_model(image_path)["depth"])
    roi = extract_union_roi(image_roi,tool_mask,tissue_mask,depth_map)  
    roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_LINEAR)
    roi_np = np.transpose(roi_resized, (2, 0, 1)).astype(np.float32) / 255.0
    
    return roi_np

if __name__ == '__main__':
    # create_train()
    x_train , y_train = create_train()
    model = ROIClassifier(2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_model(model,optimizer,loss_fn,x_train,y_train,40)
    