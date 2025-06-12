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
import re
import json
import matplotlib.pyplot as plt
import random

LABELS_PATH = './Dataset/out/'
VIDEOS_PATH = './Dataset/LC 5 sec clips 30fps/'


def _load_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_count

def _load_frame(cap, frame_idx, transform = None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        raise ValueError(f"Failed to read frame {frame_idx}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #return transform(Image.fromarray(frame))
    return Image.fromarray(frame)

def normalize(name: str) -> str:
        return re.sub(r'[^A-Za-z0-9]', '', name).lower()

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
    W, H = image.shape[:2]
    
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

def get_polygon(polygon):
    to_write = ''
    for vertex in polygon.keys():
        x = polygon[vertex]['x']
        y = polygon[vertex]['y']
        to_write += ' ' + str(x) + ' ' + str(y)
    return to_write

def create_train():
    
    videos = os.listdir(VIDEOS_PATH)
    labels = os.listdir(LABELS_PATH)
    y_train = []
    x_train = []

    count = 0
    for video in videos:
        print(f"Processing video {count}/{len(videos)}: {video}")
        cap, frame_count = _load_video(os.path.join(VIDEOS_PATH, video))

        base_video = os.path.splitext(video)[0]
        key_video  = normalize(base_video)


        matched_json = None
        for fname in os.listdir(LABELS_PATH):
            name_no_ext = os.path.splitext(fname)[0]
            if normalize(name_no_ext) == key_video:
                matched_json = os.path.join(LABELS_PATH, fname)
                break

        if not matched_json:
            print(f"[WARNING] No JSON file for «{video}». Skipped")
            count += 1
            continue

        with open(matched_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            frame_indices = list(data['labels'].keys())
            frame_indices = [int(idx) for idx in frame_indices if len(data['labels'][idx]) != 0]

        for idx in frame_indices:
            if int(idx) > 150:
                continue
            frame = _load_frame(cap, idx)
            W, H = frame.size
            
            len_dict = len(data['labels'][str(idx)])
            
            if len_dict < 2:
                continue
            
            tissue_mask = None
            tool_mask = None
            non_tool_mask = None
            
            if len_dict == 2:
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    if 'is_tti' in data['labels'][str(idx)][j].keys():
                            if data['labels'][str(idx)][j]['is_tti'] == 1:
                                tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                                tti_polygon_str = get_polygon(tti_polygon)
                                tissue_mask = parse_mask_string(tti_polygon_str,H,W)
                            else:
                                continue
                    else:
                        instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                        instrument_polygon_str = get_polygon(instrument_polygon)
                        tool_mask = parse_mask_string(instrument_polygon_str,H,W)
                        
                # frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                # show_mask_overlay(frame_cv, tool_mask)
                # show_mask_overlay(frame_cv, tissue_mask)

                if tool_mask is not None and tissue_mask is not None:
                    x_train.append(get_x_train(frame,tool_mask,tissue_mask))
                    y_train.append(1)
                # exit()
          
          
            if len_dict == 3:
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    if 'is_tti' in data['labels'][str(idx)][j].keys():
                        if data['labels'][str(idx)][j]['is_tti'] == 1:
                            tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                            tti_polygon_str = get_polygon(tti_polygon)
                            tissue_mask = parse_mask_string(tti_polygon_str,H,W)
                        else:
                            non_int_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                            non_int_polygon_str = get_polygon(non_int_polygon)
                            non_tool_mask = parse_mask_string(non_int_polygon_str,H,W)
                    else:
                        instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                        instrument_polygon_str = get_polygon(instrument_polygon)
                        tool_mask = parse_mask_string(instrument_polygon_str,H,W)
                if tool_mask is not None and tissue_mask is not None:
                    x_train.append(get_x_train(frame,tool_mask,tissue_mask))
                    y_train.append(1)   
                if non_tool_mask is not None and tissue_mask is not None:       
                    x_train.append(get_x_train(frame,non_tool_mask,tissue_mask))
                    y_train.append(0)          
                
            
            if len_dict == 4:
                pair_list = []
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    if 'is_tti' in data['labels'][str(idx)][j].keys():
                        if data['labels'][str(idx)][j]['is_tti'] == 1:
                            tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                            tti_polygon_str = get_polygon(tti_polygon)
                            tissue_mask = parse_mask_string(tti_polygon_str,H,W)
                            interaction_tool_name = data['labels'][str(idx)][j]['interaction_tool']
                            pair_list.append((tissue_mask, interaction_tool_name))
        
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    if 'is_tti' not in data['labels'][str(idx)][j].keys():
                        for pair in pair_list:
                            tool_name = pair[1]
                            tissue_mask = pair[0]
                            instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                            instrument_polygon_str = get_polygon(instrument_polygon)
                            tool_mask = parse_mask_string(instrument_polygon_str,H,W)
                            if tissue_mask is not None and tool_mask is not None:
                                x_train.append(get_x_train(frame,tissue_mask,tool_mask))
                                if tool_name == data['labels'][str(idx)][j]['instrument_type']:
                                    y_train.append(1)   
                                else:
                                    y_train.append(0)

        if count == 12:
            break  
                                                        
        count += 1
       
    
    return x_train , y_train

depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf") 

def get_x_train(image,tool_mask,tissue_mask):
    depth_map = np.array(depth_model(image)["depth"])
    image =  np.array(image)
    roi = extract_union_roi(image,tool_mask,tissue_mask,depth_map) 
    # print(roi.shape)
    # cv2.imshow("depth",roi[...,3:])
    roi_brg = cv2.cvtColor(roi[...,:3], cv2.COLOR_RGB2BGR)
    # cv2.imshow("image",roi_brg)
    roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_LINEAR) 
    
    roi_resized_brg = cv2.cvtColor(roi_resized[...,:3], cv2.COLOR_RGB2BGR)
    # cv2.imshow("image RESIZED",roi_resized_brg)
    # print(roi_resized.shape)
    # cv2.imshow("depth RESIZED",roi_resized[...,3:])
    # cv2.waitKey(0)
    roi_np = np.transpose(roi_resized, (2, 0, 1)).astype(np.float32) / 255.0
    # exit()
    return roi_np


def random_flip(img):
    """Specchiamento orizzontale."""
    return cv2.flip(img, 1)

def random_rotate(img, max_angle=45):
    """Rotazione casuale tra -45° e +45°."""
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)

def random_translate(img, max_shift=40):
    """Traslazione casuale orizzontale/verticale fino a 40 px."""
    tx = random.uniform(-max_shift, max_shift)
    ty = random.uniform(-max_shift, max_shift)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)

aug_funcs = [random_flip, random_rotate, random_translate]

if __name__ == '__main__':
    x_train, y_train = create_train()
    x_train = np.array(x_train)  
    y_train = np.array(y_train)

    zeros = np.sum(y_train == 0)
    ones  = np.sum(y_train == 1)
    print(f"Before augmentation → 0: {zeros}, 1: {ones}")

    if zeros < ones:
        minority_label = 0
        needed = ones - zeros
    elif ones < zeros:
        minority_label = 1
        needed = zeros - ones
    else:
        print("Already balanced")
        needed = 0

    minority_idx = np.where(y_train == minority_label)[0].tolist()
    random.seed(42)
    augmented_x, augmented_y = [], []
    selected_indices = []

    for _ in range(needed):
        idx = random.choice(minority_idx)
        selected_indices.append(idx)

        img_chw = x_train[idx]
        img_hwc = (np.transpose(img_chw, (1,2,0)) * 255).astype(np.uint8)

        fn = random.choice(aug_funcs)
        img_aug_hwc = fn(img_hwc)

        img_aug_chw = np.transpose(img_aug_hwc.astype(np.float32)/255.0, (2,0,1))
        augmented_x.append(img_aug_chw)
        augmented_y.append(minority_label)

    x_all = np.concatenate([x_train, np.stack(augmented_x)], axis=0) if needed>0 else x_train
    y_all = np.concatenate([y_train, np.array(augmented_y)], axis=0) if needed>0 else y_train
    perm = np.random.RandomState(42).permutation(len(y_all))
    x_train_aug = x_all[perm]
    y_train_aug = y_all[perm]

    zeros_new = np.sum(y_train_aug == 0)
    ones_new  = np.sum(y_train_aug == 1)
    print(f"After augmentation     → 0: {zeros_new}, 1: {ones_new}")


    '''To visualize some augumentations'''
    for start in range(0, needed, 4):
        end = min(start + 4, needed)
        chunk = selected_indices[start:end]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(6, 3*len(chunk)))
        if len(chunk) == 1:
            axes = np.expand_dims(axes, 0)

        for i, orig_idx in enumerate(chunk):
            orig_img = np.transpose(x_train[orig_idx], (1,2,0))
            aug_img  = np.transpose(augmented_x[start + i], (1,2,0))

            axes[i,0].imshow(orig_img)
            axes[i,0].set_title(f"Originale idx={orig_idx}")
            axes[i,0].axis('off')

            axes[i,1].imshow(aug_img)
            axes[i,1].set_title("Augmented")
            axes[i,1].axis('off')

        plt.tight_layout()
        plt.show()

    
    