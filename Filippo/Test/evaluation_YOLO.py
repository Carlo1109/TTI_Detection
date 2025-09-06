import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import os
import random
from ultralytics import YOLO
from utils import _load_video, _load_frame, normalize , get_polygon ,  parse_mask_string ,to_tool_id , to_tti_id , points_to_mask
import json
from sklearn.metrics import accuracy_score

IMG_SIZE    = (256, 256)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT  = './runs_sampled_dataset/segment/train/weights/best.pt'

LABELS_PATH = '../../Dataset/video_dataset/labels/val/'
VIDEOS_PATH = '../../Dataset/video_dataset/videos/val/'

LABELS_TXT_PATH = '../../Dataset/evaluation/labels/'
BINARY_MASKS_PATH = './Evaluation_dataset/labels/'
IMAGES_FOR_EVAL_PATH ='./Evaluation_dataset/images/'



def create_test():
    
    videos = os.listdir(VIDEOS_PATH)

    
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
            to_write = ''
            masks = []
            if int(idx) > 150:
                continue
            frame = _load_frame(cap, idx)
            
            W, H = frame.size
            
            len_dict = len(data['labels'][str(idx)])
            
            if len_dict < 2:
                continue

            frame.save('./Evaluation_dataset/images/'+ f'video{count:04d}_frame{idx:04d}.jpg')
            
            tissue_mask = None
            tool_mask = None
            non_tool_mask = None
            tool_class = 0
            tti_class = 0
            non_tool_class = 0
            
            if len_dict == 2:
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    
                    if 'is_tti' in data['labels'][str(idx)][j].keys():
                            if data['labels'][str(idx)][j]['is_tti'] == 1:
                                tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                                tti_class = to_tti_id(data['labels'][str(idx)][j]['interaction_type'])
       
                                tti_polygon_str = get_polygon(tti_polygon)
                                tissue_mask = parse_mask_string(tti_polygon_str,H,W)                   
                                
                            else:
                                continue
                    else:
                        instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                        instrument_polygon_str = get_polygon(instrument_polygon)
                        tool_mask = parse_mask_string(instrument_polygon_str,H,W)
                        tool_class = to_tool_id(data['labels'][str(idx)][j]['instrument_type'])
                       
                        

                if tool_mask is not None and tissue_mask is not None:
                    to_write += '1' + ' ' + str(tool_mask) + ' ' + str(tissue_mask)  + ' ' + '\n'
                    masks.append(tool_mask)
                    masks.append(tissue_mask)
                    # y_train.append(1)
          
          
            if len_dict == 3:
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    if 'is_tti' in data['labels'][str(idx)][j].keys():
                        if data['labels'][str(idx)][j]['is_tti'] == 1:
                            tti_class = to_tti_id(data['labels'][str(idx)][j]['interaction_type']) 
                            tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                            tti_polygon_str = get_polygon(tti_polygon)
                            tissue_mask = parse_mask_string(tti_polygon_str,H,W)
                        else:
                            non_int_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                            non_int_polygon_str = get_polygon(non_int_polygon)
                            non_tool_mask = parse_mask_string(non_int_polygon_str,H,W)
                            non_tool_class = to_tool_id(data['labels'][str(idx)][j]['non_interaction_tool'])
                    else:
                        instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                        instrument_polygon_str = get_polygon(instrument_polygon)
                        tool_mask = parse_mask_string(instrument_polygon_str,H,W)
                        tool_class = to_tool_id(data['labels'][str(idx)][j]['instrument_type'])

                if tool_mask is not None and tissue_mask is not None:
                    to_write += '1' + ' ' + str(tool_mask) + ' ' + str(tissue_mask)  + ' ' + '\n'
                    masks.append(tool_mask)
                    masks.append(tissue_mask)
                    # y_train.append(1)   
                # if non_tool_mask is not None and tissue_mask is not None:       
                #     to_write += '0' + ' ' + str(non_tool_class) + ' ' + str(tti_class)  + ' ' + '\n'
                #     # y_train.append(0)          
                
            
            if len_dict == 4:
                pair_list = []
                for j in range(len_dict):
                    if not bool(data['labels'][str(idx)][j].keys()):
                        continue
                    if 'is_tti' in data['labels'][str(idx)][j].keys():
                        if data['labels'][str(idx)][j]['is_tti'] == 1:
                            tti_class = to_tti_id(data['labels'][str(idx)][j]['interaction_type'])
                            tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                            tti_polygon_str = get_polygon(tti_polygon)
                            tissue_mask = parse_mask_string(tti_polygon_str,H,W)
                            interaction_tool_name = data['labels'][str(idx)][j]['interaction_tool']
                            pair_list.append((tissue_mask, interaction_tool_name , tti_class))
        
                for pair in pair_list:
                    for j in range(len_dict):
                        if not bool(data['labels'][str(idx)][j].keys()):
                            continue
                        if 'is_tti' not in data['labels'][str(idx)][j].keys():
                            tool_name = pair[1]
                            tissue_mask = pair[0]
                            tti_class = pair[2]
                            instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                            instrument_polygon_str = get_polygon(instrument_polygon)
                            tool_mask = parse_mask_string(instrument_polygon_str,H,W)
                            tool_class = to_tool_id(tool_name)
                            
                            if tissue_mask is not None and tool_mask is not None:
                                if tool_name == data['labels'][str(idx)][j]['instrument_type']:
                                    # y_train.append(1)   
                                    to_write += '1' + ' ' + str(tool_mask) + ' ' + str(tissue_mask)  + ' ' + '\n'
                                    masks.append(tool_mask)
                                    masks.append(tissue_mask)
                                # else:
                                    # y_train.append(0) 
                                    # to_write += '0' + ' ' + str(to_tool_id(data['labels'][str(idx)][j]['instrument_type'])) + ' ' + str(tti_class)  + ' ' + '\n'
              
            # with open('./Evaluation_dataset/labels/'+ f'video{count:04d}_frame{idx:04d}.txt', 'w', encoding='utf-8') as f:
            #     f.write(to_write) 
            msk = points_to_mask(masks,H,W)
            cv2.imwrite('./Evaluation_dataset/labels/'+ f'video{count:04d}_frame{idx:04d}.png',msk)
                                                        
        count += 1



def expand_mask(mask, pixels):
    kernel_size = 2 * pixels + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return expanded


def vis(model,im,depth_model):

    preds = model.predict(im,verbose=False)
    
    ann = preds[0].plot() 
    ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB) 
    plt.imshow(ann_rgb)
    plt.show()
    
    img = cv2.imread(im)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    tool_list = []
    tissue_list = []
    
   
    for result in preds:
        # get array results
        if result.masks is None:
            return
        masks = result.masks.data
        boxes = result.boxes.data
        # extract classes
        clss = boxes[:, 5]
        
        for i in range(masks.shape[0]):
            mask_resized = cv2.resize(masks[i].cpu().numpy().astype('uint8'), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            # print(mask_resized.shape)
            if clss[i] == 1:
                tool_list.append(mask_resized)
            else:
                tissue_list.append(mask_resized)
        
           
    pairs = []
    for tool in tool_list:
        # print("tool found")
        tool_mask = tool
        depth_map = np.array(depth_model(im)["depth"]) / 255
        tool_depth = depth_map[tool_mask.astype(bool)]
        
        med_depth = np.percentile(tool_depth,15)
        
        t = np.logical_and(tool_mask , depth_map < med_depth)
        
        tool_mask = t


        for tissue in tissue_list:
            tissue_mask = tissue
            # plt.imshow(tissue_mask)
            # plt.show()
            # plt.imshow(expand_mask(tissue_mask,10))
            # plt.show()
            
            intersection_mask = np.logical_and(expand_mask(tool_mask,10),expand_mask(tissue_mask,10))
            inter_numb = intersection_mask.sum()
            
            plt.imshow(img)
            plt.imshow(np.ma.masked_where(expand_mask(tool_mask,10) == 0,expand_mask(tool_mask,10)),alpha=0.7)
            plt.imshow(np.ma.masked_where(expand_mask(tissue_mask,10) == 0,expand_mask(tissue_mask,10)),alpha=0.7)
            plt.show()
            
            if inter_numb > 5:
                # plt.imshow(img)
                # plt.imshow(np.ma.masked_where(intersection_mask == 0, intersection_mask)  ,alpha=0.8)
                # plt.title("PRE EXPAND")
                # plt.show()
                
                intersection_mask = expand_mask(intersection_mask,20)
                
                plt.imshow(img)
                plt.imshow(np.ma.masked_where(intersection_mask == 0, intersection_mask),alpha=0.8)
                plt.show()
                
                plt.imshow(depth_map)
                plt.show()
                
                tool_int = np.logical_and(intersection_mask.astype(bool),tool_mask.astype(bool))
                tissue_int = np.logical_and(intersection_mask.astype(bool),tissue_mask.astype(bool))
                
                if not np.any(tool_int) or not np.any(tissue_int):
                    continue
  
                plt.imshow(tool_int)
                plt.show()
                plt.imshow(tissue_int)
                plt.show()
                
                depth_tool_int = depth_map[tool_int.astype(bool)]
                depth_tissue_int = depth_map[tissue_int.astype(bool)]
                
                #depth median
                med_tool = np.mean(depth_tool_int)
                med_tissue = np.mean(depth_tissue_int)
                
                # print(med_tool)
                # print(med_tissue)
                
                if np.isnan(med_tool) or np.isnan(med_tissue):
                    continue

                
                tolerance = 0.1
    
                
                if np.abs(med_tool - med_tissue) <= tolerance:
                    pairs.append((tool_mask,tissue_mask))
                    
                    
    if len(pairs) == 0:
        print("NO TTI FOUND")
    for pair in pairs:  
        plt.imshow(img)
        to = np.ma.masked_where(pair[0] == 0, pair[0])  
        ti = np.ma.masked_where(pair[1] == 0, pair[1])  
        plt.imshow(to,alpha=0.5, cmap='jet')
        plt.imshow(ti,alpha=0.5,cmap='Greens')
        plt.title(" TTI ")
        plt.axis("off")
        plt.show()
    return pairs
                
        
        
        
    
    
def count_ttis(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        ttis = 0
        for line in f:
            if not line:
                continue
            line = line.strip()
            is_tti = int(line[0])
            if is_tti == 1:
                ttis += 1
                
    if ttis > 2:
        print("More than 2 ttis")
        return 2
    return ttis
        
            
    
    
def evaluate():
    model = YOLO(CHECKPOINT) 
    depth = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-small-hf")
    
    images = os.listdir(IMAGES_FOR_EVAL_PATH)
    random.shuffle(images)
    
    y_pred = []
    y_true= []
    i = 0
    
    n_no_pred = 0
    n_wrong_masks = 0
    n_diff_length = 0
    tot_tti = 0
    n_good_predictions = 0
    
    for img_fp in images:
        print(f'processing {i}/{len(images)}')
        image_path = os.path.join(IMAGES_FOR_EVAL_PATH,img_fp)
        binary_mask_path = os.path.join(BINARY_MASKS_PATH,img_fp.replace('.jpg','.png'))
        labels_path = os.path.join(LABELS_TXT_PATH,img_fp.replace('.jpg','.txt'))
        
        msk = cv2.imread(binary_mask_path,cv2.IMREAD_GRAYSCALE)
        
        n_tti = count_ttis(labels_path)
        pairs = vis(model,image_path,depth)
        tot_tti += n_tti
        
        if n_tti > 0 and pairs is None:
            y_true.append(1)
            y_pred.append(0)
            n_no_pred += 1
            continue
        elif n_tti == 0 and pairs is None:
            y_true.append(1)
            y_pred.append(1)
            n_good_predictions+=1
            continue
        
        # if n_tti != len(pairs):
        #     y_true.append(1)
        #     y_pred.append(0)
        #     n_diff_length += 1
        
        for tool_mask , tissue_mask in pairs:
            tool_int = np.logical_and(tool_mask,msk).sum()
            mask_int = np.logical_and(tissue_mask,msk).sum()
            
            if tool_int > 0 or mask_int > 0:
                y_true.append(1)
                y_pred.append(1)
                n_good_predictions+=1
            else:
                y_true.append(1)
                y_pred.append(0)
                n_wrong_masks += 1
        i+=1
        
    print(f'Number different length: {n_diff_length}')
    print(f'Number no preds: {n_no_pred}')
    print(f'Number Wrong masks: {n_wrong_masks}')
    print(f'Total TTIS: {tot_tti}')
    print(f'Total good predictions: {n_good_predictions}')
    
    return y_pred,y_true
            


if __name__ == '__main__':
    # create_test()
    y_pred,y_true = evaluate()
    
    zeros = 0
    ones = 0
    for i in y_pred:
        if i == 0:
            zeros += 1
        else:
            ones += 1
    
    print("Ones: ", ones)
    print("Zeros: ", zeros)       
    print("Accuracy: ", accuracy_score(y_true, y_pred))