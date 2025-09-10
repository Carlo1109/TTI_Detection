import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Dataset
import cv2
import json
from torchvision import transforms
from PIL import Image
import re
import random
import shutil
import torchvision.transforms as T




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


def to_tool_id(name):
    if name == None :
        return 0
    name_to_id = {
        'unknown_tool': 0,
        'dissector': 1,
        'scissors': 2,
        'suction': 3,
        'grasper 3': 4,
        'harmonic': 5,
        'grasper': 6,
        'bipolar': 7,
        'grasper 2': 8,
        'cautery (hook, spatula)': 9,
        'ligasure': 10,
        'stapler': 11,        
    }
    name = name.lower()
    if name not in name_to_id.keys():
      return 0
    return name_to_id[name]


def to_tti_id(name):
    if name == None:
      return 12
    name_to_id = {
        'unknown_tti': 12,
        'coagulation': 13,
        'other': 14,
        'retract and grab': 15,
        'blunt dissection': 16,
        'energy - sharp dissection': 17,
        'staple': 18,
        'retract and push': 19,
        'cut - sharp dissection': 20,
    }
    name = name.lower()
    if name not in name_to_id.keys():
        return 12
    
    return name_to_id[name]



"""CODE TO CREATE THE DATASET"""

def normalize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', name).lower()


def create_dataset():
    file_path   = './fine_tuning_dataset'
    videos_path = os.path.join( '../../Dataset/video_dataset/videos/val/')
    json_folder = os.path.join('../../Dataset/video_dataset/labels/val/')


    i = 1
    videos = os.listdir(videos_path)
    random.shuffle(videos)

    n_images = 0
    
    for video in videos:
        print(f"Processing video {i}/{len(videos)}: {video}")

        cap, frame_count = _load_video(os.path.join(videos_path, video))


        base_video = os.path.splitext(video)[0]
        key_video  = normalize(base_video)

        matched_json = None
        for fname in os.listdir(json_folder):
            name_no_ext = os.path.splitext(fname)[0]
            if normalize(name_no_ext) == key_video:
                matched_json = os.path.join(json_folder, fname)
                break

        if not matched_json:
            print(f"[WARNING] No JSON file for «{video}». Skipped")
            i += 1
            continue

        
        with open(matched_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            frame_indices = list(data['labels'].keys())
            frame_indices = [int(idx) for idx in frame_indices if len(data['labels'][idx]) != 0]

        for idx in frame_indices:
            if int(idx) > 150:
                continue
            frame = _load_frame(cap, idx)
            out_img = os.path.join(file_path, 'images', 'val',f'video{i:04d}_frame{idx:04d}.jpg')
            frame.save(out_img)
            print(f"Saved {out_img}") 
            


            to_write = ''
            # print(len(data['labels'][str(idx)]))
            
            for j in range(len(data['labels'][str(idx)])):
                current_dict = data['labels'][str(idx)][j]
                
                
                dict_len = len(current_dict.keys())
                if dict_len == 4:
                    
                    is_tti = data['labels'][str(idx)][j]['is_tti']
                    interaction_type = to_tti_id(data['labels'][str(idx)][j]['interaction_type'])
                    to_write += str(0) 
                    tti_polygon = data['labels'][str(idx)][j]['tti_polygon']
                    
                    for vertex in tti_polygon.keys():
                        x = data['labels'][str(idx)][j]['tti_polygon'][vertex]['x']
                        y = data['labels'][str(idx)][j]['tti_polygon'][vertex]['y']
                        to_write += ' ' + str(x) + ' ' + str(y)
                        
                    to_write += '\n'
                    
                elif dict_len == 3:
                    
                    is_tti = data['labels'][str(idx)][j]['is_tti']
                    non_interaction_tool = to_tool_id(data['labels'][str(idx)][j]['non_interaction_tool'])
                    to_write += str(1)
                    instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                    
                    for vertex in instrument_polygon.keys():
                        x = instrument_polygon[vertex]['x']
                        y = instrument_polygon[vertex]['y']
                        to_write += ' ' + str(x) + ' '+ str(y)
                        
                    to_write += '\n'
                    
                elif dict_len == 2:
                    instr_type = to_tool_id(data['labels'][str(idx)][j]['instrument_type'])
                    
                    to_write += str(1) 
                    instrument_polygon = data['labels'][str(idx)][j]['instrument_polygon']
                    
                    for vertex in instrument_polygon.keys():
                        x = instrument_polygon[vertex]['x']
                        y = instrument_polygon[vertex]['y']
                        to_write += ' ' + str(x) + ' '+ str(y)
                        
                    to_write += '\n'
                    
                
            with open(file_path+'/labels/val/'+ f'video{i:04d}_frame{idx:04d}.txt', 'w', encoding='utf-8') as f:
                f.write(to_write) 
              
            if n_images > 100:
                exit()  
            n_images += 1  

        i += 1


    
if __name__ == "__main__":
    create_dataset()
    

    
