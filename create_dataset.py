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


# Dataset organization:
# TODO
"""
/dataset/
  /images/
    /train/
      videoXXX_frame_000001.png
      videoXXX_frame_000002.png
      ...
    /val/
  /tti_labels/
    /train/
      videoXXX_frame_000001.txt
      videoXXX_frame_000002.txt
      ...
    /val/
  /instrument_labels/
    /train/
      videoXXX_frame_000001.txt
      videoXXX_frame_000002.txt
      ...
    /val/
"""
#1. Images extracted from videos and labelled by frame_id and arranged as above:
#1.1. Images are named by video and frame idx.
#1.2. They are organized into train/val/test

#2. labels for TTI extracted from the json to .txt file and organized as above:
#2.1. Each file represent a single frame.
#2.2. Each file contains one or multiple line labels for one or multiple TTI in that frame.
#2.3 The format for each label is:
#   <interaction> <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
#2.4. The <interaction> label is a binary of 0 or 1 for if there is a TTI or not.

#3. labels for Instrument extracted from the json to .txt file and organized as above:
#3.1. Each file represent a single frame.
#3.2. Each file contains one or multiple line labels for one or multiple Instrument in that frame.
#3.3 The format for each label is:
#   <interaction> <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
#3.4. The <interaction> label is a binary of 0 or 1 for if the instrument is involved in TTI or not.


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
    if name == None:
      return 0
    name_to_id = {
        "unknown_tool": 0,
        "harmonic": 1,
        "grasper": 2,
        "grasper 2": 3,
        "stapler": 4,
        "dissector": 5,
        "ligasure": 6,
        "cautery": 7,
        "grasper 3": 8,
        "bipolar": 9,
        "suction": 10,
        "scissors": 11,
        "cautery (hook, spatula)": 12,
    }
    name = name.lower()
    return name_to_id[name]


def to_tti_id(name):
    if name == None:
      return 0
    name_to_id = {
        "unknown_tti": 13,
        "other": 14,
        "dissector": 15,
        "cautery (hook, spatula)": 16,
        "cut - sharp dissection": 17,
        "retract and grab": 18,
        "coagulation": 19,
        "blunt dissection": 20,
        "retract and push": 21,
        "staple": 22,
        "energy - sharp dissection": 23
    }
    name = name.lower()
    return name_to_id[name]



"""CODE TO CREATE THE DATASET"""

def create_dataset():
    file_path   = 'Dataset'
    videos_path = os.path.join(file_path, 'LC 5 sec clips 30fps')
    json_folder = os.path.join(file_path, 'out')

    def normalize(name: str) -> str:
        return re.sub(r'[^A-Za-z0-9]', '', name).lower()

    i = 1
    videos = os.listdir(videos_path)

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
            print(f"[WARNING] No JSON file forr «{video}». Skipped")
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
            out_img = os.path.join(file_path, 'dataset', 'images', 'train',f'video{i:04d}_frame{idx:04d}.png')
            frame.save(out_img)
            print(f"Saved {out_img}")

            
            if 'is_tti' in data['labels'][str(idx)][0].keys():
                is_tti = data['labels'][str(idx)][0]['is_tti']

                if int(is_tti) == 1:
                    interaction_type = to_tti_id(data['labels'][str(idx)][0]['interaction_type'])
                else:
                    interaction_type = '0'

                polygon = data['labels'][str(idx)][0]['tti_polygon']

                to_write = ''
                to_write += str(is_tti) + ' ' + str(interaction_type)
                
                with open(file_path+'/dataset/tti_labels/train/'+ f'video{i:04d}_frame{idx:04d}.txt', 'w', encoding='utf-8') as f:
                    for vertex in polygon.keys():
                        x = data['labels'][str(idx)][0]['tti_polygon'][vertex]['x']
                        y = data['labels'][str(idx)][0]['tti_polygon'][vertex]['y']
                        to_write += ' ' + str(x) + ' '+ str(y)
                    f.write(to_write)
            else:
                entries = data['labels'][str(idx)]

                for e in entries:
                    if 'instrument_type' in e:
                        inst_entry = e
                        break
                    else:
                        continue

                tool_type = to_tool_id(inst_entry['instrument_type'])
                instrument_polygon = inst_entry['instrument_polygon']

                to_write = ''
                to_write += str(0) + ' ' + str(tool_type)

                with open(file_path+'/dataset/instrument_labels/train/'+ f'video{i:04d}_frame{idx:04d}.txt', 'w', encoding='utf-8') as f:
                    for vertex in instrument_polygon.keys():
                        x = inst_entry['instrument_polygon'][vertex]['x']
                        y = inst_entry['instrument_polygon'][vertex]['y']
                        to_write += ' ' + str(x) + ' '+ str(y)
                    f.write(to_write)

        i += 1


def move_file(fname, split):
    base_dir    = './Dataset/dataset'
    # image
    src_img = os.path.join(base_dir, 'images', 'train', fname)
    dst_img = os.path.join(base_dir, 'images', split, fname)
    shutil.move(src_img, dst_img)

    # instrument label
    lbl = fname.replace('.png', '.txt')
    src_lbl = os.path.join(base_dir, 'instrument_labels', 'train', lbl)
    dst_lbl = os.path.join(base_dir, 'instrument_labels', split, lbl)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

    # tti label
    src_lbl2 = os.path.join(base_dir, 'tti_labels', 'train', lbl)
    dst_lbl2 = os.path.join(base_dir, 'tti_labels', split, lbl)
    if os.path.exists(src_lbl2):
        shutil.move(src_lbl2, dst_lbl2)

def split_dataset():

    base_dir    = './Dataset/dataset'
    folders     = ['images', 'instrument_labels', 'tti_labels']
    splits      = ['train', 'val', 'test']
    train_split = 0.8
    val_split   = 0.1
    # test_split (0.1)


    for folder in folders:
        for split in splits:
            path = os.path.join(base_dir, folder, split)
            os.makedirs(path, exist_ok=True)



    img_train_dir = os.path.join(base_dir, 'images', 'train')
    all_imgs = [f for f in os.listdir(img_train_dir) if f.lower().endswith('.png')]

    # shuffle per random
    random.seed(42)
    random.shuffle(all_imgs)

    n = len(all_imgs)
    print("The whole dataset is composed by " + str(n) + " images")
    n_train = int(n * train_split)
    n_val   = int(n * val_split)
    n_test = n - n_train - n_val

    train_imgs = all_imgs[:n_train]
    val_imgs   = all_imgs[n_train:n_train + n_val]
    test_imgs  = all_imgs[n_train + n_val:]


    for fname in train_imgs:
        move_file(fname, 'train')
    for fname in val_imgs:
        move_file(fname, 'val')
    for fname in test_imgs:
        move_file(fname, 'test')

    print(f"Final distribution: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    
    
    
if __name__ == "__main__":
    # create_dataset()
    split_dataset()
