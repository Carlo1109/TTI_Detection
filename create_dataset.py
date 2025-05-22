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



def bbox_to_yolo(bbox, image_w, image_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / image_w
    y_center = (y_min + y_max) / 2 / image_h
    width = (x_max - x_min) / image_w
    height = (y_max - y_min) / image_h
    return [x_center, y_center, width, height]


def cv2Rect_to_yolo(rect, frame):
    (x_min, y_min) = (rect[0], rect[1])
    (w, h) = (rect[2], rect[3])
    # calculate maximum pixel
    (x_max, y_max) = (x_min + w, y_min + h)

    # calculate normalized center
    (x_c, y_c) = (((x_max + x_min)/2)/frame.shape[1],
                   ((y_max + y_min)/2)/frame.shape[0])

    # normalize width and height
    (w, h) = w/frame.shape[1], h/frame.shape[0]

    return (x_c, y_c, w, h)


def yolo_to_cv2Rect(rect, frame):
    (x_c, y_c, w, h) = rect
    w = w*frame.shape[1]
    h = h*frame.shape[0]
    x_min = (frame.shape[1]*x_c) - w/2
    y_min = (frame.shape[0]*y_c) - h/2

    return (round(x_min), round(y_min),
            round(w), round(h))


def polygon_to_yolo_seg(class_id, polygon, image_w, image_h):
    # Flatten and normalize the polygon coordinates
    normalized_coords = []
    for x, y in polygon:
        normalized_coords.append(str(x / image_w))
        normalized_coords.append(str(y / image_h))

    return f"{class_id} " + " ".join(normalized_coords)


def polygon_to_bbox(polygon):
    """
    polygon: list of [x, y] points
    returns: [x_min, y_min, x_max, y_max]
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max, y_max]


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


class TTIDatasetFromTxt(Dataset):
    image_prefix = "/images/"
    tti_prefix = "/tti_labels/"
    instrument_prefix = "/instrument_labels/"

    def __init__(self, image_dir, label_dir, split='train', transform=None):
        self.image_dir = os.path.join(image_dir, split)
        self.label_dir = os.path.join(label_dir, split)
        self.transform =  transform or T.Compose([ T.Resize((640, 640)),
                                                    T.ToTensor(),
                                                ])
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])
        self.labels = self._build_labels()
        
    def _build_labels(self):
        labels = []
        for idx in range(len(self.image_files)):
            sample = self.__getitem__(idx)
            # Build the dictionary as expected by Ultralytics
            labels.append({
                "im_file": os.path.join(self.image_dir, self.image_files[idx]),
                "bboxes": torch.tensor(sample["bboxes"], dtype=torch.float32),  # shape [N, 4]
                "cls": sample["tools"].unsqueeze(-1),  # shape [N, 1]
                "segments": [torch.tensor(p, dtype=torch.float32) for p in sample["polygons"]],  # list of [N, 2]
                # add other keys as needed
            })
        return labels
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        instrument_label_path = image_path.replace(self.image_prefix, self.instrument_prefix).replace('.png', '.txt').replace('.png', '.txt')
        tti_label_path = instrument_label_path.replace(self.instrument_prefix, self.tti_prefix)

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        w, h = image.shape[2], image.shape[1]

        # Load labels
        polygons = []
        class_ids = []
        interactions = []
        is_tti = []
        bboxes = []

        if os.path.exists(tti_label_path):
            with open(tti_label_path, 'r') as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    interaction = int(values[0])
                    class_id = int(values[1])
                    coords = values[2:]
                    if len(coords) % 2 != 0:
                        continue  # skip malformed
                    # Reshape coords to list of [x, y] and scale to pixel coords
                    polygon = [[coords[i] * w, coords[i+1] * h] for i in range(0, len(coords), 2)]
                    polygons.append(polygon)
                    class_ids.append(class_id)
                    interactions.append(interaction)
                    is_tti.append(1)
                    bboxes.append( polygon_to_bbox(polygon))

        if os.path.exists(instrument_label_path):
            with open(instrument_label_path, 'r') as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    interaction = int(values[0])
                    class_id = int(values[1])
                    coords = values[2:]
                    if len(coords) % 2 != 0:
                        continue
                    polygon = [[coords[i] * w, coords[i+1] * h] for i in range(0, len(coords), 2)]
                    polygons.append(polygon)
                    class_ids.append(class_id)
                    interactions.append(interaction)
                    is_tti.append(0)
                    bboxes.append( polygon_to_bbox(polygon))

        return {
            'img': image,                              # Tensor[C, H, W]
            'polygons': polygons,                        # List[List[[x, y], ...]]
            'tools': torch.tensor(class_ids),        # Tensor[N]
            'ttis': torch.tensor(interactions),   # Tensor[N]
            'is_tti': torch.tensor(is_tti),
            "bboxes" : bboxes,
            'batch_idx': torch.zeros(len(class_ids), dtype=torch.int64)
        }



class TTIDatasetFromJSON(Dataset):
    """
    Type-A: dict_keys(['instrument_polygon', 'instrument_type'])
    Type-B: dict_keys(['is_tti', 'interaction_type', 'interaction_tool', 'tti_polygon'])
    Type-C: dict_keys(['is_tti', 'non_interaction_tool', 'tti_polygon'])
    """

    def __init__(self, video_dir, label_dir, split='train', transform=None):
        self.video_dir = os.path.join(video_dir, split)
        self.label_dir = os.path.join(label_dir, split)
        self.transform = transform or T.ToTensor()
        self.data = self.prepare_data(self.label_dir)

    def prepare_data(self, label_dir):
        data = dict()
        for file in os.listdir(label_dir):
            video = os.path.basename(file).replace(".json", ".mp4")
            json_file = json.load(open(os.path.join(label_dir, file)))["labels"]
            for frame_idx, anns in json_file.items():
                IS_TTI, POLYGON, TTI, TOOL, BBOX = [],[],[],[],[]
                i = 0
                for ann in anns:
                    if 'is_tti' in ann.keys():
                        is_tti = ann['is_tti']
                        polygon = ann['tti_polygon']
                        tti = to_tti_id(ann['interaction_type'] if is_tti else 'unknown')
                        tool = to_tool_id(ann['interaction_tool'] if is_tti else ann['non_inteeraction_tool'])
                    else:
                        is_tti = 0
                        tti = 0
                        polygon = ann['instrument_polygon']
                        tool = to_tool_id(ann['instrument_type'])
                    bbox = polygon_to_bbox(polygon)
                    IS_TTI.append(is_tti)
                    POLYGON.append(list(polygon.values()))
                    TTI.append(tti)
                    TOOL.append(tool)
                data[i] = dict(
                    video = os.path.join(self.video_dir, video),
                    frame_id = frame_idx,
                    is_tti = torch.tensor(IS_TTI),
                    tti = torch.tensor(TTI),
                    polygon = torch.tensor(POLYGON),
                    bbox = torch.tensor(BBOX),
                    tool = torch.tensor(TOOL)
                )
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        # Load image
        video, n = _load_video(record['video'])
        image = _load_frame(video, record['frame_id'], self.transform)
        w, h = image.shape[2], image.shape[1]
        polygons = record['polygon'] * torch.tensor([[w, h]])
        bboxes = record['bbox'] * torch.tensor([[w, h, w, h]])

        return {
            'image': image,                              # Tensor[C, H, W]
            'polygons': polygons,                        # List[List[[x, y], ...]]
            'bboxes': bboxes,
            'tools': torch.tensor(record['tool']),        # Tensor[N]
            'ttis': torch.tensor(record['tti']),   # Tensor[N]
            'is_tti': torch.tensor(record['is_tti'])

        }



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
