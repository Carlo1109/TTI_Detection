import os
import re
import json
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import pipeline
from train_NN import train_model
from Model import CNN_TCN_Classifier
from pipeline import show_mask_overlay_from_binary_mask


LABELS_PATH = './Dataset/out/'
VIDEOS_PATH = './Dataset/LC 5 sec clips 30fps/'

depth_model = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

def _load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_count


def _load_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        raise ValueError(f"Failed to read frame {frame_idx}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def normalize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', name).lower()


def parse_mask_string(mask_str, H, W):
    if mask_str is None or not mask_str.strip():
        return np.zeros((0, 2), dtype=np.int32)
    tokens = mask_str.strip().split()
    n_vals = len(tokens) // 2 * 2
    coords = []
    for i in range(0, n_vals, 2):
        try:
            x_norm = float(tokens[i])
            y_norm = float(tokens[i+1])
        except ValueError:
            continue
        x_pix = int(x_norm * W)
        y_pix = int(y_norm * H)
        x_pix = max(0, min(W-1, x_pix))
        y_pix = max(0, min(H-1, y_pix))
        coords.append((x_pix, y_pix))
    return np.array(coords, dtype=np.int32) if coords else np.zeros((0,2), dtype=np.int32)


def extract_union_roi(image, tool_mask, tissue_mask, depth_map=None):
    H, W = image.shape[:2]
    bin_tool = np.zeros((H, W), np.uint8)
    if tool_mask.size:
        cv2.fillPoly(bin_tool, [tool_mask], 1)
    bin_tissue = np.zeros((H, W), np.uint8)
    if tissue_mask.size:
        cv2.fillPoly(bin_tissue, [tissue_mask], 1)
    combined = np.clip(bin_tool + bin_tissue, 0, 1).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(combined)
    roi = image[y:y+h, x:x+w]
    if depth_map is not None:
        depth_roi = depth_map[y:y+h, x:x+w]
        roi = np.concatenate([roi, depth_roi[..., None]], axis=-1)
    mask_crop = (bin_tool[y:y+h, x:x+w] | bin_tissue[y:y+h, x:x+w])[..., None]
    roi = np.concatenate([roi, mask_crop*255], axis=-1)
    return roi


def get_polygon(polygon):
    to_write = ''
    for vertex in polygon.keys():
        x = polygon[vertex]['x']
        y = polygon[vertex]['y']
        to_write += f" {x} {y}"
    return to_write


def get_roi_tensor(frame: Image.Image, tool_mask, tissue_mask):
    depth_map = np.array(depth_model(frame)["depth"])
    image = np.array(frame)
    roi = extract_union_roi(image, tool_mask, tissue_mask, depth_map)
    roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_LINEAR)
    roi_chw = np.transpose(roi_resized, (2,0,1)).astype(np.float32) / 255.0
    return roi_chw

def create_temporal_train(window_radius=3):
    x_seqs = []
    y_labels = []
    for video in os.listdir(VIDEOS_PATH):
        cap, frame_count = _load_video(os.path.join(VIDEOS_PATH, video))
        base = os.path.splitext(video)[0]
        key = normalize(base)
        matched_json = None
        for fname in os.listdir(LABELS_PATH):
            if normalize(os.path.splitext(fname)[0]) == key:
                matched_json = os.path.join(LABELS_PATH, fname)
                break
        if not matched_json:
            continue
        with open(matched_json, 'r', encoding='utf-8') as f:
            data = json.load(f)['labels']
        frame_indices = [int(idx) for idx in data.keys() if data[idx]]
        for idx in frame_indices:
            if idx - window_radius < 0 or idx + window_radius >= frame_count:
                continue
            objects = data[str(idx)]
            if len(objects) < 2:
                continue
            tissue_mask = None
            tool_mask = None
            non_tool_mask = None
            if len(objects) == 2:
                for obj in objects:
                    if not obj:
                        continue
                    if 'is_tti' in obj and obj['is_tti'] == 1:
                        tissue_mask = parse_mask_string(get_polygon(obj['tti_polygon']), frame_count, frame_count)
                    else:
                        tool_mask = parse_mask_string(get_polygon(obj['instrument_polygon']), frame_count, frame_count)
                if tissue_mask is not None and tool_mask is not None:
                    label = 1
                else:
                    continue
            elif len(objects) == 3:
                for obj in objects:
                    if not obj:
                        continue
                    if 'is_tti' in obj and obj['is_tti'] == 1:
                        tissue_mask = parse_mask_string(get_polygon(obj['tti_polygon']), frame_count, frame_count)
                    elif 'is_tti' in obj and obj['is_tti'] == 0:
                        non_tool_mask = parse_mask_string(get_polygon(obj['instrument_polygon']), frame_count, frame_count)
                    else:
                        tool_mask = parse_mask_string(get_polygon(obj['instrument_polygon']), frame_count, frame_count)
                if tissue_mask is None or tool_mask is None:
                    continue
                if tool_mask is not None:
                    label = 1
                if non_tool_mask is not None:
                    seq = []
                    for t in range(idx - window_radius, idx + window_radius + 1):
                        frame = _load_frame(cap, t)
                        seq.append(get_roi_tensor(frame, non_tool_mask, tissue_mask))
                    x_seqs.append(np.stack(seq))
                    y_labels.append(0)
                    continue
            else:  
                pair_list = []
                for obj in objects:
                    if not obj:
                        continue
                    if 'is_tti' in obj and obj['is_tti'] == 1:
                        tissue_mask = parse_mask_string(get_polygon(obj['tti_polygon']), frame_count, frame_count)
                        interaction_tool = obj.get('interaction_tool')
                        pair_list.append((tissue_mask, interaction_tool))
                for obj in objects:
                    if not obj or 'is_tti' in obj:
                        continue
                    tool_mask = parse_mask_string(get_polygon(obj['instrument_polygon']), frame_count, frame_count)
                    for tissue, tool_name in pair_list:
                        if tissue is None or tool_mask is None:
                            continue
                        label = 1 if tool_name == obj.get('instrument_type') else 0
                        seq = []
                        for t in range(idx - window_radius, idx + window_radius + 1):
                            frame = _load_frame(cap, t)
                            seq.append(get_roi_tensor(frame, tool_mask, tissue))
                        x_seqs.append(np.stack(seq))
                        y_labels.append(label)
                continue
            seq = []
            for t in range(idx - window_radius, idx + window_radius + 1):
                frame = _load_frame(cap, t)
                seq.append(get_roi_tensor(frame, tool_mask, tissue_mask))
            x_seqs.append(np.stack(seq))
            y_labels.append(label)
    return np.array(x_seqs, dtype=np.float32), np.array(y_labels, dtype=np.int64)

if __name__ == '__main__':
    x_train, y_train = create_temporal_train(window_radius=3)
    print(f"Dataset shapes → X: {x_train.shape}, y: {y_train.shape}")

    model = CNN_TCN_Classifier(
        tcn_channels=[256,128], sequence_length=7, num_classes=1
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train_model(
    #     model=model,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    #     x_train=x_train,
    #     y_train=y_train,
    #     epochs=40
    # )
