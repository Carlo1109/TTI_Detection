import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from tqdm import tqdm
from pipeline import end_to_end_pipeline
from transformers import pipeline as transformers_pipeline
from models.VIT import ROIClassifierViT
from draw_pipe_output import INSTRUMENT_NAME_TO_ID, TTI_NAME_TO_ID


VIDEO_INPUT_PATH = "./video_dataset/videos/test/"
VIDEO_OUTPUT_PATH = "./video_output/"
YOLO_WEIGHTS = "./runs/segment/train/weights/best.pt"
VIT_WEIGHTS = "./models/ViT2.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def draw_detections_on_frame(frame, detections, tti_predictions):
    """
    Draws masks and TTI bounding boxes on a frame
    Returns: frame with masks and frame with boxes
    """
    image = frame.copy()
    
    # Image with masks
    img_with_masks = image.astype(np.float32).copy()
    alpha = 0.5
    
    for i, det in enumerate(detections):
        mask = det['mask']
        cls = det['class']
        # Resize mask to match image dimensions
        mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Color coding: red for tools, cyan for tissues
        if cls in range(0, 7):
            color = np.array([200, 0, 0]).astype(np.float32)
        else:
            color = np.array([0, 220, 220]).astype(np.float32)
        
        # Blend only on masked pixels
        img_with_masks[mask_resized] = (1 - alpha) * img_with_masks[mask_resized] + alpha * color

    img_with_masks = img_with_masks.astype(np.uint8)
    
    # Draw bounding boxes for TTI pairs
    img_with_boxes = img_with_masks.copy()
    for tti in tti_predictions:
        if tti['tti_class'] == 0:
            continue  # Skip non-TTI pairs
            
        tool_det = tti['tool']
        tissue_det = tti['tissue']
        
        # Get masks for tool and tissue
        tool_mask = tool_det['mask']
        tissue_mask = tissue_det['mask']
        
        # Resize masks to match image dimensions
        tool_mask_resized = cv2.resize(tool_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        tissue_mask_resized = cv2.resize(tissue_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Combine masks with logical OR
        combined_mask = np.logical_or(tool_mask_resized, tissue_mask_resized).astype(np.uint8)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(combined_mask)
        
        # Draw rectangle
        color = (0, 220, 220)  # Cyan in BGR
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
        
        # Get tool class name
        tool_class = tool_det['class']
        tool_name = None
        for name, class_id in INSTRUMENT_NAME_TO_ID.items():
            if class_id == tool_class:
                tool_name = name
                break
        if tool_name is None:
            tool_name = f"Tool {tool_class}"
        
        # Get tissue class name
        tissue_class = tissue_det['class']
        tissue_name = None
        for name, class_id in TTI_NAME_TO_ID.items():
            if class_id == tissue_class:
                tissue_name = name
                break
        if tissue_name is None:
            tissue_name = f"Class {tissue_class}"
        
        # Combine tool and tissue names
        class_name = f"{tool_name} - {tissue_name}"
        
        # Draw text above the bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_color = (0, 0, 0)  # Black text
        text_thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, text_thickness)
        
        # Get image dimensions
        img_height, img_width = img_with_boxes.shape[:2]
        
        # Check if text fits above the bounding box
        text_y_top = y - text_height - baseline - 5
        text_y_bottom = y
        
        if text_y_top < 0:
            # Text goes out of bounds on top, put it below the box
            text_y_top = y + h + 5
            text_y_bottom = y + h + text_height + baseline + 10
        
        # Calculate initial text position
        text_x = x + 2
        
        # Check if text goes out of bounds on the right
        if text_x + text_width + 5 > img_width:
            text_x = img_width - text_width - 7
            text_x = max(0, text_x)
        
        # Draw background rectangle for text
        bg_color = (0, 220, 220)  # Cyan background
        cv2.rectangle(img_with_boxes, 
                     (text_x, text_y_top),
                     (min(text_x + text_width + 5, img_width), text_y_bottom),
                     bg_color, -1)
        
        # Draw text
        text_y = text_y_top + text_height + baseline - 2
        cv2.putText(img_with_boxes, class_name, (text_x, text_y), font, font_scale, text_color, text_thickness)
    
    return img_with_masks, img_with_boxes


def process_video(video_path, yolo_model, depth_model, tti_classifier, output_path):
    """
    Process a video and save output with masks, TTI boxes and labels
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 120:  # Fallback for invalid FPS
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    frame_count = 0
    
    # Process each frame with progress bar
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get detections and TTI predictions
            detections, tti_predictions = end_to_end_pipeline(frame_rgb, yolo_model, depth_model, tti_classifier, DEVICE, depth_map_required=True)
            
            # Skip frame if processing failed
            if detections is None or tti_predictions is None:
                print(f"Warning: Failed to process frame {frame_count}")
                # Write original frame instead
                out_video.write(frame)
                frame_count += 1
                pbar.update(1)
                continue
            
            # Draw on frame
            img_with_masks, img_with_boxes = draw_detections_on_frame(frame_rgb, detections, tti_predictions)
            
            # Convert back to BGR for video writing
            img_with_boxes_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
            
            # Write frame to output video
            out_video.write(img_with_boxes_bgr)
            
            frame_count += 1
            pbar.update(1)
    
    # Release everything
    cap.release()
    out_video.release()
    
    print(f"Processed {frame_count} frames")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(VIDEO_OUTPUT_PATH, exist_ok=True)
    
    # Load models
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    print("YOLO model loaded successfully!")
    
    print("Loading depth model...")
    depth_model = transformers_pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    print("Depth model loaded successfully!")
    
    print("Loading ViT classifier...")
    tti_classifier = ROIClassifierViT(num_hoi_classes=2)
    tti_classifier.load_state_dict(torch.load(VIT_WEIGHTS, map_location=DEVICE))
    tti_classifier.to(DEVICE)
    tti_classifier.eval()
    print("ViT classifier loaded successfully!")
    
    # Process all videos in input directory
    video_files = [f for f in os.listdir(VIDEO_INPUT_PATH) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {VIDEO_INPUT_PATH}")
    
    for video_name in video_files:
        video_path = os.path.join(VIDEO_INPUT_PATH, video_name)
        
        # Create output path
        base_name = os.path.splitext(video_name)[0]
        output_path = os.path.join(VIDEO_OUTPUT_PATH, f"{base_name}_output.mp4")
        
        # Process video
        process_video(video_path, yolo_model, depth_model, tti_classifier, output_path)
        print("-" * 80)
    
    print("All videos processed!")
