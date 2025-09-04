import os, json, cv2, numpy as np, matplotlib.pyplot as plt, random


IMGS_PATH = './medSAM2_dataset/images/'
MASKS_PATH = './medSAM2_dataset/binary_mask_msam_for_val/'
# OUT_IMAGE = './step_1_seg_train/images/train/'
# OUT_LABELS = './step_1_seg_train/labels/train/'
OUT_IMAGE = './step_1_seg_train/test_img/'
OUT_LABELS = './step_1_seg_train/test_label/'



def save_label(mask, base, out_folder):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    H, W = mask.shape[:2]
    lines = []
    
    for val in np.unique(mask):
        if val == 0:
            continue
        bin_mask = (mask == val).astype('uint8') * 255
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        cnt = max(contours, key=cv2.contourArea)
        eps = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
        pts = pts.astype(float)
        pts[:, 0] /= W
        pts[:, 1] /= H
        cls = 1 if int(val) == 1 else 0
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        
        lines.append(f"{cls} {coords}")
            
    path = os.path.join(out_folder, base + '.txt')
    if lines:
        with open(path, 'w') as f:
            f.write("\n".join(lines))
    else:
        with open(path, 'w') as f:
            f.write("0")

            
            



if __name__ == '__main__':
    masks = os.listdir(MASKS_PATH)
    i = 0
    for video_name in masks:
        print(f'processsing video {i}/{len(masks)}')
        mask_path = os.path.join(MASKS_PATH,video_name)
        imgs_path = os.path.join(IMGS_PATH,video_name)
        
        frames = os.listdir(imgs_path)
        
        for frame in frames:
            msk = os.path.join(mask_path,frame).replace('.jpg','.png')
            img = os.path.join(imgs_path,frame)
            
            img = cv2.imread(img)
            
            mask = cv2.imread(msk)
         
            cv2.imwrite(f'{OUT_IMAGE}{video_name}_{frame}',img)
            
            save_label(mask, f"{video_name}_{frame.replace('.jpg','')}", OUT_LABELS)
            
        i+=1
            
            



    