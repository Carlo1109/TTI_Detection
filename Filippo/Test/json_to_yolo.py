import os, json, cv2, numpy as np, matplotlib.pyplot as plt, random
from pycocotools import mask as mask_utils

imgs_path = './step_1_seg_train/images/train/'
masks_path = './step_1_seg_train/masks/'
OUT_FOLDER = './step_1_seg_train/labels/train/'

def decode_rle(rle):
    if isinstance(rle.get("counts"), str): rle = dict(rle); rle["counts"] = rle["counts"].encode("utf-8")
    return mask_utils.decode(rle).astype(np.uint8)

img_files = [f for f in os.listdir(imgs_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
random.shuffle(img_files)   

k = 1
for fn in img_files:
    if 'video' not in fn:
        continue
    print(f'processing {k}/{len(img_files)}')
    base = os.path.splitext(fn)[0]
    jpth = os.path.join(masks_path, base + ".json")
    ipth = os.path.join(imgs_path, fn)
    
    img = cv2.cvtColor(cv2.imread(ipth), cv2.COLOR_BGR2RGB); H,W = img.shape[:2]

    with open(jpth) as f: 
        anns = json.load(f)

    yolo_lines = []  # accumulo righe per questo file .txt

    for ann in anns:
        # ottieni rle (compatibilità con più strutture)
        rle = ann.get("mask",ann).get("rle") if isinstance(ann.get("mask",None), dict) else ann.get("rle", ann)
        if rle is None: continue
        try:
            m = decode_rle(rle)
        except Exception as e:
            print("rle decode fail", base, e); continue
            
        if m.shape != (H,W):
            m = cv2.resize(m.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST)
            
        m = (m>0).astype(np.uint8)
        if m.sum()==0: continue

        
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = max(contours, key=cv2.contourArea)

        # semplifico il poligono (facoltativo, evita punti eccessivi)
        eps = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2)

        if len(pts) < 3: 
            continue  

        # classe: cerca category_id o class se presenti, altrimenti 0
        cls = ann['labels']
       

        # normalizzo i punti (x/W, y/H) e formato stringa YOLO seg
        norm_pts = pts.astype(float)
        norm_pts[:,0] /= W
        norm_pts[:,1] /= H

        # flatten e formatta con 6 decimali
        pairs = [f"{x:.6f} {y:.6f}" for x,y in norm_pts]
        line = str(cls-1) + " " + " ".join(pairs)
        yolo_lines.append(line)

    # scrivo il file .txt (una riga per oggetto). Se non ci sono oggetti, lo rimuovo se esiste.
    label_file = os.path.join(OUT_FOLDER, base + ".txt")
    if yolo_lines:
        with open(label_file, 'w') as f:
            f.write("\n".join(yolo_lines))
    else:
        if os.path.exists(label_file):
            os.remove(label_file)
    k+=1
    