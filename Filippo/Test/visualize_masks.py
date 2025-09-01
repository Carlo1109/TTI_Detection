import os, json, cv2, numpy as np, matplotlib.pyplot as plt, random
from pycocotools import mask as mask_utils

imgs_path = './step_1_seg_train/images/'
masks_path = './step_1_seg_train/masks/'

def decode_rle(rle):
    if isinstance(rle.get("counts"), str): rle = dict(rle); rle["counts"] = rle["counts"].encode("utf-8")
    return mask_utils.decode(rle).astype(np.uint8)

img_files = [f for f in os.listdir(imgs_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
random.shuffle(img_files)   

for fn in img_files:
    base = os.path.splitext(fn)[0]
    jpth = os.path.join(masks_path, base + ".json")
    ipth = os.path.join(imgs_path, fn)
    if not os.path.exists(jpth): print("no json for", fn); continue
    img = cv2.cvtColor(cv2.imread(ipth), cv2.COLOR_BGR2RGB); H,W = img.shape[:2]

    with open(jpth) as f: anns = json.load(f)
    plt.figure(figsize=(8,6)); plt.imshow(img); plt.axis('off')

    for ann in anns:
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
        plt.imshow(np.ma.masked_where(m==0, m), cmap='jet', alpha=0.4)

        ys,xs = np.where(m)
        x1,y1,x2,y2 = xs.min(), ys.min(), xs.max(), ys.max()
        plt.gca().add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='white', linewidth=1))

    plt.title(fn); plt.show()
