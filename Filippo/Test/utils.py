import cv2
import numpy as np
import re
from PIL import Image, ImageDraw



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



def get_polygon(polygon):
    to_write = ''
    for vertex in polygon.keys():
        x = polygon[vertex]['x']
        y = polygon[vertex]['y']
        to_write += ' ' + str(x) + ' ' + str(y)
    return to_write





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


def points_to_mask(polys, H, W):
    """Converte poligono/i (Nx2) in maschera binaria 0/255"""
    if polys is None or len(polys) == 0:
        return np.zeros((H, W), np.uint8)

    # se è un singolo poligono (lista di punti) → lo incapsulo in lista
    if isinstance(polys[0][0], (int, float)):
        polys = [polys]

    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    for p in polys:
        if len(p) < 3: 
            continue
        p = np.array(p)
        cx, cy = p[:,0].mean(), p[:,1].mean()
        angles = np.arctan2(p[:,1]-cy, p[:,0]-cx)
        ordered = p[np.argsort(angles)]
        pts = [tuple(map(int, pt)) for pt in ordered]
        draw.polygon(pts, fill=1)

    return np.array(img, np.uint8) * 255