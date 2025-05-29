from ultralytics import YOLO


from transformers import pipeline
from PIL import Image 

if __name__ == "__main__":
   

    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    image = Image.open('./Dataset/dataset/images/test/video0008_frame0035.png')
    depth = pipe(image)["depth"]
    depth.show()
    # Image.show(depth)
  
    # model = YOLO('./runs/segment/train/weights/best.pt')
    
    # res = model.predict('./Dataset/dataset/images/test/video0014_frame0011.png')
    # model.predict('./Dataset/LC 5 sec clips 30fps/Adnanset-Lc 10-003.mp4',show=True)
    # print(len(res[0]))
    # for r in res:
    #     pass
        # print(len(r.masks.data))
        # print(r.boxes.cls)
        # print(r.masks.data)     