from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('./runs/segment/train/weights/best.pt')
    
    res = model.predict('./Dataset/dataset/images/test/video0014_frame0011.png')
    # model.predict('./Dataset/LC 5 sec clips 30fps/Adnanset-Lc 10-003.mp4',show=True)
    print(len(res[0]))
    for r in res:
        pass
        # print(len(r.masks.data))
        # print(r.boxes.cls)
        # print(r.masks.data)     