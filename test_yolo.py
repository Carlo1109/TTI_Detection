from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('./runs/segment/train/weights/best.pt')
    
    # model.predict('./Dataset/dataset/images/test',save=True)
    model.predict('./Dataset/LC 5 sec clips 30fps/Adnanset-Lc 100-002.mp4',show=True)