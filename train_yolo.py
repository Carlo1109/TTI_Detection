from ultralytics.models.yolo.segment import SegmentationTrainer
from create_dataset import TTIDatasetFromTxt
from ultralytics import YOLO
from torch.utils.data import DataLoader

IMAGE_DIR = './Dataset/dataset/images'
LABEL_DIR = './Dataset/dataset/instrument_labels'


if __name__== "__main__":
    
    model = YOLO("yolo11n-seg.pt")

    
    #results = model.train(trainer=CustomSegmentationTrainer, data="dataset.yaml", epochs=3)        
    results = model.train(data="dataset.yaml", epochs=3)
    # trainer = CustomSegmentationTrainer(overrides=args)
    # trainer.train()

