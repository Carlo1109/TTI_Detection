import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch

class ROIClassifier(nn.Module):
    def __init__(self, num_hoi_classes):
        super().__init__()
      
        self.first = nn.Conv2d(5, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.pre_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.fc = nn.Linear(512, num_hoi_classes)
        
    def forward(self, x):
        
        x = self.first(x)
        x = self.pre_conv(x)         
        features = self.backbone(x)   
        out = F.sigmoid(self.fc(features))
        return out
