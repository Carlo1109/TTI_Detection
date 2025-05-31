import torchvision.models as models
import torch.nn as nn
import torch

class ROIClassifier(nn.Module):
    def __init__(self, num_hoi_classes):
        super().__init__()
      
        self.pre_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.fc = nn.Linear(512, num_hoi_classes)

        with torch.no_grad():
            w_orig = self.backbone.conv1.weight  
           
            w_preconv = torch.zeros((3, 4, 1, 1), dtype=w_orig.dtype)
            
            w_mean_rgb = w_orig.mean(dim=[2,3])  
            
            for i in range(3):
                w_preconv[i, i, 0, 0] = w_mean_rgb[i, i]  

            self.pre_conv.weight.copy_(w_preconv)

    def forward(self, x):
        
        x = self.pre_conv(x)         
        features = self.backbone(x)   
        out = self.fc(features)
        return out
