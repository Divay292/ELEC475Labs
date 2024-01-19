import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class PetNoseModel(nn.Module):
    def __init__(self):
        super(PetNoseModel, self).__init__()
        self.model = resnet34(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_feats = self.model.layer4[-1].bn2.num_features
        self.regressor = nn.Linear(num_feats, 2)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        coords = self.regressor(x)
        return coords