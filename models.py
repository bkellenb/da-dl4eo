'''
    Standard ResNet-based image classification models.

    2020 Benjamin Kellenberger
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet



class ClassificationModel(nn.Module):
    in_channels = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }

    def __init__(self, num_classes, backbone='resnet18', pretrained=True, convertToInstanceNorm=False):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        self.pretrained = pretrained
        self.convertToInstanceNorm = convertToInstanceNorm

        feClass = getattr(resnet, self.backbone)
        self.fe = feClass(self.pretrained)
        self.fe = nn.Sequential(*list(self.fe.children())[:-2])
        if self.convertToInstanceNorm:
            for layer in self.fe.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer = nn.InstanceNorm2d(layer.num_features,
                                    affine=False, track_running_stats=False)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.cls = nn.Linear(self.in_channels[self.backbone], self.num_classes, bias=True)


    def forward(self, x, return_fVec=False):
        fVec = self.pool(self.fe(x)).squeeze()
        logits = self.cls(fVec)

        if return_fVec:
            return logits, fVec
        else:
            return logits