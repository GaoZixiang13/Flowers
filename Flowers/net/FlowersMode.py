import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from Flowers.net import backbone

class modeFlowers(nn.Module):
    def __init__(self, num_classes):
        super(modeFlowers, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone.FlowersBone(self.num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x