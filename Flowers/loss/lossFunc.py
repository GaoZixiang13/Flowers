import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class flowersLoss(nn.Module):
    def __init__(self, input_shape, num_classes, device, label_smoothing=0, times=(8, 16, 32), cuda=True):
        super(flowersLoss, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device
        self.times = times
        self.cuda = cuda
        self.cal0 = torch.tensor([0]).to(self.device)
        # self.cal1 = torch.tensor([1]).to(self.device)
        self.cal_input = torch.tensor([self.input_shape]).to(self.device)
        self.label_smoothing = label_smoothing
        self.eps = 1e-8

    def forward(self, pred, tar):
        return self.Focalloss(pred, tar)

    def Focalloss(self, pred, target):
        # print(pred.shape == target.shape)
        alpha, gamma = .25, 2

        noobj_mask = (target == 0)
        pt = torch.clone(pred)
        pt[noobj_mask] = 1 - pred[noobj_mask]

        return -alpha * (1 - pt).pow(gamma) * torch.log(pt + self.eps)