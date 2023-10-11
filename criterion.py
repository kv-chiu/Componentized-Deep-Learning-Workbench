import torch
import torch.nn as nn
import torch.nn.functional as F


class myCriterion(nn.Module):
    '''
    Label Smoothing Cross Entropy Loss
    '''
    
    def __init__(self, config):
        super(myCriterion, self).__init__()
        self.config = config
        self.confidence = 1.0 - config.smoothing
        self.smoothing = config.smoothing
        self.dim = config.dim
        self.num_classes = config.num_classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        # breakpoint()
        target = F.one_hot(target, num_classes=self.num_classes)
        true_dist.scatter_(1, target.data.squeeze(), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))