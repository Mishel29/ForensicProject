import torch
import torch.nn as nn

class WeightedReconstructionLoss(nn.Module):
    def __init__(self, eta=0.001):
        super().__init__()
        self.eta = eta
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target, weight_map):
        return self.eta * self.l1_loss(weight_map * pred, weight_map * target)

class StandardReconstructionLoss(nn.Module):
    def __init__(self, eta=0.001):
        super().__init__()
        self.eta = eta
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target, weight_map=None):
        return self.eta * self.l1_loss(pred, target)

class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.bce_loss(pred, target)