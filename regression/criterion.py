import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.weights = None

    def forward(self, input, target):
        if self.weights is None or len(self.weights) != len(target):
            self.update_weights(target)
        return (self.weights * (input - target) ** 2).mean()

    def update_weights(self, target):
        values, counts = torch.unique(target, return_counts=True)
        weights = 1.0 / counts.float()
        weight_map = {v.item(): w for v, w in zip(values, weights)}
        self.weights = torch.tensor([weight_map[t.item()] for t in target], device=target.device)



class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        error = input - target
        is_small_error = torch.abs(error) <= self.delta
        
        squared_loss = 0.5 * error**2
        linear_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        
        loss = torch.where(is_small_error, squared_loss, linear_loss)
        return loss.mean()