# -*- coding: utf-8 -*-
"""
Loss functions for the IR-UNet_plus training pipeline.

Available losses:
    MSE   -> nn.MSELoss
    MAE   -> nn.L1Loss
    Huber -> HuberLoss (smooth L1)
"""

import torch
import torch.nn as nn


# ============================================================
# LOSSES
# ============================================================
class HuberLoss(nn.Module):
    """
    Huber loss (smooth L1).

    For |r| <= delta: 0.5 * r^2
    For |r| >  delta: delta * (|r| - 0.5 * delta)
    """

    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target):
        r = pred - target
        a = torch.abs(r)
        q = torch.clamp(a, max=self.delta)
        l = a - q
        loss = 0.5 * q ** 2 + self.delta * l

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================
# FACTORY
# ============================================================
def build_loss(config):
    """
    Build a loss module from a CONFIG dictionary.

    Recognised `loss_function` values:
        'mse'   -> nn.MSELoss
        'mae'   -> nn.L1Loss
        'huber' -> HuberLoss
    """
    lf = config['loss_function'].lower()

    if lf == 'mse':
        criterion = nn.MSELoss()
        loss_str = "MSE"

    elif lf == 'mae':
        criterion = nn.L1Loss()
        loss_str = "MAE"

    elif lf == 'huber':
        delta = config.get('huber_delta', 1.0)
        criterion = HuberLoss(delta=delta)
        loss_str = f"Huber (delta={delta})"

    else:
        raise ValueError("loss_function must be 'mse', 'mae', or 'huber'")

    return criterion, loss_str
