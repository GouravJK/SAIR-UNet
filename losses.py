# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:53:50 2026

@author: ectyf
"""

# -*- coding: utf-8 -*-
"""
Loss functions for IR-UNet_plus training pipeline
=================================================
- RainWeightedMSELoss : rain-weighted MSE (prevents collapse on sparse rainfall)
- HuberLoss           : smooth L1 / Huber loss
"""

import torch
import torch.nn as nn


# ============================================================
# LOSSES
# ============================================================
class RainWeightedMSELoss(nn.Module):
    """
    Rain-weighted Mean Squared Error.

    Pixels where the target exceeds `rain_threshold` receive a weight of
    `rain_weight`, others receive weight 1.0. This up-weights the sparse
    rainy pixels so the network does not collapse to predicting zeros.
    """

    def __init__(self, rain_threshold=0.05, rain_weight=25.0):
        super().__init__()
        self.rain_threshold = rain_threshold
        self.rain_weight = rain_weight

    def forward(self, pred, target):
        w = torch.where(
            target > self.rain_threshold,
            torch.full_like(target, self.rain_weight),
            torch.ones_like(target),
        )
        return (w * (pred - target) ** 2).mean()


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
        'rain_mse' -> RainWeightedMSELoss
        'mse'      -> nn.MSELoss
        'mae'      -> nn.L1Loss
        'huber'    -> HuberLoss
    """
    lf = config['loss_function'].lower()

    if lf == 'rain_mse':
        criterion = RainWeightedMSELoss(
            rain_threshold=config.get('rain_threshold', 0.05),
            rain_weight=config.get('rain_weight', 25.0),
        )
        loss_str = (f"Rain-Weighted MSE "
                    f"(threshold={config.get('rain_threshold', 0.05)}, "
                    f"weight={config.get('rain_weight', 25.0)})")

    elif lf == 'mse':
        criterion = nn.MSELoss()
        loss_str = "MSE"

    elif lf == 'mae':
        criterion = nn.L1Loss()
        loss_str = "MAE"

    elif lf == 'huber':
        criterion = HuberLoss(delta=config.get('huber_delta', 1.0))
        loss_str = f"Huber (delta={config.get('huber_delta', 1.0)})"

    else:
        raise ValueError(
            "loss_function must be 'rain_mse', 'mse', 'mae', or 'huber'"
        )

    return criterion, loss_str