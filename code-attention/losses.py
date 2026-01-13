"""
Loss functions for HRF segmentation
Using Focal Tversky Loss (recommended for class imbalance in medical imaging)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    
    def __init__(self, smooth: float = 1.0, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, 1, H, W) - predicted logits
            targets: (B, 1, H, W) - ground truth masks
        """
        predictions = torch.sigmoid(predictions)
        
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1.0 - dice


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - excellent for severe class imbalance
    Great for medical image segmentation where HRF are rare
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 4/3, smooth: float = 1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha      # Weight for false positives
        self.beta = beta        # Weight for false negatives
        self.gamma = gamma      # Focal parameter
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, 1, H, W) - predicted logits
            targets: (B, 1, H, W) - ground truth masks
        """
        predictions = torch.sigmoid(predictions)
        
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (predictions_flat * targets_flat).sum()
        FP = ((1.0 - targets_flat) * predictions_flat).sum()
        FN = (targets_flat * (1.0 - predictions_flat)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1.0 - tversky) ** self.gamma
        
        return focal_tversky


class DiceBCELoss(nn.Module):
    """Combination of Dice Loss and Binary Cross Entropy"""
    
    def __init__(self, smooth: float = 1.0, dice_weight: float = 0.5, bce_weight: float = 0.5, **kwargs):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, 1, H, W) - predicted logits
            targets: (B, 1, H, W) - ground truth masks
        """
        predictions_prob = torch.sigmoid(predictions)
        
        predictions_flat = predictions_prob.view(-1)
        targets_flat = targets.view(-1)
        
        # Dice Loss
        intersection = (predictions_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (predictions_flat.sum() + targets_flat.sum() + self.smooth)
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets)
        
        # Combined
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments
        
    Returns:
        Loss function instance
    """
    loss_dict = {
        'dice': DiceLoss,
        'focal_tversky': FocalTverskyLoss,
        'dice_bce': DiceBCELoss,
    }
    
    if loss_name.lower() not in loss_dict:
        raise ValueError(f"Loss '{loss_name}' not found. Available: {list(loss_dict.keys())}")
    
    return loss_dict[loss_name.lower()](**kwargs)
