"""
Evaluation metrics for segmentation models
"""

import numpy as np
import torch
from typing import Dict


def calculate_dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Calculate Dice Similarity Coefficient"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = np.sum(pred * target)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    
    return float(dice)


def calculate_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Calculate Intersection over Union (Jaccard Index)"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def calculate_precision_recall(pred: np.ndarray, target: np.ndarray) -> tuple:
    """Calculate precision and recall"""
    pred = pred.flatten()
    target = target.flatten()
    
    TP = np.sum(pred * target)
    FP = np.sum(pred * (1 - target))
    FN = np.sum((1 - pred) * target)
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    
    return float(precision), float(recall)


def calculate_f1_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate F1 Score"""
    precision, recall = calculate_precision_recall(pred, target)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return float(f1)


def calculate_specificity(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate specificity (True Negative Rate)"""
    pred = pred.flatten()
    target = target.flatten()
    
    TN = np.sum((1 - pred) * (1 - target))
    FP = np.sum(pred * (1 - target))
    
    specificity = TN / (TN + FP + 1e-8)
    return float(specificity)


def evaluate_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a batch of predictions.
    
    Args:
        predictions: Predicted logits (B, 1, H, W)
        targets: Ground truth masks (B, 1, H, W)
        threshold: Threshold for binarization
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Binarize predictions
    predictions = (predictions > threshold).astype(np.float32)
    
    # Calculate metrics for each sample
    batch_size = predictions.shape[0]
    metrics = {
        'dice': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'specificity': 0.0,
    }
    
    for i in range(batch_size):
        pred = predictions[i, 0]
        target = targets[i, 0]
        
        metrics['dice'] += calculate_dice_score(pred, target)
        metrics['iou'] += calculate_iou(pred, target)
        precision, recall = calculate_precision_recall(pred, target)
        metrics['precision'] += precision
        metrics['recall'] += recall
        metrics['f1'] += calculate_f1_score(pred, target)
        metrics['specificity'] += calculate_specificity(pred, target)
    
    # Average metrics
    for key in metrics:
        metrics[key] /= batch_size
    
    return metrics


class MetricsTracker:
    """Track metrics across epochs"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'specificity': [],
        }
    
    def update(self, batch_metrics: Dict[str, float]):
        """Update metrics with batch results"""
        for key, value in batch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_average(self) -> Dict[str, float]:
        """Get average metrics"""
        return {key: np.mean(values) if values else 0.0 for key, values in self.metrics.items()}
    
    def get_std(self) -> Dict[str, float]:
        """Get standard deviation of metrics"""
        return {key: np.std(values) if values else 0.0 for key, values in self.metrics.items()}
