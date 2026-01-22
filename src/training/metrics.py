"""
Evaluation metrics for semantic segmentation
Shared metrics calculation functions for both PointNet and PointNet2++
"""

import numpy as np


def calculate_iou_per_class(predictions, targets, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class
    
    Args:
        predictions: predicted class labels [N]
        targets: ground truth labels [N] 
        num_classes: number of classes
        
    Returns:
        iou_per_class: IoU for each class [num_classes]
    """
    iou_per_class = np.zeros(num_classes)
    
    for cls in range(num_classes):
        # True Positives: predicted=cls AND target=cls
        intersection = np.sum((predictions == cls) & (targets == cls))
        
        # Union: predicted=cls OR target=cls
        union = np.sum((predictions == cls) | (targets == cls))
        
        if union > 0:
            iou_per_class[cls] = intersection / union
        else:
            iou_per_class[cls] = 0.0  # No samples for this class
            
    return iou_per_class


def calculate_metrics(predictions, targets, num_classes, class_to_idx):
    """
    Calculate comprehensive metrics for semantic segmentation
    
    Args:
        predictions: predicted class labels [N]
        targets: ground truth labels [N]
        num_classes: number of classes
        class_to_idx: mapping from class names to indices
        
    Returns:
        dict with all metrics
    """
    # Convert to numpy if needed
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    
    # Overall accuracy (per-point accuracy)
    overall_acc = np.sum(predictions == targets) / len(targets)
    
    # Per-class IoU and mIoU
    iou_per_class = calculate_iou_per_class(predictions, targets, num_classes)
    miou = np.mean(iou_per_class)
    
    # Per-class accuracy
    class_acc = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    for cls in range(num_classes):
        mask = (targets == cls)
        if np.sum(mask) > 0:
            class_acc[cls] = np.sum((predictions == targets) & mask) / np.sum(mask)
            class_counts[cls] = np.sum(mask)
    
    # Mean class accuracy
    valid_classes = class_counts > 0
    mean_class_acc = np.mean(class_acc[valid_classes]) if np.any(valid_classes) else 0.0

    # Per-class precision, recall, F1
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        tp = np.sum((predictions == cls) & (targets == cls))
        fp = np.sum((predictions == cls) & (targets != cls))
        fn = np.sum((predictions != cls) & (targets == cls))
        precision_den = tp + fp
        recall_den = tp + fn
        precision_per_class[cls] = (tp / precision_den) if precision_den > 0 else 0.0
        recall_per_class[cls] = (tp / recall_den) if recall_den > 0 else 0.0
        denom = precision_per_class[cls] + recall_per_class[cls]
        f1_per_class[cls] = (2 * precision_per_class[cls] * recall_per_class[cls] / denom) if denom > 0 else 0.0

    return {
        'overall_accuracy': overall_acc * 100,
        'mean_iou': miou * 100,
        'iou_per_class': iou_per_class * 100,
        'class_accuracy': class_acc * 100,
        'mean_class_accuracy': mean_class_acc * 100,
        'class_counts': class_counts,
        'precision_per_class': precision_per_class * 100,
        'recall_per_class': recall_per_class * 100,
        'f1_per_class': f1_per_class * 100
    }




