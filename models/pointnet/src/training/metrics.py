"""
Shared metrics for training evaluation
"""
import numpy as np
import torch


def calculate_iou(pred, target, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class
    
    Args:
        pred: predicted class indices [B*N] or [B, N]
        target: target class indices [B*N] or [B, N]
        num_classes: number of classes
    
    Returns:
        iou_per_class: IoU for each class [num_classes]
        mean_iou: mean IoU across all classes
    """
    if len(pred.shape) > 1:
        pred = pred.view(-1)
    if len(target.shape) > 1:
        target = target.view(-1)
    
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        iou_per_class.append(iou)
    
    mean_iou = np.mean(iou_per_class)
    return np.array(iou_per_class), mean_iou


def calculate_accuracy(pred, target):
    """
    Calculate accuracy
    
    Args:
        pred: predicted class indices [B*N] or [B, N]
        target: target class indices [B*N] or [B, N]
    
    Returns:
        accuracy: overall accuracy
    """
    if len(pred.shape) > 1:
        pred = pred.view(-1)
    if len(target.shape) > 1:
        target = target.view(-1)
    
    correct = (pred == target).sum().item()
    total = target.size(0)
    
    return correct / total if total > 0 else 0.0


def calculate_per_class_accuracy(pred, target, num_classes):
    """
    Calculate per-class accuracy
    
    Args:
        pred: predicted class indices [B*N] or [B, N]
        target: target class indices [B*N] or [B, N]
        num_classes: number of classes
    
    Returns:
        per_class_acc: accuracy for each class [num_classes]
    """
    if len(pred.shape) > 1:
        pred = pred.view(-1)
    if len(target.shape) > 1:
        target = target.view(-1)
    
    per_class_acc = []
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    
    for cls in range(num_classes):
        mask = (target == cls)
        if mask.sum() > 0:
            per_class_total[cls] = mask.sum().item()
            per_class_correct[cls] = ((pred == cls) & mask).sum().item()
            per_class_acc.append(per_class_correct[cls] / per_class_total[cls])
        else:
            per_class_acc.append(0.0)
    
    return np.array(per_class_acc), per_class_correct, per_class_total

