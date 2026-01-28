"""
Enhanced test script for Point Transformer V3 that calculates metrics 
both with all classes and excluding specified classes, with precise allAcc calculation
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
import json
import pickle

from data_utils import create_dataloaders
from model_v3 import AverageMeter, intersectionAndUnionGPU, PointTransformerV3


class PTv3WithHead(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.backbone = PointTransformerV3(in_channels=in_channels, enable_flash=True)
        self.cls = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, data_dict):
        point = self.backbone(data_dict)
        logits = self.cls(point.feat)
        return logits


def validate_with_detailed_stats(loader, model, criterion, num_classes, ignore_label, class_names, fast_debug=False):
    """
    Run evaluation on a dataloader and return detailed statistics for PTv3
    """
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    pred_list = []

    with torch.no_grad():
        for i, (grid_coord, coord, feat, target, offset) in enumerate(tqdm(loader, desc="Testing", unit="batch")):
            grid_coord, coord, feat, target, offset = (
                grid_coord.cuda(non_blocking=True),
                coord.cuda(non_blocking=True),
                feat.cuda(non_blocking=True),
                target.cuda(non_blocking=True),
                offset.cuda(non_blocking=True),
            )

            if target.shape[-1] == 1:
                target = target[:, 0]

            data_dict = {
                'coord': coord,
                'grid_coord': grid_coord,
                'feat': coord.float(),
                'offset': offset,
            }

            output = model(data_dict)
            loss = criterion(output, target)
            output = output.max(1)[1]

            n = grid_coord.size(0)
            intersection, union, target_count = intersectionAndUnionGPU(output, target, num_classes, ignore_label)
            intersection, union, target_count = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target_count.cpu().numpy(),
            )

            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target_count)
            loss_meter.update(loss.item(), n)

            pred_list.append(output.cpu().numpy())

            if fast_debug and i >= 0:
                break

    # Calculate metrics for all classes
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    
    # Handle valid mask for original metrics
    valid_mask = union_meter.sum > 0
    if valid_mask.any():
        mIoU = np.mean(iou_class[valid_mask])
        mAcc = np.mean(accuracy_class[valid_mask])
    else:
        mIoU = 0.0
        mAcc = 0.0
    
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # Return detailed statistics
    return {
        'loss': loss_meter.avg,
        'mIoU': mIoU,
        'mAcc': mAcc,
        'allAcc': allAcc,
        'iou_class': iou_class,
        'accuracy_class': accuracy_class,
        'intersection_sum': intersection_meter.sum,
        'union_sum': union_meter.sum,
        'target_sum': target_meter.sum,
        'predictions': np.hstack(pred_list)
    }


def calculate_metrics_excluding_classes(stats, class_names, exclude_classes):
    """
    Calculate precise metrics excluding specified classes
    
    Args:
        stats: Dictionary containing detailed statistics
        class_names: List of all class names
        exclude_classes: List of class names to exclude
    
    Returns:
        Dictionary with recalculated metrics
    """
    # Get indices of classes to include
    include_indices = []
    included_names = []
    excluded_indices = []
    
    for i, name in enumerate(class_names):
        if name not in exclude_classes:
            include_indices.append(i)
            included_names.append(name)
        else:
            excluded_indices.append(i)
    
    # Extract stats for included classes only
    intersection_sum = stats['intersection_sum']
    union_sum = stats['union_sum']
    target_sum = stats['target_sum']
    
    included_intersection = intersection_sum[include_indices]
    included_union = union_sum[include_indices]
    included_target = target_sum[include_indices]
    
    # Calculate per-class metrics for included classes
    iou_included = included_intersection / (included_union + 1e-10)
    acc_included = included_intersection / (included_target + 1e-10)
    
    # Calculate aggregate metrics (PRECISE)
    new_mIoU = np.mean(iou_included)
    new_mAcc = np.mean(acc_included)
    new_allAcc = np.sum(included_intersection) / (np.sum(included_target) + 1e-10)
    
    return {
        'mIoU': new_mIoU,
        'mAcc': new_mAcc,
        'allAcc': new_allAcc,
        'iou_per_class': {included_names[i]: iou_included[i] for i in range(len(included_names))},
        'acc_per_class': {included_names[i]: acc_included[i] for i in range(len(included_names))},
        'pixel_counts': {included_names[i]: int(included_target[i]) for i in range(len(included_names))},
        'correct_pixels': {included_names[i]: int(included_intersection[i]) for i in range(len(included_names))},
        'included_classes': included_names,
        'excluded_classes': exclude_classes
    }


def write_comprehensive_log(log_file, class_names, all_stats, excluded_stats, exclude_classes, model_info=None):
    """
    Write comprehensive test results to log file
    """
    with open(log_file, "w") as f:
        # Write header
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE TEST RESULTS - Point Transformer V3\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if model_info:
            f.write(f"Model: {model_info.get('name', 'PTv3')}\n")
            f.write(f"Checkpoint Epoch: {model_info.get('epoch', 'unknown')}\n")
        f.write("="*80 + "\n\n")
        
        # Original results (all classes)
        f.write("ORIGINAL RESULTS (All Classes):\n")
        f.write("-"*40 + "\n")
        f.write(f"Test Results: mIoU={all_stats['mIoU']:.4f}, mAcc={all_stats['mAcc']:.4f}, allAcc={all_stats['allAcc']:.4f}\n")
        for i, cls in enumerate(class_names):
            f.write(f"{cls:<20} IoU={all_stats['iou_class'][i]:.4f}, Acc={all_stats['accuracy_class'][i]:.4f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Recalculated results (excluding specified classes)
        f.write(f"RECALCULATED RESULTS (Excluding: {', '.join(exclude_classes)}):\n")
        f.write("-"*40 + "\n")
        f.write(f"Test Results: mIoU={excluded_stats['mIoU']:.4f}, mAcc={excluded_stats['mAcc']:.4f}, allAcc={excluded_stats['allAcc']:.4f} (PRECISE)\n")
        f.write("\nIncluded Classes Performance:\n")
        for cls in excluded_stats['included_classes']:
            f.write(f"{cls:<20} IoU={excluded_stats['iou_per_class'][cls]:.4f}, "
                   f"Acc={excluded_stats['acc_per_class'][cls]:.4f}, "
                   f"Pixels={excluded_stats['pixel_counts'][cls]:,}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Comparison summary
        f.write("METRICS COMPARISON:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Metric':<10} {'All Classes':>15} {'Excluding 3':>15} {'Difference':>15}\n")
        f.write(f"{'mIoU':<10} {all_stats['mIoU']:>15.4f} {excluded_stats['mIoU']:>15.4f} "
               f"{(excluded_stats['mIoU'] - all_stats['mIoU']):>+15.4f}\n")
        f.write(f"{'mAcc':<10} {all_stats['mAcc']:>15.4f} {excluded_stats['mAcc']:>15.4f} "
               f"{(excluded_stats['mAcc'] - all_stats['mAcc']):>+15.4f}\n")
        f.write(f"{'allAcc':<10} {all_stats['allAcc']:>15.4f} {excluded_stats['allAcc']:>15.4f} "
               f"{(excluded_stats['allAcc'] - all_stats['allAcc']):>+15.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Note: allAcc for excluded classes is calculated precisely using pixel-level statistics.\n")
        f.write("Model: Point Transformer V3 with Flash Attention\n")
        f.write("="*80 + "\n")


def save_detailed_stats(save_dir, all_stats, excluded_stats, class_names, exclude_classes, model_info=None):
    """
    Save detailed statistics to JSON and pickle files
    """
    detailed_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'Point Transformer V3',
        'model_info': model_info,
        'class_names': class_names,
        'exclude_classes': exclude_classes,
        'all_classes_stats': {
            'mIoU': float(all_stats['mIoU']),
            'mAcc': float(all_stats['mAcc']),
            'allAcc': float(all_stats['allAcc']),
            'loss': float(all_stats['loss']),
            'iou_per_class': all_stats['iou_class'].tolist(),
            'acc_per_class': all_stats['accuracy_class'].tolist(),
            'intersection_sum': all_stats['intersection_sum'].tolist(),
            'union_sum': all_stats['union_sum'].tolist(),
            'target_sum': all_stats['target_sum'].tolist()
        },
        'excluded_classes_stats': {
            'mIoU': float(excluded_stats['mIoU']),
            'mAcc': float(excluded_stats['mAcc']),
            'allAcc': float(excluded_stats['allAcc']),
            'included_classes': excluded_stats['included_classes'],
            'excluded_classes': excluded_stats['excluded_classes'],
            'iou_per_class': {k: float(v) for k, v in excluded_stats['iou_per_class'].items()},
            'acc_per_class': {k: float(v) for k, v in excluded_stats['acc_per_class'].items()},
            'pixel_counts': excluded_stats['pixel_counts'],
            'correct_pixels': excluded_stats['correct_pixels']
        }
    }
    
    # Save as JSON
    json_path = os.path.join(save_dir, 'detailed_test_stats_ptv3.json')
    with open(json_path, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    
    # Save as pickle for exact numerical precision
    pkl_path = os.path.join(save_dir, 'detailed_test_stats_ptv3.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(detailed_data, f)
    
    print(f"Saved detailed statistics to:")
    print(f"  - {json_path}")
    print(f"  - {pkl_path}")


def run_enhanced_test_ptv3(cfg, checkpoint_path=None, exclude_classes=None):
    """
    Enhanced test function for PTv3 that calculates metrics with and without specified classes
    
    Args:
        cfg: Configuration object
        checkpoint_path: Path to model checkpoint (if None, uses model_best.pth)
        exclude_classes: List of class names to exclude (default: ['Vehicle', 'Pedestrian', 'Noise'])
    """
    if exclude_classes is None:
        exclude_classes = ['Vehicle', 'Pedestrian', 'Noise']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = cfg.class_names
    num_classes = len(class_names)
    
    print("="*80)
    print("ENHANCED TEST WITH PRECISE METRICS - Point Transformer V3")
    print("="*80)
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes to exclude: {exclude_classes}")
    print("="*80)

    # Load data
    print("\nLoading test data...")
    _, _, test_loader, _ = create_dataloaders(cfg)

    if getattr(cfg, "fast_debug", False):
        print("[DEBUG] Running test in fast_debug mode (1 batch only)")
        from itertools import islice
        test_loader = list(islice(test_loader, 1))

    # Build and load model
    print("Building Point Transformer V3 model...")
    model = PTv3WithHead(num_classes=num_classes, in_channels=cfg.feature_dim)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.save_path, "model", "model_best.pth")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    model_info = {
        'name': 'Point Transformer V3',
        'epoch': checkpoint.get('epoch', 'unknown'),
        'checkpoint_path': checkpoint_path
    }
    print(f"Loaded PTv3 model from epoch {model_info['epoch']}")

    # Run validation with detailed statistics
    print("\nRunning test...")
    all_stats = validate_with_detailed_stats(
        test_loader, model, criterion,
        num_classes, cfg.ignore_label,
        class_names,
        fast_debug=getattr(cfg, "fast_debug", False)
    )

    # Calculate metrics excluding specified classes
    print(f"\nCalculating metrics excluding {exclude_classes}...")
    excluded_stats = calculate_metrics_excluding_classes(all_stats, class_names, exclude_classes)

    # Save results
    save_dir = os.path.join(cfg.save_path, "model")
    os.makedirs(save_dir, exist_ok=True)
    
    # Write comprehensive log
    log_file = os.path.join(save_dir, "test_log_ptv3_enhanced.txt")
    write_comprehensive_log(log_file, class_names, all_stats, excluded_stats, exclude_classes, model_info)
    print(f"\nTest results written to: {log_file}")
    
    # Save detailed statistics
    save_detailed_stats(save_dir, all_stats, excluded_stats, class_names, exclude_classes, model_info)

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY - Point Transformer V3")
    print("="*80)
    print("\nAll Classes:")
    print(f"  mIoU: {all_stats['mIoU']:.4f}")
    print(f"  mAcc: {all_stats['mAcc']:.4f}")
    print(f"  allAcc: {all_stats['allAcc']:.4f}")
    
    print(f"\nExcluding {', '.join(exclude_classes)}:")
    print(f"  mIoU: {excluded_stats['mIoU']:.4f}")
    print(f"  mAcc: {excluded_stats['mAcc']:.4f}")
    print(f"  allAcc: {excluded_stats['allAcc']:.4f} (PRECISE)")
    
    # Show improvement
    print("\nImprovement after excluding classes:")
    print(f"  ΔmIoU: {(excluded_stats['mIoU'] - all_stats['mIoU']):+.4f}")
    print(f"  ΔmAcc: {(excluded_stats['mAcc'] - all_stats['mAcc']):+.4f}")
    print(f"  ΔallAcc: {(excluded_stats['allAcc'] - all_stats['allAcc']):+.4f}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)

    return all_stats, excluded_stats


# Backward compatibility function
def run_test(cfg, checkpoint_path=None):
    """
    Original run_test function for backward compatibility
    Runs enhanced test but returns only the standard results
    """
    all_stats, _ = run_enhanced_test_ptv3(cfg, checkpoint_path)
    return (all_stats['loss'], all_stats['mIoU'], all_stats['mAcc'], 
            all_stats['allAcc'], all_stats['iou_class'], 
            all_stats['accuracy_class'], all_stats['predictions'])


if __name__ == "__main__":
    from config import get_config
    
    # Configure your test - MODIFY THESE PARAMETERS
    config_kwargs = {
        'real_ratio': 0,  # Adjust based on your model (e.g., 25 for model_25_lr01)
        'fast_debug': False,  # Set to False for full test
        'batch_size': 32,
        'use_preprocessed': True,  # Based on your setup
        'model_name': 'point_transformer_v3',  # Specify PTv3
        # Add other config parameters as needed
    }
    
    # Get configuration
    cfg = get_config(**config_kwargs)
    
    # Specify checkpoint path (None to use default model_best.pth)
    checkpoint_path = None  # Or specify: "/path/to/your/ptv3/model_best.pth"
    
    # Specify classes to exclude
    exclude_classes = ['Vehicle', 'Pedestrian', 'Noise']
    
    # Run enhanced test
    all_stats, excluded_stats = run_enhanced_test_ptv3(
        cfg, 
        checkpoint_path=checkpoint_path,
        exclude_classes=exclude_classes
    )