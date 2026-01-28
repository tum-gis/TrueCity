"""
Enhanced test script that calculates metrics both with all classes 
and excluding specified classes, with precise allAcc calculation,
and saves per-point XYZ with predicted labels to .txt, .npy, and .ply.
Includes coordinate denormalization and label mapping.
Works with both preprocessed and original data.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
import json
import pickle

from model import Model, AverageMeter, intersectionAndUnionGPU
from data_utils import create_dataloaders


def write_ply_file(points, output_path):
    """
    Write PLY file with point coordinates and label information
    
    Args:
        points: numpy array with shape (N, 4), containing x, y, z, label
        output_path: output file path
    """
    n_points = len(points)
    
    # PLY file header
    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property int label
end_header
"""
    
    # Write file
    with open(output_path, 'w') as f:
        f.write(header)
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(point[3])}\n")


def denormalize_coordinates(normalized_xyz, original_mean, original_max_norm):
    """
    Denormalize coordinates to original scale
    
    Args:
        normalized_xyz: normalized coordinates (N, 3)
        original_mean: original mean of coordinates (3,)
        original_max_norm: original max norm value
    
    Returns:
        denormalized coordinates (N, 3)
    """
    # Reverse the normalization process
    denorm_xyz = normalized_xyz * original_max_norm
    denorm_xyz = denorm_xyz + original_mean
    return denorm_xyz


def map_labels(predictions, label_mapping=None):
    """
    Map specific labels to new values
    
    Args:
        predictions: predicted labels (N,)
        label_mapping: dict mapping old labels to new labels
    
    Returns:
        mapped labels (N,)
    """
    if label_mapping is None:
        label_mapping = {3: 11, 4: 11}  # Default mapping: Vehicle and Pedestrian to 11
    
    mapped_predictions = predictions.copy()
    for old_label, new_label in label_mapping.items():
        mapped_predictions[mapped_predictions == old_label] = new_label
    
    return mapped_predictions


def validate_with_detailed_stats(loader, model, criterion, num_classes, ignore_label, class_names, original_stats=None, fast_debug=False):
    """
    Run evaluation on a dataloader and return detailed statistics.
    Also collects coordinates and predicted labels for saving.
    Properly handles local normalization used during training.
    """
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()

    pred_list = []
    coords_list = []
    # Store original coordinates before any normalization
    original_coords_list = []
    batch_norm_params = []

    with torch.no_grad():
        for i, (points, feat, target, offset) in enumerate(tqdm(loader, desc="Testing", unit="batch")):
            # points: (N, 3) - these are locally normalized coordinates from data loader
            
            points, feat, target, offset = (
                points.cuda(non_blocking=True),
                feat.cuda(non_blocking=True),
                target.cuda(non_blocking=True),
                offset.cuda(non_blocking=True),
            )
            if target.shape[-1] == 1:
                target = target[:, 0]

            output = model([points, feat, offset])      # logits, shape (N, num_classes)
            loss = criterion(output, target)
            output = output.max(1)[1]                   # predicted labels, shape (N,)

            n = points.size(0)
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

            # Collect per-point results
            pred_list.append(output.cpu().numpy())      # (N,)
            
            # Get locally normalized coordinates
            locally_normalized_coords = points.cpu().numpy()    # (N,3) - locally normalized
            coords_list.append(locally_normalized_coords)
            
            # We need to reverse the local normalization to get back to global coordinates,
            # then apply global denormalization to get original coordinates
            
            # Since we can't easily reverse local normalization without the original data,
            # we'll use a different approach: store the batch boundaries and process later
            
            # For now, store the locally normalized coordinates and batch info
            batch_info = {
                'start_idx': sum(len(p) for p in pred_list[:-1]),  # Starting index for this batch
                'end_idx': sum(len(p) for p in pred_list),         # Ending index for this batch
                'coords': locally_normalized_coords.copy()
            }
            batch_norm_params.append(batch_info)

            if fast_debug and i >= 0:
                break

    # Concatenate over all batches
    all_pred = np.concatenate(pred_list, axis=0)   # (TotalN,)
    all_xyz_normalized = np.concatenate(coords_list, axis=0)  # (TotalN, 3) - locally normalized

    print(f"DEBUG: Total predictions shape: {all_pred.shape}")
    print(f"DEBUG: Total coordinates shape: {all_xyz_normalized.shape}")
    print(f"DEBUG: Prediction distribution: {np.unique(all_pred, return_counts=True)}")

    # Calculate metrics for all classes
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)

    valid_mask = union_meter.sum > 0
    if valid_mask.any():
        mIoU = np.mean(iou_class[valid_mask])
        mAcc = np.mean(accuracy_class[valid_mask])
    else:
        mIoU = 0.0
        mAcc = 0.0

    allAcc = np.sum(intersection_meter.sum) / (np.sum(target_meter.sum) + 1e-10)

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
        # payloads for saving:
        'predictions': all_pred,                      # (TotalN,)
        'coords_locally_normalized': all_xyz_normalized,  # (TotalN,3) - locally normalized
        'batch_norm_params': batch_norm_params,       # List of batch info
    }


def calculate_metrics_excluding_classes(stats, class_names, exclude_classes):
    """
    Calculate precise metrics excluding specified classes.
    """
    include_indices = []
    included_names = []
    excluded_indices = []

    for i, name in enumerate(class_names):
        if name not in exclude_classes:
            include_indices.append(i)
            included_names.append(name)
        else:
            excluded_indices.append(i)

    intersection_sum = stats['intersection_sum']
    union_sum = stats['union_sum']
    target_sum = stats['target_sum']

    included_intersection = intersection_sum[include_indices]
    included_union = union_sum[include_indices]
    included_target = target_sum[include_indices]

    iou_included = included_intersection / (included_union + 1e-10)
    acc_included = included_intersection / (included_target + 1e-10)

    new_mIoU = np.mean(iou_included) if len(iou_included) > 0 else 0.0
    new_mAcc = np.mean(acc_included) if len(acc_included) > 0 else 0.0
    new_allAcc = np.sum(included_intersection) / (np.sum(included_target) + 1e-10) if np.sum(included_target) > 0 else 0.0

    return {
        'mIoU': new_mIoU,
        'mAcc': new_mAcc,
        'allAcc': new_allAcc,
        'iou_per_class': {included_names[i]: float(iou_included[i]) for i in range(len(included_names))},
        'acc_per_class': {included_names[i]: float(acc_included[i]) for i in range(len(included_names))},
        'pixel_counts': {included_names[i]: int(included_target[i]) for i in range(len(included_names))},
        'correct_pixels': {included_names[i]: int(included_intersection[i]) for i in range(len(included_names))},
        'included_classes': included_names,
        'excluded_classes': exclude_classes
    }


def write_comprehensive_log(log_file, class_names, all_stats, excluded_stats, exclude_classes):
    """
    Write comprehensive test results to log file.
    """
    with open(log_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE TEST RESULTS\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write("ORIGINAL RESULTS (All Classes):\n")
        f.write("-"*40 + "\n")
        f.write(f"Test Results: mIoU={all_stats['mIoU']:.4f}, mAcc={all_stats['mAcc']:.4f}, allAcc={all_stats['allAcc']:.4f}\n")
        for i, cls in enumerate(class_names):
            f.write(f"{cls:<20} IoU={all_stats['iou_class'][i]:.4f}, Acc={all_stats['accuracy_class'][i]:.4f}\n")

        f.write("\n" + "="*80 + "\n\n")

        f.write(f"RECALCULATED RESULTS (Excluding: {', '.join(exclude_classes)}):\n")
        f.write("-"*40 + "\n")
        f.write(f"Test Results: mIoU={excluded_stats['mIoU']:.4f}, mAcc={excluded_stats['mAcc']:.4f}, allAcc={excluded_stats['allAcc']:.4f} (PRECISE)\n")
        f.write("\nIncluded Classes Performance:\n")
        for cls in excluded_stats['included_classes']:
            f.write(f"{cls:<20} IoU={excluded_stats['iou_per_class'][cls]:.4f}, "
                    f"Acc={excluded_stats['acc_per_class'][cls]:.4f}, "
                    f"Pixels={excluded_stats['pixel_counts'][cls]:,}\n")

        f.write("\n" + "="*80 + "\n\n")

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
        f.write("="*80 + "\n")


def save_detailed_stats(save_dir, all_stats, excluded_stats, class_names, exclude_classes):
    """
    Save detailed statistics to JSON and pickle files.
    """
    detailed_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
            'excluded_classes': excluded_stats['excluded_classes']
        }
    }

    json_path = os.path.join(save_dir, 'detailed_test_stats.json')
    with open(json_path, 'w') as f:
        json.dump(detailed_data, f, indent=2)

    pkl_path = os.path.join(save_dir, 'detailed_test_stats.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(detailed_data, f)

    print("Saved detailed statistics to:")
    print(f"  - {json_path}")
    print(f"  - {pkl_path}")


def save_pointwise_results_with_denorm(save_dir, all_stats, original_stats=None, 
                                     label_mapping=None, prefix="test_points"):
    """
    Save per-point XYZ with predicted labels to .txt, .npy, and .ply.
    Uses preserved original coordinates when available.
    
    Args:
        save_dir: directory to save files
        all_stats: statistics from validation
        original_stats: dict with 'mean' and 'max_norm' for denormalization
        label_mapping: dict mapping old labels to new labels
        prefix: file prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use preserved original coordinates if available
    if 'coords_original' in all_stats and all_stats['coords_original'] is not None:
        denorm_xyz = all_stats['coords_original']
        print("Using preserved original coordinates (most accurate)")
    else:
        # Fallback: try to denormalize from locally normalized coordinates
        if 'coords_locally_normalized' in all_stats:
            xyz = all_stats['coords_locally_normalized']
        else:
            xyz = all_stats.get('coords', all_stats.get('coords_normalized', None))
        
        if xyz is None:
            print("ERROR: No coordinate data found in all_stats")
            return None, None, None
            
        if original_stats is not None:
            print("WARNING: Using global denormalization on locally normalized data")
            print("This may not be accurate. Consider using preserved original coordinates.")
            denorm_xyz = denormalize_coordinates(
                xyz, 
                original_stats['mean'], 
                original_stats['max_norm']
            )
        else:
            print("Warning: No original statistics provided, using normalized coordinates")
            denorm_xyz = xyz

    pred = all_stats['predictions']     # (N,)

    # Verify coordinate and prediction array lengths match
    if len(denorm_xyz) != len(pred):
        print(f"ERROR: Coordinate length {len(denorm_xyz)} != prediction length {len(pred)}")
        return None, None, None

    # Map labels (3, 4 -> 11)
    if label_mapping is None:
        label_mapping = {3: 11, 4: 11}
    
    print(f"Mapping labels: {label_mapping}")
    mapped_pred = map_labels(pred, label_mapping)
    
    # Show label mapping statistics
    unique_orig, counts_orig = np.unique(pred, return_counts=True)
    unique_mapped, counts_mapped = np.unique(mapped_pred, return_counts=True)
    print("Original label distribution:")
    for label, count in zip(unique_orig, counts_orig):
        print(f"  Label {label}: {count:,} points")
    print("Mapped label distribution:")
    for label, count in zip(unique_mapped, counts_mapped):
        print(f"  Label {label}: {count:,} points")

    # Prepare data for saving
    data_npy = np.concatenate(
        [denorm_xyz.astype(np.float32), mapped_pred.astype(np.float32)[:, None]],
        axis=1
    )  # (N,4) -> [x, y, z, pred_label]

    # Save NPY file
    npy_path = os.path.join(save_dir, f"{prefix}_{ts}.npy")
    np.save(npy_path, data_npy)

    # Save TXT file: x y z pred_label
    txt_path = os.path.join(save_dir, f"{prefix}_{ts}.txt")
    with open(txt_path, "w") as f:
        for i in range(denorm_xyz.shape[0]):
            x, y, z = denorm_xyz[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(mapped_pred[i])}\n")

    # Save PLY file for CloudCompare visualization
    ply_path = os.path.join(save_dir, f"{prefix}_{ts}.ply")
    write_ply_file(data_npy, ply_path)

    print("Saved point-wise predictions to:")
    print(f"  - {txt_path}")
    print(f"  - {npy_path}")
    print(f"  - {ply_path}")
    print(f"Total points saved: {len(denorm_xyz):,}")

    return txt_path, npy_path, ply_path


def compute_global_normalization_stats_for_test(data_root, real_ratio, use_preprocessed=True):
    """
    Compute global normalization statistics - adapted for your data structure
    """
    all_xyz = []
    
    if use_preprocessed:
        # Use preprocessed data from octree_fps folder (YOUR ACTUAL DATA PATH)
        train_folder = os.path.join(data_root, 'train', f"datav2_{real_ratio}_octree_fps")
        
        if not os.path.exists(train_folder):
            print(f"Preprocessed folder not found: {train_folder}")
            # Fallback to original method if preprocessed folder doesn't exist
            return compute_original_stats(data_root, real_ratio)
        
        # Load all .npy files in the folder
        npy_files = [f for f in os.listdir(train_folder) if f.endswith('.npy')]
        if not npy_files:
            print(f"No .npy files found in {train_folder}")
            return compute_original_stats(data_root, real_ratio)
        
        print(f"Computing normalization from {len(npy_files)} preprocessed files...")
        
        for npy_file in sorted(npy_files):
            file_path = os.path.join(train_folder, npy_file)
            data = np.load(file_path)  # Shape: [2048, 4] -> [x, y, z, label]
            xyz = data[:, :3]  # X, Y, Z columns
            all_xyz.append(xyz)
    
    else:
        return compute_original_stats(data_root, real_ratio)
    
    if not all_xyz:
        print("No data loaded from preprocessed files, trying original method...")
        return compute_original_stats(data_root, real_ratio)
    
    # Combine all training data
    all_xyz = np.vstack(all_xyz)
    
    # Compute global statistics
    global_mean = np.mean(all_xyz, axis=0)
    centered_xyz = all_xyz - global_mean
    norms = np.linalg.norm(centered_xyz, axis=1)
    global_max_norm = np.max(norms)
    
    stats = {
        'mean': global_mean,
        'max_norm': global_max_norm
    }
    
    print(f"Computed normalization stats from {len(all_xyz):,} training points")
    print(f"  Mean: [{global_mean[0]:.6f}, {global_mean[1]:.6f}, {global_mean[2]:.6f}]")
    print(f"  Max norm: {global_max_norm:.6f}")
    
    return stats


def compute_original_stats(data_root, real_ratio):
    """
    Fallback method using original file structure (from your data_utils.py)
    """
    all_xyz = []
    train_files = [f"train1_{real_ratio}.npy", f"train2_{real_ratio}.npy"]
    
    for f in train_files:
        file_path = os.path.join(data_root, 'train', f)
        if os.path.exists(file_path):
            data = np.load(file_path)
            xyz = data[:, :3]  # X, Y, Z columns
            all_xyz.append(xyz)
            print(f"Loaded {len(xyz):,} points from {f}")
        else:
            print(f"Warning: {file_path} not found!")
    
    if not all_xyz:
        raise ValueError("No training data found for computing normalization stats")
    
    # Combine all training data
    all_xyz = np.vstack(all_xyz)
    
    # Compute global statistics
    global_mean = np.mean(all_xyz, axis=0)
    centered_xyz = all_xyz - global_mean
    norms = np.linalg.norm(centered_xyz, axis=1)
    global_max_norm = np.max(norms)
    
    stats = {
        'mean': global_mean,
        'max_norm': global_max_norm
    }
    
    print(f"Computed normalization stats from {len(all_xyz):,} training points (original method)")
    print(f"  Mean: [{global_mean[0]:.6f}, {global_mean[1]:.6f}, {global_mean[2]:.6f}]")
    print(f"  Max norm: {global_max_norm:.6f}")
    
    return stats


def load_normalization_stats_from_config(cfg):
    """Load normalization statistics from config or file"""
    # First try to get from config
    if hasattr(cfg, 'normalization_stats') and cfg.normalization_stats is not None:
        return cfg.normalization_stats
    
    # Try to load from saved file
    stats_path = os.path.join(cfg.save_path, "normalization_stats.json")
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from: {stats_path}")
        if stats_path.endswith('.json'):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            stats['mean'] = np.array(stats['mean'])
        else:
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
        return stats
    
    # Try alternative paths
    pkl_path = os.path.join(cfg.save_path, "normalization_stats.pkl")
    if os.path.exists(pkl_path):
        print(f"Loading normalization stats from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            stats = pickle.load(f)
        return stats
    
    # If no saved stats found, try to compute them on the fly
    print("No saved normalization statistics found. Computing from training data...")
    try:
        data_root = cfg.base_data_root
        use_preprocessed = getattr(cfg, 'use_preprocessed', False)
        stats = compute_global_normalization_stats_for_test(data_root, cfg.real_ratio, use_preprocessed)
        
        # Save the computed stats for future use
        save_path = os.path.join(cfg.save_path, "normalization_stats.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        stats_json = {
            'mean': stats['mean'].tolist(),
            'max_norm': float(stats['max_norm'])
        }
        with open(save_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        print(f"Computed and saved normalization stats to: {save_path}")
        return stats
        
    except Exception as e:
        print(f"Warning: Failed to compute normalization stats: {e}")
        return None


def run_enhanced_test(cfg, checkpoint_path=None, exclude_classes=None, 
                     original_stats=None, label_mapping=None):
    """
    Enhanced test function that calculates metrics with and without specified classes
    and saves per-point outputs to TXT/NPY/PLY with denormalization and label mapping.
    
    Args:
        cfg: configuration object
        checkpoint_path: path to model checkpoint
        exclude_classes: list of class names to exclude from metrics
        original_stats: dict with 'mean' and 'max_norm' for coordinate denormalization
        label_mapping: dict mapping old labels to new labels
    """
    if exclude_classes is None:
        exclude_classes = ['Vehicle', 'Pedestrian', 'Noise']
    
    if label_mapping is None:
        label_mapping = {3: 11, 4: 11}  # Vehicle and Pedestrian to class 11

    # Auto-load normalization stats if not provided
    if original_stats is None:
        original_stats = load_normalization_stats_from_config(cfg)
    
    if original_stats is None:
        print("Warning: No normalization statistics available. Using normalized coordinates.")
    else:
        print(f"Using normalization stats - Mean: {original_stats['mean']}, Max norm: {original_stats['max_norm']}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = cfg.class_names
    num_classes = len(class_names)

    print("="*80)
    print("ENHANCED TEST WITH PRECISE METRICS AND DENORMALIZATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes to exclude: {exclude_classes}")
    print(f"Label mapping: {label_mapping}")
    if original_stats:
        print(f"Denormalization - Mean: {original_stats['mean']}")
        print(f"Denormalization - Max norm: {original_stats['max_norm']}")
    else:
        print("Warning: No denormalization parameters provided")
    print("="*80)

    # Load data with special handling for test data to preserve original coordinates
    print("\nLoading test data...")
    if getattr(cfg, 'use_preprocessed', False):
        from data_utils import create_test_dataloader_with_original_coords
        # Use special test dataloader that preserves original coordinates
        test_loader, original_coords_blocks = create_test_dataloader_with_original_coords(cfg)
        # Concatenate all original coordinates
        all_original_coords = np.vstack(original_coords_blocks)
        print(f"Loaded {len(all_original_coords):,} original coordinates for denormalization")
    else:
        _, _, test_loader, _ = create_dataloaders(cfg)
        all_original_coords = None
    
    # Get normalization stats
    if original_stats is None:
        original_stats = load_normalization_stats_from_config(cfg)
    
    if original_stats is None:
        print("ERROR: No normalization statistics available!")
        print("Please run compute_normalization_stats.py first to generate normalization parameters.")
        return None, None
    else:
        print(f"Successfully loaded normalization stats:")
        print(f"  Mean: [{original_stats['mean'][0]:.6f}, {original_stats['mean'][1]:.6f}, {original_stats['mean'][2]:.6f}]")
        print(f"  Max norm: {original_stats['max_norm']:.6f}")
    
    if getattr(cfg, "fast_debug", False):
        print("[DEBUG] Running test in fast_debug mode (1 batch only)")
        test_iter = iter(test_loader)
        test_loader = [next(test_iter)]

    # Build and load model
    print("Building model...")
    model = Model(c=cfg.feature_dim, k=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.save_path, "model", "model_best.pth")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Run validation with detailed statistics
    print("\nRunning test...")
    all_stats = validate_with_detailed_stats(
        test_loader, model, criterion,
        num_classes, cfg.ignore_label,
        class_names,
        original_stats=original_stats,
        fast_debug=getattr(cfg, "fast_debug", False)
    )
    
    # Add original coordinates to stats if available
    if all_original_coords is not None:
        print("Using preserved original coordinates for accurate denormalization")
        all_stats['coords_original'] = all_original_coords
    else:
        print("No preserved original coordinates, will use computed denormalization")
        all_stats['coords_original'] = None

    # Calculate metrics excluding specified classes
    print(f"\nCalculating metrics excluding {exclude_classes}...")
    excluded_stats = calculate_metrics_excluding_classes(all_stats, class_names, exclude_classes)

    # Save results
    save_dir = os.path.join(cfg.save_path, "model")
    os.makedirs(save_dir, exist_ok=True)

    # Write comprehensive log
    log_file = os.path.join(save_dir, "test_log_without34.txt")
    write_comprehensive_log(log_file, class_names, all_stats, excluded_stats, exclude_classes)
    print(f"\nTest results written to: {log_file}")

    # Save detailed statistics
    save_detailed_stats(save_dir, all_stats, excluded_stats, class_names, exclude_classes)

    # Save per-point XYZ + pred_label to TXT/NPY/PLY with denormalization and label mapping
    save_pointwise_results_with_denorm(
        save_dir, all_stats, 
        original_stats=original_stats,
        label_mapping=label_mapping,
        prefix="test_points"
    )

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\nAll Classes:")
    print(f"  mIoU: {all_stats['mIoU']:.4f}")
    print(f"  mAcc: {all_stats['mAcc']:.4f}")
    print(f"  allAcc: {all_stats['allAcc']:.4f}")

    print(f"\nExcluding {', '.join(exclude_classes)}:")
    print(f"  mIoU: {excluded_stats['mIoU']:.4f}")
    print(f"  mAcc: {excluded_stats['mAcc']:.4f}")
    print(f"  allAcc: {excluded_stats['allAcc']:.4f} (PRECISE)")

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("Files saved: TXT, NPY, PLY (ready for CloudCompare)")
    print("="*80)

    return all_stats, excluded_stats


if __name__ == "__main__":
    from config import get_config

    # Configure your test - MODIFY THESE PARAMETERS
    config_kwargs = {
        'real_ratio': 0,     # Adjust based on your model (match with compute_normalization_stats.py)
        'fast_debug': False,  # Set to False for full test
        'batch_size': 32,
        'use_preprocessed': True,  # Should match your training setup
        # Add other config parameters as needed
    }

    # Get configuration
    cfg = get_config(**config_kwargs)

    # Specify checkpoint path (None to use default model_best.pth)
    checkpoint_path = None  # Or specify: "/path/to/your/model_best.pth"

    # Specify classes to exclude
    exclude_classes = ['Vehicle', 'Pedestrian']

    # Specify original statistics for denormalization
    # The script will now automatically load these from saved files or config
    original_stats = None  # Will be auto-loaded from training

    # Specify label mapping (3, 4 -> 11)
    label_mapping = {3: 11, 4: 11}

    # Run enhanced test (original_stats will be auto-loaded)
    all_stats, excluded_stats = run_enhanced_test(
        cfg,
        checkpoint_path=checkpoint_path,
        exclude_classes=exclude_classes,
        original_stats=original_stats,  # None means auto-load
        label_mapping=label_mapping
    )