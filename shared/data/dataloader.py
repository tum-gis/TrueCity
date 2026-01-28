import os
import glob
import numpy as np
from torch.utils.data import DataLoader
from shared.data.dataset import IngolstadtDataset


def create_ingolstadt_dataloaders(data_path='/home/stud/nguyenti/storage/user/EARLy/data', 
                                 batch_size=32, n_points=1024, num_workers=4, sample_multiplier=1.0, is_precomputed=False):
    """
    Create dataloaders for Ingolstadt dataset with FPS sampling
    
    Args:
        data_path: str - path to Ingolstadt data folder
        batch_size: int - batch size for training
        n_points: int - number of points to sample per point cloud using FPS (ignored if precomputed)
        num_workers: int - number of workers for data loading
        sample_multiplier: float - multiplier for samples (0.1=testing, 1.0=normal, 2.5=high diversity) (ignored if precomputed)
        is_precomputed: bool - True if data is already FPS-sampled (no sampling needed)
    
    Returns:
        dict: containing 'train', 'val', 'test' dataloaders and dataset info
    """
    
    # Create datasets
    datasets = {}
    dataloaders = {}
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Check if the split directory or files exist before creating dataset
        split_path = os.path.join(data_path, split)
        split_files_pattern = os.path.join(data_path, f"{split}*.npy")
        
        if os.path.exists(split_path) or glob.glob(split_files_pattern):
            datasets[split] = IngolstadtDataset(
                data_path=data_path,
                split=split,
                n_points=n_points,
                normalize=True,
                use_fps=True,  # Enable FPS sampling
                sample_multiplier=sample_multiplier,
                is_precomputed=is_precomputed
            )
            
            shuffle = (split == 'train')  # Only shuffle training data
            
            # Adjust batch size and drop_last based on dataset size
            dataset_size = len(datasets[split])
            effective_batch_size = min(batch_size, dataset_size)  # Don't exceed dataset size
            # Always drop last batch to avoid batch_size=1 issues with BatchNorm
            drop_last = True
            
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=effective_batch_size,
                shuffle=shuffle,
                num_workers=min(num_workers, 2),  # Reduce workers for stability
                pin_memory=True,
                drop_last=drop_last
            )
            
            print(f"✅ Created {split} dataloader: {len(datasets[split])} samples, batch_size={effective_batch_size}")
        else:
            print(f"⚠️ Skipping {split} split: no data found")
            datasets[split] = None
            dataloaders[split] = None
    
    # Get dataset info from training set
    dataset_info = {}
    if datasets['train'] is not None and len(datasets['train']) > 0:
        dataset_info = {
            'num_classes': datasets['train'].num_classes,
            'class_to_idx': datasets['train'].class_to_idx,
            'idx_to_class': datasets['train'].idx_to_class,
            'n_points': n_points
        }

    else:
        raise ValueError("Training dataset is required but not found or empty!")
    
    return {
        'dataloaders': dataloaders,
        'datasets': datasets,
        'info': dataset_info
    }


def analyze_ingolstadt_dataset(data_path='/home/stud/nguyenti/storage/user/EARLy/data', 
                               split='train', max_files=None, sample_multiplier=1.0):
    """
    Comprehensive analysis of the Ingolstadt dataset
    
    Args:
        data_path: str - path to Ingolstadt data folder
        split: str - which split to analyze ('train', 'val', 'test')
        max_files: int - maximum number of files to analyze (None for all)
        sample_multiplier: float - multiplier for samples
    """
    print(f"📊 Comprehensive Ingolstadt Dataset Analysis")
    print("=" * 60)
    
    # Create dataset instance for file discovery
    temp_dataset = IngolstadtDataset(data_path=data_path, split=split, n_points=1024, sample_multiplier=sample_multiplier)
    files_to_analyze = temp_dataset.file_paths
    
    if max_files:
        files_to_analyze = files_to_analyze[:max_files]
        print(f"🔍 Analyzing {len(files_to_analyze)} files (limited from {len(temp_dataset.file_paths)} total)")
    else:
        print(f"🔍 Analyzing all {len(files_to_analyze)} files")
    
    # Initialize statistics tracking
    all_point_counts = []
    all_classes = {}
    coordinate_bounds = {'x_min': float('inf'), 'x_max': float('-inf'),
                        'y_min': float('inf'), 'y_max': float('-inf'),
                        'z_min': float('inf'), 'z_max': float('-inf')}
    total_points_analyzed = 0
    
    print(f"\n📁 File Analysis:")
    for i, file_path in enumerate(files_to_analyze):
        filename = os.path.basename(file_path)
        print(f"   [{i+1:3d}/{len(files_to_analyze)}] {filename}")
        
        # Load and analyze file
        data = temp_dataset.load_single_file(file_path)
        num_points = len(data)
        all_point_counts.append(num_points)
        total_points_analyzed += num_points
        
        # Coordinate analysis
        coords = data[:, :3].astype(float)
        coordinate_bounds['x_min'] = min(coordinate_bounds['x_min'], np.min(coords[:, 0]))
        coordinate_bounds['x_max'] = max(coordinate_bounds['x_max'], np.max(coords[:, 0]))
        coordinate_bounds['y_min'] = min(coordinate_bounds['y_min'], np.min(coords[:, 1]))
        coordinate_bounds['y_max'] = max(coordinate_bounds['y_max'], np.max(coords[:, 1]))
        coordinate_bounds['z_min'] = min(coordinate_bounds['z_min'], np.min(coords[:, 2]))
        coordinate_bounds['z_max'] = max(coordinate_bounds['z_max'], np.max(coords[:, 2]))
        
        # Class analysis
        unique_classes, counts = np.unique(data[:, 3], return_counts=True)
        for cls, count in zip(unique_classes, counts):
            all_classes[cls] = all_classes.get(cls, 0) + count
        
        print(f"       📊 {num_points:,} points, Classes: {dict(zip(unique_classes, counts))}")
    
    # Print comprehensive statistics
    print(f"\n📈 Overall Dataset Statistics:")
    print(f"   📁 Total files: {len(files_to_analyze)}")
    print(f"   📊 Total points: {total_points_analyzed:,}")
    print(f"   📊 Avg points per file: {np.mean(all_point_counts):,.0f}")
    print(f"   📊 Median points per file: {np.median(all_point_counts):,.0f}")
    print(f"   📊 Min points per file: {min(all_point_counts):,}")
    print(f"   📊 Max points per file: {max(all_point_counts):,}")
    print(f"   📊 Std points per file: {np.std(all_point_counts):,.0f}")
    
    print(f"\n🌍 Coordinate Bounds:")
    print(f"   X: [{coordinate_bounds['x_min']:.2f}, {coordinate_bounds['x_max']:.2f}] (range: {coordinate_bounds['x_max']-coordinate_bounds['x_min']:.2f})")
    print(f"   Y: [{coordinate_bounds['y_min']:.2f}, {coordinate_bounds['y_max']:.2f}] (range: {coordinate_bounds['y_max']-coordinate_bounds['y_min']:.2f})")
    print(f"   Z: [{coordinate_bounds['z_min']:.2f}, {coordinate_bounds['z_max']:.2f}] (range: {coordinate_bounds['z_max']-coordinate_bounds['z_min']:.2f})")
    
    print(f"\n🏷️ Complete Class Distribution:")
    for cls in sorted(all_classes.keys()):
        count = all_classes[cls]
        percentage = 100 * count / total_points_analyzed
        print(f"   {cls}: {count:,} points ({percentage:.1f}%)")
    
    print(f"\n🔄 Sampling Impact Analysis:")
    files_needing_fps = sum(1 for count in all_point_counts if count > 1024)
    files_needing_upsampling = sum(1 for count in all_point_counts if count < 1024)
    files_exact_fit = sum(1 for count in all_point_counts if count == 1024)
    
    print(f"   📊 Files with >1024 points (will use FPS): {files_needing_fps} ({100*files_needing_fps/len(all_point_counts):.1f}%)")
    print(f"   📊 Files with <1024 points (will upsample): {files_needing_upsampling} ({100*files_needing_upsampling/len(all_point_counts):.1f}%)")
    print(f"   📊 Files with exactly 1024 points: {files_exact_fit} ({100*files_exact_fit/len(all_point_counts):.1f}%)")
    
    # Calculate potential data loss/gain
    total_after_sampling = len(files_to_analyze) * 1024
    efficiency = total_after_sampling / total_points_analyzed
    print(f"   📊 Sampling efficiency: {efficiency:.2f} (1.0 = no loss, <1.0 = data reduction, >1.0 = upsampling)")
    
    return {
        'file_count': len(files_to_analyze),
        'total_points': total_points_analyzed,
        'point_stats': {
            'mean': np.mean(all_point_counts),
            'median': np.median(all_point_counts),
            'min': min(all_point_counts),
            'max': max(all_point_counts),
            'std': np.std(all_point_counts)
        },
        'coordinate_bounds': coordinate_bounds,
        'class_distribution': all_classes,
        'sampling_stats': {
            'fps_files': files_needing_fps,
            'upsample_files': files_needing_upsampling,
            'exact_files': files_exact_fit,
            'efficiency': efficiency
        }
    }

