"""
Standalone script to compute and save normalization statistics
Run this before training to ensure consistent normalization across all experiments.
"""

import os
import numpy as np
import json
import pickle
from pathlib import Path

def compute_global_normalization_stats(data_root, real_ratio, use_preprocessed=True):
    """
    Compute global normalization statistics from training data
    """
    print("="*60)
    print("COMPUTING GLOBAL NORMALIZATION STATISTICS")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Real ratio: {real_ratio}")
    print(f"Use preprocessed: {use_preprocessed}")
    
    all_xyz = []
    
    if use_preprocessed:
        # Use preprocessed data from octree_fps folder
        train_folder = os.path.join(data_root, 'train', f"datav2_{real_ratio}_octree_fps")
        print(f"Loading from preprocessed folder: {train_folder}")
        
        if not os.path.exists(train_folder):
            raise ValueError(f"Preprocessed training folder not found: {train_folder}")
        
        # Load all .npy files in the folder
        npy_files = [f for f in os.listdir(train_folder) if f.endswith('.npy')]
        if not npy_files:
            raise ValueError(f"No .npy files found in {train_folder}")
        
        print(f"Found {len(npy_files)} preprocessed files")
        
        for i, npy_file in enumerate(sorted(npy_files)):
            file_path = os.path.join(train_folder, npy_file)
            print(f"Loading {npy_file}... ({i+1}/{len(npy_files)})")
            data = np.load(file_path)  # Shape: [2048, 4] -> [x, y, z, label]
            print(f"  File shape: {data.shape}")
            xyz = data[:, :3]  # X, Y, Z columns
            all_xyz.append(xyz)
            print(f"  Loaded {len(xyz):,} points")
            
            # Progress indicator for large datasets
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(npy_files)} files processed")
    
    else:
        # Use original data files
        train_files = [f"train1_{real_ratio}.npy", f"train2_{real_ratio}.npy"]
        print(f"Loading from original files: {train_files}")
        
        for f in train_files:
            file_path = os.path.join(data_root, 'train', f)
            if os.path.exists(file_path):
                print(f"Loading {file_path}...")
                data = np.load(file_path)
                print(f"  File shape: {data.shape}")
                xyz = data[:, :3]  # X, Y, Z columns
                all_xyz.append(xyz)
                print(f"  Loaded {len(xyz):,} points")
            else:
                print(f"Warning: {file_path} not found!")
    
    if not all_xyz:
        raise ValueError("No training data found for computing normalization stats")
    
    # Combine all training data
    all_xyz = np.vstack(all_xyz)
    print(f"\nTotal training points: {len(all_xyz):,}")
    print(f"Coordinate ranges:")
    print(f"  X: {all_xyz[:, 0].min():.3f} to {all_xyz[:, 0].max():.3f}")
    print(f"  Y: {all_xyz[:, 1].min():.3f} to {all_xyz[:, 1].max():.3f}")
    print(f"  Z: {all_xyz[:, 2].min():.3f} to {all_xyz[:, 2].max():.3f}")
    
    # Compute global statistics
    global_mean = np.mean(all_xyz, axis=0)
    print(f"\nGlobal mean: [{global_mean[0]:.6f}, {global_mean[1]:.6f}, {global_mean[2]:.6f}]")
    
    centered_xyz = all_xyz - global_mean
    norms = np.linalg.norm(centered_xyz, axis=1)
    global_max_norm = np.max(norms)
    print(f"Global max norm: {global_max_norm:.6f}")
    
    # Additional statistics for verification
    print(f"\nAdditional statistics:")
    print(f"  Mean norm: {np.mean(norms):.6f}")
    print(f"  Std norm: {np.std(norms):.6f}")
    print(f"  95th percentile norm: {np.percentile(norms, 95):.6f}")
    
    stats = {
        'mean': global_mean,
        'max_norm': global_max_norm,
        'data_info': {
            'total_points': int(len(all_xyz)),
            'real_ratio': real_ratio,
            'use_preprocessed': use_preprocessed,
            'data_source': f"datav2_{real_ratio}_octree_fps" if use_preprocessed else f"train1_{real_ratio}.npy, train2_{real_ratio}.npy",
            'coordinate_ranges': {
                'x_min': float(all_xyz[:, 0].min()),
                'x_max': float(all_xyz[:, 0].max()),
                'y_min': float(all_xyz[:, 1].min()),
                'y_max': float(all_xyz[:, 1].max()),
                'z_min': float(all_xyz[:, 2].min()),
                'z_max': float(all_xyz[:, 2].max())
            },
            'norm_statistics': {
                'mean_norm': float(np.mean(norms)),
                'std_norm': float(np.std(norms)),
                'percentile_95': float(np.percentile(norms, 95))
            }
        }
    }
    
    return stats


def save_normalization_stats(stats, save_path):
    """Save normalization statistics to file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    pkl_path = save_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(stats, f)
    
    # Save as JSON (for human readability)
    stats_json = {
        'mean': stats['mean'].tolist(),
        'max_norm': float(stats['max_norm']),
        'data_info': stats['data_info']
    }
    
    json_path = save_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(stats_json, f, indent=2)
    
    print("="*60)
    print("NORMALIZATION STATISTICS SAVED")
    print("="*60)
    print(f"JSON file: {json_path}")
    print(f"Pickle file: {pkl_path}")
    print("="*60)


def load_and_verify_stats(stats_path):
    """Load and verify saved statistics"""
    print("="*60)
    print("VERIFYING SAVED STATISTICS")
    print("="*60)
    
    if stats_path.endswith('.json'):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        mean = np.array(stats['mean'])
        max_norm = stats['max_norm']
    else:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        mean = stats['mean']
        max_norm = stats['max_norm']
    
    print(f"Loaded from: {stats_path}")
    print(f"Mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"Max norm: {max_norm:.6f}")
    
    if 'data_info' in stats:
        info = stats['data_info']
        print(f"Total points used: {info['total_points']:,}")
        print(f"Real ratio: {info['real_ratio']}")
        
        if 'coordinate_ranges' in info:
            ranges = info['coordinate_ranges']
            print(f"Original coordinate ranges:")
            print(f"  X: {ranges['x_min']:.3f} to {ranges['x_max']:.3f}")
            print(f"  Y: {ranges['y_min']:.3f} to {ranges['y_max']:.3f}")
            print(f"  Z: {ranges['z_min']:.3f} to {ranges['z_max']:.3f}")
    
    print("="*60)
    return stats


def main():
    """Main function to compute normalization statistics"""
    
    # Configuration - MODIFY THESE PATHS
    data_root = '../../dataset/datav2_final'  # Base data directory
    real_ratio = 75  # Your model's real_ratio
    model_name = 'point_transformer_v1'
    use_preprocessed = True  # Set to True for octree_fps data, False for original train1_75.npy files
    
    # Output directory
    save_dir = f"experiments/{model_name}_logs/model_{real_ratio}"
    stats_path = os.path.join(save_dir, "normalization_stats.json")
    
    print("NORMALIZATION STATISTICS COMPUTATION")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Real ratio: {real_ratio}")
    print(f"Use preprocessed: {use_preprocessed}")
    print(f"Output path: {stats_path}")
    print("="*60)
    
    try:
        # Check if data exists
        if use_preprocessed:
            train_folder = os.path.join(data_root, 'train', f"datav2_{real_ratio}_octree_fps")
            if not os.path.exists(train_folder):
                print(f"ERROR: Preprocessed training folder not found: {train_folder}")
                print("Please check your data_root and real_ratio settings.")
                return False
            
            npy_files = [f for f in os.listdir(train_folder) if f.endswith('.npy')]
            if not npy_files:
                print(f"ERROR: No .npy files found in {train_folder}")
                return False
            
            print(f"Found {len(npy_files)} preprocessed files in {train_folder}")
        else:
            # Check original files
            train_files = [f"train1_{real_ratio}.npy", f"train2_{real_ratio}.npy"]
            for f in train_files:
                file_path = os.path.join(data_root, 'train', f)
                if not os.path.exists(file_path):
                    print(f"ERROR: Training file not found: {file_path}")
                    print("Please check your data_root and real_ratio settings.")
                    return False
        
        # Compute statistics
        stats = compute_global_normalization_stats(data_root, real_ratio, use_preprocessed)
        
        # Save statistics
        save_normalization_stats(stats, stats_path)
        
        # Verify by loading back
        load_and_verify_stats(stats_path)
        
        print("\n" + "="*60)
        print("SUCCESS: Normalization statistics computed and saved!")
        print("You can now run training and testing with consistent normalization.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("\nFailed to compute normalization statistics.")
        print("Please check your configuration and try again.")
    else:
        print("\nNormalization statistics computation completed successfully!")


# Example usage for different configurations:
def compute_for_multiple_ratios():
    """Compute stats for multiple real_ratio values"""
    data_root = '../../dataset/datav2_final'
    model_name = 'point_transformer_v1'
    ratios = [0, 25, 50, 75, 100]
    
    for ratio in ratios:
        print(f"\n{'='*60}")
        print(f"Computing stats for real_ratio = {ratio}")
        print(f"{'='*60}")
        
        save_dir = f"experiments/{model_name}_logs/model_{ratio}"
        stats_path = os.path.join(save_dir, "normalization_stats.json")
        
        try:
            stats = compute_global_normalization_stats(data_root, ratio)
            save_normalization_stats(stats, stats_path)
            print(f"✓ Success for ratio {ratio}")
        except Exception as e:
            print(f"✗ Failed for ratio {ratio}: {e}")


# Uncomment to run for multiple ratios:
if __name__ == "__main__":
    compute_for_multiple_ratios()