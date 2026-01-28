#!/usr/bin/env python3

import os
import sys
import glob
import torch
import numpy as np
from tqdm import tqdm

# Add project root (TrueCity) to path for `shared.*`
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from shared.utils.fps import farthest_point_sample_torch

def precompute_octree_fps(source_path='/home/stud/nguyenti/storage/user/EARLy/data/data_0_octree',
                         output_path='/home/stud/nguyenti/storage/user/EARLy/data/data_0_octree_fps',
                         n_points=2048, sample_multiplier=0.2, gpu_batch_size=16):
    """
    Precompute FPS sampling for octree dataset structure
    
    Args:
        source_path: str - path to octree data folder (flat structure with train*.npy)
        output_path: str - path to save precomputed FPS samples
        n_points: int - number of points per sample (default: 2048)
        sample_multiplier: float - multiplier for samples (0.2=fast training, 1.0=full)
        gpu_batch_size: int - number of FPS samples to process in parallel on GPU
    """
    print("🚀 Precomputing FPS samples for Octree dataset...")
    print(f"📁 Source: {source_path}")
    print(f"📁 Output: {output_path}")
    print(f"🎯 Points per sample: {n_points}")
    print(f"🔢 Sample multiplier: {sample_multiplier}")
    print(f"🏗️ GPU batch size: {gpu_batch_size}")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Find all training files
    train_dir = os.path.join(source_path, "train")
    if os.path.exists(train_dir):
        train_files = sorted(glob.glob(os.path.join(train_dir, "*.npy")))
        print(f"📂 Found {len(train_files)} training files in train/ subdirectory")
    else:
        # Fallback to old behavior for flat directory structure
        train_files = sorted(glob.glob(os.path.join(source_path, "train*.npy")))
        print(f"📂 Found {len(train_files)} training files in root directory")
    
    if not train_files:
        raise ValueError(f"No .npy files found in {train_dir if os.path.exists(train_dir) else source_path}")
    
    total_samples_created = 0
    total_original_points = 0
    
    # Process each file
    for file_idx, file_path in enumerate(tqdm(train_files, desc="Processing files")):
        filename = os.path.basename(file_path)
        print(f"\n📄 Processing {filename}...")
        
        # Load original file
        try:
            data = np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            continue
            
        if len(data.shape) != 2 or data.shape[1] < 4:
            print(f"⚠️ Skipping {filename} - invalid shape {data.shape}")
            continue
        
        total_points = len(data)
        total_original_points += total_points
        
        # Calculate how many samples to create
        theoretical_max = total_points // n_points
        num_samples = max(1, int(sample_multiplier * theoretical_max))
        
        print(f"   📊 {total_points:,} points → max {theoretical_max} samples → creating {num_samples} samples")
        
        # Extract coordinates and labels
        points = data[:, :3].astype(np.float32)
        labels = data[:, 3]
        
        # Process in batches for GPU efficiency
        sample_idx = 0
        for batch_start in tqdm(range(0, num_samples, gpu_batch_size), 
                               desc=f"  FPS batches", leave=False):
            batch_end = min(batch_start + gpu_batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            # For each sample in the batch
            for i in range(current_batch_size):
                # Always use FPS for proper geometric sampling
                # Add some randomization for diversity between samples
                if num_samples > 1:
                    # For multiple samples, randomly subsample first to get different starting regions
                    if total_points > n_points * 3:  # Only if we have enough points for diversity
                        # Randomly sample a larger subset first, then apply FPS
                        subset_size = min(total_points, n_points * 3)
                        subset_indices = np.random.choice(total_points, subset_size, replace=False)
                        subset_points = points[subset_indices]
                        
                        # Apply FPS to the subset
                        fps_indices = farthest_point_sample_torch(subset_points, n_points, device)
                        # Map back to original indices
                        indices = subset_indices[fps_indices.cpu().numpy()]
                    else:
                        # Not enough points for subset, just use FPS directly
                        indices_torch = farthest_point_sample_torch(points, n_points, device)
                        indices = indices_torch.cpu().numpy()
                else:
                    # Single sample: use FPS directly
                    indices_torch = farthest_point_sample_torch(points, n_points, device)
                    indices = indices_torch.cpu().numpy()
                
                # Get sampled data
                sampled_points = points[indices]
                sampled_labels = labels[indices]
                
                # Combine back to original format [x, y, z, class]
                sampled_data = np.column_stack([sampled_points, sampled_labels])
                
                # Save sample
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_fps_{sample_idx:04d}.npy"
                output_file_path = os.path.join(output_path, output_filename)
                
                np.save(output_file_path, sampled_data)
                sample_idx += 1
                total_samples_created += 1
            
            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Cleanup after each file
        del data, points, labels
        import gc
        gc.collect()
        
        print(f"   ✅ Created {num_samples} FPS samples from {filename}")
    
    print(f"\n🏆 FPS Precomputation Complete!")
    print(f"📊 Statistics:")
    print(f"   📁 Processed files: {len(train_files)}")
    print(f"   📊 Original points: {total_original_points:,}")
    print(f"   🎯 FPS samples created: {total_samples_created:,}")
    print(f"   📈 Compression ratio: {total_original_points / (total_samples_created * n_points):.1f}x")
    print(f"   💾 Output directory: {output_path}")
    
    print(f"\n🚀 To use precomputed dataset:")
    print(f"   python training.py --data_path {output_path} --num_epochs 10 --batch_size 32")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Precompute FPS samples for Octree dataset')
    parser.add_argument('--source_path', type=str, 
                       default='/home/stud/nguyenti/storage/user/EARLy/data/data_0_octree',
                       help='Path to source octree dataset')
    parser.add_argument('--output_path', type=str,
                       default='/home/stud/nguyenti/storage/user/EARLy/data/data_0_octree_fps', 
                       help='Path to save FPS precomputed dataset')
    parser.add_argument('--n_points', type=int, default=2048,
                       help='Number of points per sample (default: 2048)')
    parser.add_argument('--sample_multiplier', type=float, default=0.2,
                       help='Sample multiplier (0.2=fast, 1.0=full dataset)')
    parser.add_argument('--gpu_batch_size', type=int, default=16,
                       help='GPU batch size for FPS processing')
    
    args = parser.parse_args()
    
    precompute_octree_fps(
        source_path=args.source_path,
        output_path=args.output_path,
        n_points=args.n_points,
        sample_multiplier=args.sample_multiplier,
        gpu_batch_size=args.gpu_batch_size
    )


if __name__ == "__main__":
    main() 