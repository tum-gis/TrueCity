#!/usr/bin/env python3

import numpy as np
import os
import glob

def calculate_chunk_radius_from_data(data_path):
    """
    Calculate the actual radius of chunks in an octree-split dataset
    
    Args:
        data_path: Path to octree-split dataset directory
    """
    print(f"🔍 Analyzing chunk sizes in: {data_path}")
    
    # Find all .npy files
    file_patterns = [
        os.path.join(data_path, "*.npy"),
        os.path.join(data_path, "train", "*.npy"),
        os.path.join(data_path, "test", "*.npy"),
        os.path.join(data_path, "val", "*.npy"),
    ]
    
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        print(f"❌ No .npy files found in {data_path}")
        return
    
    print(f"📄 Found {len(all_files)} chunk files")
    
    # Analyze chunk spatial properties
    chunk_radii = []
    chunk_sizes = []
    chunk_point_counts = []
    
    for i, file_path in enumerate(all_files[:10]):  # Sample first 10 files
        try:
            data = np.load(file_path)
            points = data[:, :3]  # x, y, z coordinates
            
            if len(points) == 0:
                continue
                
            # Calculate bounding box of this chunk
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            ranges = maxs - mins
            center = (mins + maxs) / 2
            
            # Calculate different radius measures
            max_side_length = np.max(ranges)
            diagonal_radius = np.linalg.norm(ranges) / 2  # Half the space diagonal
            max_distance_from_center = np.max(np.linalg.norm(points - center, axis=1))
            
            chunk_radii.append({
                'file': os.path.basename(file_path),
                'point_count': len(points),
                'max_side_length': max_side_length,
                'diagonal_radius': diagonal_radius,
                'max_distance_from_center': max_distance_from_center,
                'ranges': ranges,
                'center': center
            })
            
            chunk_point_counts.append(len(points))
            
            print(f"📦 {os.path.basename(file_path)}:")
            print(f"   Points: {len(points):,}")
            print(f"   Ranges: X={ranges[0]:.2f}, Y={ranges[1]:.2f}, Z={ranges[2]:.2f}")
            print(f"   Max side: {max_side_length:.2f}m")
            print(f"   Diagonal radius: {diagonal_radius:.2f}m")
            print(f"   Max distance from center: {max_distance_from_center:.2f}m")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    if chunk_radii:
        # Calculate statistics
        diagonal_radii = [chunk['diagonal_radius'] for chunk in chunk_radii]
        max_distances = [chunk['max_distance_from_center'] for chunk in chunk_radii]
        side_lengths = [chunk['max_side_length'] for chunk in chunk_radii]
        
        print("=" * 60)
        print("📊 CHUNK RADIUS STATISTICS")
        print("=" * 60)
        print(f"Average diagonal radius: {np.mean(diagonal_radii):.2f}m")
        print(f"Average max distance from center: {np.mean(max_distances):.2f}m")
        print(f"Average max side length: {np.mean(side_lengths):.2f}m")
        print(f"Average points per chunk: {np.mean(chunk_point_counts):,.0f}")
        print()
        print(f"Range of diagonal radii: {np.min(diagonal_radii):.2f}m - {np.max(diagonal_radii):.2f}m")
        print(f"Range of max distances: {np.min(max_distances):.2f}m - {np.max(max_distances):.2f}m")
        print(f"Range of side lengths: {np.min(side_lengths):.2f}m - {np.max(side_lengths):.2f}m")
        print()
        print("💡 Different radius definitions:")
        print("   - Diagonal radius: Half the space diagonal of the bounding box")
        print("   - Max distance from center: Actual maximum distance from chunk center to any point")
        print("   - Max side length: Length of the longest side of the bounding box")


def estimate_theoretical_radius(total_points, target_points_per_chunk=500000, estimated_point_cloud_size=100):
    """
    Estimate theoretical chunk radius based on octree parameters
    
    Args:
        total_points: Total points in original point cloud
        target_points_per_chunk: Target points per chunk
        estimated_point_cloud_size: Estimated size of the original point cloud (meters)
    """
    print("\n🧮 THEORETICAL RADIUS ESTIMATION")
    print("=" * 60)
    
    # Calculate octree depth (same logic as in split_point_cloud_grid.py)
    target_chunks = max(1, total_points // target_points_per_chunk)
    depth = max(1, int(np.log(target_chunks) / np.log(8)) + 1)
    depth = min(depth, 8)
    
    # Calculate theoretical chunk size
    initial_size = estimated_point_cloud_size * 1.1  # With padding
    chunk_side_length = initial_size / (2 ** depth)
    diagonal_radius = chunk_side_length * np.sqrt(3) / 2
    
    print(f"Estimated original point cloud size: {estimated_point_cloud_size}m")
    print(f"Initial bounding box size (with padding): {initial_size}m")
    print(f"Calculated octree depth: {depth}")
    print(f"Theoretical chunk side length: {chunk_side_length:.2f}m")
    print(f"Theoretical diagonal radius: {diagonal_radius:.2f}m")
    print(f"Number of chunks at this depth: {8**depth}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate chunk radius from octree-split dataset')
    parser.add_argument('--data_path', type=str, 
                       default='/home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_100_octree_fps',
                       help='Path to octree-split dataset')
    parser.add_argument('--total_points', type=int, default=10000000,
                       help='Total points in original point cloud (for theoretical estimation)')
    parser.add_argument('--cloud_size', type=float, default=100,
                       help='Estimated size of original point cloud in meters')
    
    args = parser.parse_args()
    
    # Analyze actual chunk radii
    calculate_chunk_radius_from_data(args.data_path)
    
    # Theoretical estimation
    estimate_theoretical_radius(
        total_points=args.total_points,
        target_points_per_chunk=500000,
        estimated_point_cloud_size=args.cloud_size
    ) 