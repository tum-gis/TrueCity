#!/usr/bin/env python3

import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
from pathlib import Path


class OctreeNode:
    """Simple octree node for spatial partitioning"""
    def __init__(self, center, size, points=None, labels=None, max_depth=10, current_depth=0):
        self.center = center
        self.size = size
        self.points = points if points is not None else []
        self.labels = labels if labels is not None else []
        self.children = [None] * 8
        self.is_leaf = True
        self.max_depth = max_depth
        self.current_depth = current_depth
    
    def subdivide(self):
        """Subdivide this node into 8 children"""
        if self.current_depth >= self.max_depth:
            return
            
        self.is_leaf = False
        half_size = self.size / 2
        
        # 8 octants
        offsets = [
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ]
        
        for i, offset in enumerate(offsets):
            child_center = self.center + np.array(offset) * half_size / 2
            self.children[i] = OctreeNode(
                child_center, half_size, 
                max_depth=self.max_depth, 
                current_depth=self.current_depth + 1
            )
        
        # Distribute points to children
        if len(self.points) > 0:
            points_array = np.array(self.points)
            labels_array = np.array(self.labels)
            
            for i, (point, label) in enumerate(zip(points_array, labels_array)):
                child_idx = self._get_child_index(point)
                if 0 <= child_idx < 8:
                    self.children[child_idx].points.append(point)
                    self.children[child_idx].labels.append(label)
        
        # Clear points from parent
        self.points = []
        self.labels = []
    
    def _get_child_index(self, point):
        """Get which child octant a point belongs to"""
        index = 0
        if point[0] > self.center[0]: index |= 1
        if point[1] > self.center[1]: index |= 2
        if point[2] > self.center[2]: index |= 4
        return index
    
    def insert(self, point, label, max_points_per_node):
        """Insert a point into the octree"""
        if self.is_leaf:
            self.points.append(point)
            self.labels.append(label)
            
            # Subdivide if we have too many points and haven't reached max depth
            if len(self.points) > max_points_per_node and self.current_depth < self.max_depth:
                self.subdivide()
        else:
            # Find appropriate child
            child_idx = self._get_child_index(point)
            if 0 <= child_idx < 8 and self.children[child_idx] is not None:
                self.children[child_idx].insert(point, label, max_points_per_node)
    
    def get_leaf_chunks(self, min_points=1000):
        """Get all leaf nodes with sufficient points as chunks"""
        chunks = []
        
        if self.is_leaf:
            if len(self.points) >= min_points:
                chunks.append({
                    'points': np.array(self.points),
                    'labels': np.array(self.labels),
                    'center': self.center,
                    'size': self.size
                })
        else:
            for child in self.children:
                if child is not None:
                    chunks.extend(child.get_leaf_chunks(min_points))
        
        return chunks


def calculate_optimal_octree_depth(total_points, target_points_per_chunk=500000):
    """Calculate optimal octree depth to get chunks near target size"""
    # Estimate how many chunks we want
    target_chunks = max(1, total_points // target_points_per_chunk)
    
    # Each level of octree can have up to 8^depth leaf nodes
    # Find depth where 8^depth ≈ target_chunks
    depth = max(1, int(np.log(target_chunks) / np.log(8)) + 1)
    depth = min(depth, 8)  # Cap at reasonable depth
    
    return depth


def split_point_cloud_octree(file_path, output_dir, target_points_per_chunk=500000):
    """
    Split a single point cloud file using octree-based spatial partitioning
    
    Args:
        file_path: Path to input .npy file
        output_dir: Directory to save chunks
        target_points_per_chunk: Target number of points per chunk
    
    Returns:
        List of saved chunk file paths
    """
    print(f"🔄 Processing {os.path.basename(file_path)}...")
    
    # Load data
    data = np.load(file_path)
    points = data[:, :3].astype(np.float32)  # x, y, z
    labels = data[:, 3].astype(np.float32)   # object_class
    
    total_points = len(points)
    print(f"   📊 Loaded {total_points:,} points")
    
    if total_points == 0:
        print(f"   ⚠️ Empty file, skipping")
        return []
    
    # Calculate octree parameters
    octree_depth = calculate_optimal_octree_depth(total_points, target_points_per_chunk)
    max_points_per_node = target_points_per_chunk
    min_points_per_chunk = target_points_per_chunk // 4  # Accept chunks with at least 25% of target
    
    print(f"   🌳 Using octree depth: {octree_depth}, max points per node: {max_points_per_node:,}")
    
    # Calculate bounding box
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2
    size = np.max(maxs - mins) * 1.1  # Add some padding
    
    print(f"   📦 Bounding box: center={center}, size={size:.2f}")
    
    # Build octree
    print(f"   🌳 Building octree...")
    octree = OctreeNode(center, size, max_depth=octree_depth)
    
    # Insert all points with progress bar
    for i, (point, label) in enumerate(tqdm(zip(points, labels), 
                                           total=len(points), 
                                           desc="   Inserting points", 
                                           leave=False)):
        octree.insert(point, label, max_points_per_node)
    
    # Get chunks from leaf nodes
    print(f"   📦 Extracting chunks...")
    chunks = octree.get_leaf_chunks(min_points=min_points_per_chunk)
    
    print(f"   ✅ Created {len(chunks)} chunks")
    
    # Save chunks
    saved_files = []
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    for i, chunk in enumerate(chunks):
        chunk_points = chunk['points']
        chunk_labels = chunk['labels']
        
        # Combine points and labels back into original format
        chunk_data = np.column_stack([chunk_points, chunk_labels])
        
        # Generate filename
        chunk_filename = f"{base_name}_octree_{i:03d}.npy"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        # Save chunk
        np.save(chunk_path, chunk_data)
        saved_files.append(chunk_path)
        
        print(f"   💾 Saved {chunk_filename}: {len(chunk_data):,} points")
    
    print(f"   ✅ Saved {len(saved_files)} chunks from {os.path.basename(file_path)}")
    return saved_files


def split_dataset_octree(source_path, output_path, target_points_per_chunk=500000):
    """
    Split entire dataset using octree-based spatial partitioning
    
    Args:
        source_path: Path to source dataset directory
        output_path: Path to output directory
        target_points_per_chunk: Target points per chunk
    
    Returns:
        Path to output dataset
    """
    print(f"🌳 Starting octree-based dataset splitting...")
    print(f"📁 Source: {source_path}")
    print(f"📁 Output: {output_path}")
    print(f"🎯 Target points per chunk: {target_points_per_chunk:,}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Find all .npy files
    file_patterns = [
        os.path.join(source_path, "*.npy"),
        os.path.join(source_path, "train", "*.npy"),
        os.path.join(source_path, "test", "*.npy"),
        os.path.join(source_path, "val", "*.npy"),
    ]
    
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        print(f"❌ No .npy files found in {source_path}")
        return None
    
    print(f"📄 Found {len(all_files)} files to process")
    
    # Process each file
    total_chunks = 0
    total_original_points = 0
    total_output_points = 0
    
    for file_path in all_files:
        print(f"\n📄 Processing {os.path.basename(file_path)}")
        
        # Determine output subdirectory based on file naming
        base_name = os.path.basename(file_path)
        if base_name.startswith('train'):
            output_subdir = os.path.join(output_path, 'train')
        elif base_name.startswith('test'):
            output_subdir = os.path.join(output_path, 'test')
        elif base_name.startswith('val'):
            output_subdir = os.path.join(output_path, 'val')
        else:
            output_subdir = output_path
        
        os.makedirs(output_subdir, exist_ok=True)
        
        # Count original points
        try:
            original_data = np.load(file_path)
            original_points = len(original_data)
            total_original_points += original_points
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
            continue
        
        # Split file
        try:
            saved_files = split_point_cloud_octree(
                file_path, 
                output_subdir, 
                target_points_per_chunk
            )
            
            # Count output points
            file_output_points = 0
            for saved_file in saved_files:
                chunk_data = np.load(saved_file)
                file_output_points += len(chunk_data)
            
            total_output_points += file_output_points
            total_chunks += len(saved_files)
            
            print(f"   📊 {original_points:,} → {file_output_points:,} points in {len(saved_files)} chunks")
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")
            continue
    
    # Summary
    print(f"\n🎉 Octree splitting completed!")
    print(f"📊 Total files processed: {len(all_files)}")
    print(f"📊 Total chunks created: {total_chunks}")
    print(f"📊 Original points: {total_original_points:,}")
    print(f"📊 Output points: {total_output_points:,}")
    print(f"📊 Average points per chunk: {total_output_points // max(1, total_chunks):,}")
    
    # Check if we achieved target
    if total_chunks > 0:
        avg_points = total_output_points / total_chunks
        print(f"🎯 Target: {target_points_per_chunk:,}, Achieved: {avg_points:,.0f}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Split large point cloud files using octree-based spatial partitioning')
    parser.add_argument('--source_path', type=str, 
                       default='/home/stud/nguyenti/storage/user/EARLy/data/data_0',
                       help='Path to source dataset')
    parser.add_argument('--output_path', type=str, 
                       default='/home/stud/nguyenti/storage/user/EARLy/data/data_0_octree',
                       help='Path to output octree dataset')
    parser.add_argument('--target_points', type=int, default=500000,
                       help='Target points per octree chunk (default: 500000)')
    
    args = parser.parse_args()
    
    # Run octree splitting
    result_path = split_dataset_octree(
        source_path=args.source_path,
        output_path=args.output_path,
        target_points_per_chunk=args.target_points
    )
    
    if result_path:
        print(f"\n🚀 Usage after octree splitting:")
        print(f"python training.py --mode train --data_path {result_path} --n_points 2048")
        print(f"\n💡 Or precompute FPS (optional, since octree chunks are small):")
        print(f"python -c \"import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared')); from shared.data.dataset import precompute_fps_dataset; precompute_fps_dataset(source_path='{result_path}', output_path='{result_path}_fps', n_points=2048, sample_multiplier=1.0)\"")


if __name__ == "__main__":
    main() 