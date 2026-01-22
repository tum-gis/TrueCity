"""
IngolstadtDataset class for point cloud semantic segmentation
Handles loading and preprocessing of point cloud data with FPS sampling
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
import glob
import pandas as pd
from ..utils.fps import farthest_point_sample_numpy


class IngolstadtDataset(Dataset):
    """
    Dataset class for Ingolstadt point cloud data
    Expected data format: NPY files with columns [x, y, z, object_class]
    """
    
    def __init__(self, data_path='/home/stud/nguyenti/storage/user/EARLy/data', split='train', 
                 n_points=1024, normalize=True, use_fps=True, sample_multiplier=1.0, is_precomputed=False,
                 allowed_classes=None):
        """
        Args:
            data_path: str - path to EARLy data folder
            split: str - 'train', 'test', or 'val'
            n_points: int - number of points to sample per point cloud using FPS
            normalize: bool - whether to normalize point coordinates
            use_fps: bool - whether to use FPS sampling (vs random)
            sample_multiplier: float - multiplier for samples (0.1=testing, 1.0=normal, 2.5=high diversity)
            is_precomputed: bool - True if data is already FPS-sampled (no sampling needed)
            allowed_classes: list - if provided, only these classes will be used (whitelist). 
                                  If None, uses default exclusion of classes 3, 4, 13
        """
        self.data_path = data_path
        self.split = split
        self.n_points = n_points
        self.normalize = normalize
        self.use_fps = use_fps
        self.sample_multiplier = sample_multiplier
        self.is_precomputed = is_precomputed
        self.allowed_classes = allowed_classes
        
        # Load file paths
        self.file_paths = self.load_file_paths()
        
        # Calculate samples per file based on total_points / sample_size
        if self.is_precomputed:
            # For precomputed data, each file is already one sample
            print(f"🚀 Precomputed dataset detected: using 1 sample per file")
            self.file_samples = [1] * len(self.file_paths)
        else:
            self.file_samples = self.calculate_samples_per_file()
        
        # Load class mapping by examining the data
        print(f"🏷️ Loading class mapping...")
        self.class_to_idx, self.idx_to_class = self.load_class_mapping()
        self.num_classes = len(self.class_to_idx)
        
        total_samples = sum(self.file_samples)
        print(f"✅ Dataset initialization complete: {total_samples:,} samples, {self.num_classes} classes")
    
    def load_file_paths(self):
        """Load all data file paths for the given split"""
        print(f"🔍 Starting file discovery for split '{self.split}'...")
        
        # Check different possible directory structures
        possible_paths = [
            os.path.join(self.data_path, self.split),  # /data/train/
            os.path.join(self.data_path),              # /data/ (all files together)
        ]
        
        file_paths = []
        
        for base_path in possible_paths:
            if os.path.exists(base_path):
                print(f"📁 Checking directory: {base_path}")
                
                # Look for files with the split name pattern
                patterns = [
                    os.path.join(base_path, f"{self.split}*.npy"),    # train*.npy
                    os.path.join(base_path, f"{self.split}_*.npy"),   # train_*.npy  
                    os.path.join(base_path, f"*{self.split}*.npy"),   # *train*.npy
                    os.path.join(base_path, "*.npy"),                 # all .npy files
                ]
                
                for pattern in patterns:
                    print(f"🔎 Searching pattern: {os.path.basename(pattern)}")
                    
                    # Use os.scandir for better performance with progress tracking
                    if pattern.endswith("*.npy"):
                        try:
                            # For large directories, use scandir with progress tracking
                            found_files = []
                            count = 0
                            
                            # Get pattern matching function
                            if f"{self.split}*.npy" in pattern:
                                filter_func = lambda name: name.startswith(self.split) and name.endswith('.npy')
                            elif f"{self.split}_*.npy" in pattern:
                                filter_func = lambda name: name.startswith(f"{self.split}_") and name.endswith('.npy')
                            elif f"*{self.split}*.npy" in pattern:
                                filter_func = lambda name: self.split in name and name.endswith('.npy')
                            else:
                                filter_func = lambda name: name.endswith('.npy')
                            
                            print(f"📊 Scanning directory with {os.path.basename(pattern)} pattern...")
                            
                            with os.scandir(base_path) as entries:
                                for entry in entries:
                                    if entry.is_file() and filter_func(entry.name):
                                        found_files.append(entry.path)
                                        count += 1
                                        
                                        # Progress logging every 5000 files
                                        if count % 5000 == 0:
                                            print(f"📈 Progress: {count:,} files found so far...")
                            
                            if found_files:
                                print(f"✅ Found {len(found_files):,} files with pattern {os.path.basename(pattern)}")
                                file_paths.extend(found_files)
                                break  # Use first successful pattern
                                
                        except Exception as e:
                            print(f"⚠️ Scandir failed, falling back to glob: {e}")
                            # Fallback to original glob method
                            found_files = glob.glob(pattern)
                            if found_files:
                                print(f"✅ Found {len(found_files):,} files with glob")
                                file_paths.extend(found_files)
                                break
                    else:
                        # Direct pattern, use glob
                        found_files = glob.glob(pattern)
                        if found_files:
                            print(f"✅ Found {len(found_files):,} files with pattern {os.path.basename(pattern)}")
                            file_paths.extend(found_files)
                            break  # Use first successful pattern
                
                if file_paths:
                    break  # Use first successful directory
        
        if len(file_paths) == 0:
            raise ValueError(f"No data files found for split '{self.split}'")
        
        print(f"🔄 Sorting {len(file_paths):,} file paths...")
        sorted_paths = sorted(list(set(file_paths)))  # Remove duplicates and sort
        print(f"✅ File discovery complete: {len(sorted_paths):,} unique files")
        
        return sorted_paths
    
    def calculate_samples_per_file(self):
        """Calculate how many samples each file can generate based on total_points/n_points"""
        print(f"🧮 Calculating samples per file for {len(self.file_paths):,} files...")
        
        file_samples = []
        total_files = len(self.file_paths)
        
        for i, file_path in enumerate(self.file_paths):
            # Progress logging every 5000 files
            if (i + 1) % 5000 == 0:
                print(f"📈 Processing file {i+1:,}/{total_files:,} ({100*(i+1)/total_files:.1f}%)")
            
            # Load file to get point count
            data = self.load_single_file(file_path)
            total_points = len(data)
            
            # Calculate theoretical max samples: total_points / n_points
            theoretical_max = total_points // self.n_points
            
            # Apply sample multiplier (can generate overlapping samples)
            samples_from_file = max(1, int(self.sample_multiplier * theoretical_max))
            file_samples.append(samples_from_file)
        
        print(f"✅ Sample calculation complete: {sum(file_samples):,} total samples from {len(file_samples):,} files")
        
        return file_samples
    
    def load_class_mapping(self):
        """Create mapping between class names and indices with custom label mapping"""
        all_classes = set()
        total_points = 0
        point_counts = []
        class_counts = {}
        
        # Sample files to get all class labels and statistics
        if len(self.file_paths) <= 100:
            sample_files = self.file_paths
        else:
            step = max(1, len(self.file_paths) // 100)
            sample_files = self.file_paths[::step]
        
        for file_path in sample_files:
            data = self.load_single_file(file_path)
            
            if len(data) > 0 and data.shape[1] >= 4:
                num_points = len(data)
                point_counts.append(num_points)
                total_points += num_points
                
                unique_classes, counts = np.unique(data[:, 3], return_counts=True)
                all_classes.update(unique_classes)
                
                for cls, count in zip(unique_classes, counts):
                    class_counts[cls] = class_counts.get(cls, 0) + count
        
        if len(all_classes) == 0:
            raise ValueError("Could not find any object classes in the data")
        
        # Use allowed_classes whitelist if provided, otherwise use exclusion filter
        if self.allowed_classes is not None:
            # Use whitelist approach - only keep classes in allowed_classes
            allowed_set = set(self.allowed_classes)
            filtered_classes = [cls for cls in all_classes if cls in allowed_set]
            print(f"📋 Using class whitelist: {sorted(self.allowed_classes)}")
        else:
            # Filter out only class 13 from the mapping (13 only exists in training set)
            # All other classes 0-11 are included
            excluded_classes = {13.0}
            filtered_classes = [cls for cls in all_classes if cls not in excluded_classes]
            print(f"🚫 Excluding classes: {sorted(excluded_classes)}")
        
        if len(filtered_classes) == 0:
            if self.allowed_classes is not None:
                raise ValueError(f"No valid classes found matching allowed_classes: {self.allowed_classes}")
            else:
                raise ValueError("No valid classes remaining after filtering")
        
        # Simple mapping: just use contiguous indices [0, 1, 2, ...]
        # This avoids the CUDA assertion error by ensuring all indices are in valid range
        sorted_classes = sorted(filtered_classes)
        class_to_idx = {cls: i for i, cls in enumerate(sorted_classes)}
        idx_to_class = {i: cls for cls, i in class_to_idx.items()}
        
        # Remove the custom mapping that was causing issues
        # Custom label mapping can be applied later if needed
        
        print(f"📊 Class mapping created:")
        print(f"   Original labels found: {sorted(all_classes)}")
        if self.allowed_classes is not None:
            missing = set(self.allowed_classes) - all_classes
            if missing:
                print(f"   ⚠️ Warning: Some allowed classes not found in data: {sorted(missing)}")
        else:
            excluded_classes = {13.0}
            print(f"   Excluded classes: {sorted(excluded_classes)}")
        print(f"   Valid classes: {sorted(filtered_classes)}")
        print(f"   Final class_to_idx: {class_to_idx}")
        print(f"   Number of classes: {len(class_to_idx)}")
        
        return class_to_idx, idx_to_class
    
    def load_single_file(self, file_path):
        """Load a single data file"""
        if file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.values
        elif file_path.endswith('.txt'):
            data = np.loadtxt(file_path, dtype=str)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Ensure we have the right shape
        if len(data.shape) != 2:
            raise ValueError(f"Data must be 2D array, got shape {data.shape}")
        
        # Ensure we have at least 4 columns [x, y, z, class]
        if data.shape[1] < 4:
            raise ValueError(f"Data must have at least 4 columns [x,y,z,class], got {data.shape[1]}")
        
        return data
    
    def normalize_points(self, points):
        """Normalize point coordinates to unit sphere"""
        if len(points) == 0:
            return points
            
        # Center points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        distances = np.linalg.norm(points, axis=1)
        max_dist = np.max(distances)
        if max_dist > 0:
            points = points / max_dist
            
        return points
    
    def sample_points(self, data, sample_idx=0):
        """Sample points efficiently: generate diverse samples by using different sampling strategies"""
        if len(data) == 0:
            raise ValueError("Empty data provided")
            
        points = data[:, :3].astype(np.float32)  # x, y, z coordinates
        labels = data[:, 3]                      # object class
        
        # Set different random seed for each sample to ensure diversity
        np.random.seed(42 + sample_idx)
        
        if len(data) <= self.n_points:
            # Small files: use all points with upsampling if needed
            indices = np.arange(len(data))
            if len(data) < self.n_points:
                additional_indices = np.random.choice(len(data), self.n_points - len(data), replace=True)
                indices = np.concatenate([indices, additional_indices])
            final_points = points[indices]
            final_labels = labels[indices]
            
        elif len(data) <= self.n_points * 2:
            # Medium files: apply FPS/random directly
            if self.use_fps:
                indices = farthest_point_sample_numpy(points, self.n_points)
            else:
                indices = np.random.choice(len(points), self.n_points, replace=False)
            final_points = points[indices]
            final_labels = labels[indices]
            
        else:
            # Large files: smart sampling to maximize diversity
            total_possible_samples = len(data) // self.n_points
            
            if sample_idx < total_possible_samples:
                # Non-overlapping sampling: each sample gets its own region
                start_idx = sample_idx * self.n_points
                end_idx = start_idx + self.n_points
                indices = np.arange(start_idx, min(end_idx, len(data)))
                
                # If we don't have enough points, fill with random
                if len(indices) < self.n_points:
                    additional_needed = self.n_points - len(indices)
                    additional_indices = np.random.choice(len(data), additional_needed, replace=False)
                    indices = np.concatenate([indices, additional_indices])
                    
            else:
                # Overlapping sampling: use random/FPS sampling with different seeds
                if self.use_fps:
                    # Apply FPS with different initialization
                    indices = farthest_point_sample_numpy(points, self.n_points)
                else:
                    # Random sampling
                    indices = np.random.choice(len(points), self.n_points, replace=False)
            
            final_points = points[indices]
            final_labels = labels[indices]
        
        # Clean up
        del points, labels
        import gc
        gc.collect()
        
        return final_points, final_labels
    
    def __len__(self):
        return sum(self.file_samples)
    
    def __getitem__(self, idx):
        """Get a single point cloud sample"""
        # Map idx to file and sample within file (with dynamic samples per file)
        current_idx = idx
        file_idx = 0
        
        # Find which file this index belongs to
        for i, num_samples in enumerate(self.file_samples):
            if current_idx < num_samples:
                file_idx = i
                sample_idx = current_idx
                break
            current_idx -= num_samples
        else:
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} samples")
        
        file_path = self.file_paths[file_idx]
        
        # Load data
        data = self.load_single_file(file_path)
        
        if self.is_precomputed:
            # For precomputed data, files already contain exactly n_points with FPS
            points = data[:, :3].astype(np.float32)  # x, y, z coordinates
            labels = data[:, 3]                      # object class
            
            # Ensure we have the right number of points
            if len(points) < self.n_points:
                # Upsample if we have fewer points
                indices = np.random.choice(len(points), self.n_points, replace=True)
                points = points[indices]
                labels = labels[indices]
            elif len(points) > self.n_points:
                # Downsample if we have more points
                indices = np.random.choice(len(points), self.n_points, replace=False)
                points = points[indices]
                labels = labels[indices]
        else:
            # For raw data, use different random seeds for multiple samples per file
            # This ensures different FPS results for each sample
            np.random.seed(42 + idx)  # Deterministic but different per sample
            
            # Sample points using chunk-based FPS
            points, labels = self.sample_points(data, sample_idx)
            
            # Clean up original data to free memory
            del data
            import gc
            gc.collect()
        
        # Normalize coordinates if requested
        if self.normalize:
            points = self.normalize_points(points)
        
        # Filter classes before converting to indices
        if self.allowed_classes is not None:
            # Use whitelist approach
            allowed_set = set(self.allowed_classes)
            mask = np.isin(labels, list(allowed_set))
        else:
            # Filter out only class 13 (default behavior)
            # All other classes 0-11 are included
            excluded_classes = {13.0}
            mask = ~np.isin(labels, list(excluded_classes))
        
        if not np.any(mask):
            # No valid points after filtering - create a fallback
            # Use the center point with the most common valid class
            center_point = np.mean(points, axis=0, keepdims=True)  # [1, 3]
            # Find the most common valid class from the class_to_idx
            most_common_class = list(self.class_to_idx.keys())[0]  # Use first valid class
            fallback_label = np.array([most_common_class])
            
            # Duplicate this point to fill n_points
            points = np.tile(center_point, (self.n_points, 1))
            labels = np.tile(fallback_label, self.n_points)
        else:
            # Filter to valid points
            points = points[mask]
            labels = labels[mask]
        
        # Convert class labels to indices (convert numpy types to Python floats for lookup)
        try:
            class_indices = np.array([self.class_to_idx[float(label)] for label in labels])
        except KeyError as e:
            # Handle missing labels by filtering to known classes only
            mask = np.array([float(label) in self.class_to_idx for label in labels])
            if not np.any(mask):
                raise ValueError(f"No valid class labels found in sample. Available classes: {list(self.class_to_idx.keys())}")
            points = points[mask]
            labels = labels[mask]
            class_indices = np.array([self.class_to_idx[float(label)] for label in labels])
            
            # Point count adjustment will be handled by the final safety check below
        
        # For object classification, we need a single label per point cloud
        # Take the most frequent class as the object class
        unique_classes, counts = np.unique(class_indices, return_counts=True)
        object_class = unique_classes[np.argmax(counts)]
        
        # Final safety check: ensure exactly n_points
        if len(points) != self.n_points:
            if len(points) < self.n_points:
                # Upsample if we have fewer points
                indices = np.random.choice(len(points), self.n_points, replace=True)
                points = points[indices]
                class_indices = class_indices[indices]
            elif len(points) > self.n_points:
                # Downsample if we have more points
                indices = np.random.choice(len(points), self.n_points, replace=False)
                points = points[indices]
                class_indices = class_indices[indices]
        
        # Convert to torch tensors
        points_tensor = torch.from_numpy(points.astype(np.float32))
        point_labels_tensor = torch.from_numpy(class_indices.astype(np.int64))
        object_class_tensor = torch.tensor(object_class, dtype=torch.long)
        
        # Cleanup numpy arrays (let Python's garbage collector handle this automatically)
        del points, labels, class_indices, unique_classes, counts
        
        return {
            'points': points_tensor,              # [n_points, 3]
            'point_labels': point_labels_tensor,  # [n_points] - individual point labels
            'object_class': object_class_tensor,  # scalar - dominant class for this point cloud
            'file_path': file_path
        }



