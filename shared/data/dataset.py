import numpy as np
import os
import torch
from torch.utils.data import Dataset
import glob
import pandas as pd
from shared.utils.fps import farthest_point_sample_numpy


class IngolstadtDataset(Dataset):
    """
    Dataset class for Ingolstadt point cloud data
    Expected data format: NPY files with columns [x, y, z, object_class]
    Data path: /home/stud/nguyenti/storage/user/EARLy/data/
    """
    
    def __init__(self, data_path='/home/stud/nguyenti/storage/user/EARLy/data', split='train', 
                 n_points=1024, normalize=True, use_fps=True, sample_multiplier=1.0, is_precomputed=False):
        """
        Args:
            data_path: str - path to EARLy data folder
            split: str - 'train', 'test', or 'val'
            n_points: int - number of points to sample per point cloud using FPS
            normalize: bool - whether to normalize point coordinates
            use_fps: bool - whether to use FPS sampling (vs random)
            sample_multiplier: float - multiplier for samples (0.1=testing, 1.0=normal, 2.5=high diversity)
            is_precomputed: bool - True if data is already FPS-sampled (no sampling needed)
        """
        self.data_path = data_path
        self.split = split
        self.n_points = n_points
        self.normalize = normalize
        self.use_fps = use_fps
        self.sample_multiplier = sample_multiplier
        self.is_precomputed = is_precomputed
        
        # Load file paths
        self.file_paths = self.load_file_paths()
        
        # Calculate samples per file based on total_points / sample_size
        if self.is_precomputed:
            # For precomputed data, each file is already one sample
            self.file_samples = [1] * len(self.file_paths)
        else:
            self.file_samples = self.calculate_samples_per_file()
        
        # Load class mapping by examining the data
        self.class_to_idx, self.idx_to_class = self.load_class_mapping()
        self.num_classes = len(self.class_to_idx)
        
        total_samples = sum(self.file_samples)
    
    def load_file_paths(self):
        """Load all data file paths for the given split"""
        # Check different possible directory structures
        possible_paths = [
            os.path.join(self.data_path, self.split),  # /data/train/
            os.path.join(self.data_path),              # /data/ (all files together)
        ]
        
        file_paths = []
        
        for base_path in possible_paths:
            if os.path.exists(base_path):
                # Look for files with the split name pattern
                patterns = [
                    os.path.join(base_path, f"{self.split}*.npy"),    # train*.npy
                    os.path.join(base_path, f"{self.split}_*.npy"),   # train_*.npy  
                    os.path.join(base_path, f"*{self.split}*.npy"),   # *train*.npy
                    os.path.join(base_path, "*.npy"),                 # all .npy files
                ]
                
                for pattern in patterns:
                    found_files = glob.glob(pattern)
                    if found_files:
                        file_paths.extend(found_files)
                        break  # Use first successful pattern
                
                if file_paths:
                    break  # Use first successful directory
        
        if len(file_paths) == 0:
            raise ValueError(f"No data files found for split '{self.split}'")
        
        return sorted(list(set(file_paths)))  # Remove duplicates and sort
    
    def calculate_samples_per_file(self):
        """Calculate how many samples each file can generate based on total_points/n_points"""
        file_samples = []
        
        for file_path in self.file_paths:
            # Load file to get point count
            data = self.load_single_file(file_path)
            total_points = len(data)
            
            # Calculate theoretical max samples: total_points / n_points
            theoretical_max = total_points // self.n_points
            
            # Apply sample multiplier (can generate overlapping samples)
            samples_from_file = max(1, int(self.sample_multiplier * theoretical_max))
            file_samples.append(samples_from_file)
            
        
        return file_samples
    
    def load_class_mapping(self):
        """Create mapping between class names and indices"""
        all_classes = set()
        total_points = 0
        point_counts = []
        class_counts = {}
        
        # Use common classes only (present in train/test, ignore val-specific classes)
        common_classes = {0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0}
        
        # Sample files to get all class labels and statistics (scan all files if reasonable size)
        if len(self.file_paths) <= 100:
            sample_files = self.file_paths  # Scan all files for small datasets
        else:
            # For large datasets, sample throughout the file list to catch all classes
            step = max(1, len(self.file_paths) // 100)
            sample_files = self.file_paths[::step]
        
        for file_path in sample_files:
            data = self.load_single_file(file_path)
            
            if len(data) > 0 and data.shape[1] >= 4:
                # Point statistics
                num_points = len(data)
                point_counts.append(num_points)
                total_points += num_points
                
                # Class analysis - filter to common classes for consistency
                unique_classes, counts = np.unique(data[:, 3], return_counts=True)
                
                # Filter to common classes if this is validation data with extra classes
                if self.split == 'val':
                    # Only keep common classes
                    mask = np.isin(unique_classes, list(common_classes))
                    unique_classes = unique_classes[mask]
                    counts = counts[mask]
                
                all_classes.update(unique_classes)
                
                # Accumulate class counts
                for cls, count in zip(unique_classes, counts):
                    class_counts[cls] = class_counts.get(cls, 0) + count
        
        if len(all_classes) == 0:
            raise ValueError("Could not find any object classes in the data")
        
        # Use only common classes for consistent mapping across splits
        if hasattr(self, 'split') and self.split in ['train', 'test', 'val']:
            common_classes = {0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0}
            # Only use classes that are actually present AND in common set
            class_list = sorted(list(all_classes.intersection(common_classes)))
        else:
            class_list = sorted(list(all_classes))
            
        class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        
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
            
            # Filter validation data to common classes only
            if self.split == 'val':
                common_classes = {0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0}
                mask = np.isin(labels, list(common_classes))
                points = points[mask]
                labels = labels[mask]
                
                # Resample to target number of points after filtering
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
        
        # Convert class labels to indices (convert numpy types to Python floats for lookup)
        try:
            class_indices = np.array([self.class_to_idx[float(label)] for label in labels])
        except KeyError as e:
            # Handle missing labels by filtering to known classes only
            # Filter to only known classes (silently ignore expected labels)
            mask = np.array([float(label) in self.class_to_idx for label in labels])
            if not np.any(mask):
                raise ValueError(f"No valid class labels found in sample. Available classes: {list(self.class_to_idx.keys())}")
            points = points[mask]
            labels = labels[mask]
            class_indices = np.array([self.class_to_idx[float(label)] for label in labels])
            
            # Ensure we have exactly n_points after filtering
            if len(points) != self.n_points:
                if len(points) < self.n_points:
                    # Upsample if we have fewer points (PointNet needs fixed size)
                    indices = np.random.choice(len(points), self.n_points, replace=True)
                    points = points[indices]
                    labels = labels[indices]
                    class_indices = class_indices[indices]
                elif len(points) > self.n_points:
                    # Downsample if we have more points
                    indices = np.random.choice(len(points), self.n_points, replace=False)
                    points = points[indices]
                    labels = labels[indices]
                    class_indices = class_indices[indices]
        
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
        
        # Final cleanup of numpy arrays
        del points, labels, class_indices, unique_classes, counts
        import gc
        gc.collect()
        
        return {
            'points': points_tensor,              # [n_points, 3]
            'point_labels': point_labels_tensor,  # [n_points] - individual point labels
            'object_class': object_class_tensor,  # scalar - dominant class for this point cloud
            'file_path': file_path
        }

