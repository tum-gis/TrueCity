"""
TrueCity DataModule for Superpoint Transformer
Handles loading and preprocessing of Ingolstadt point cloud data for graph-based models
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any
import glob

# Try to import superpoint_transformer components
try:
    from torch_points3d.datasets.dataset import BaseDataset
    from torch_points3d.core.data_transform import Compose
    TORCH_POINTS3D_AVAILABLE = True
except ImportError:
    TORCH_POINTS3D_AVAILABLE = False
    print("Warning: torch_points3d not available. Some features may not work.")


class TrueCityDataset(Dataset):
    """
    Dataset class for TrueCity/Ingolstadt data compatible with Superpoint Transformer
    Converts raw point clouds to the format expected by graph-based models
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        allowed_classes: Optional[List[float]] = None,
        transform: Optional[Any] = None,
    ):
        """
        Args:
            data_path: Path to data directory
            split: 'train', 'val', or 'test'
            allowed_classes: List of allowed class labels (whitelist). If None, uses all classes except 13
            transform: Optional transform to apply
        """
        self.data_path = data_path
        self.split = split
        self.allowed_classes = allowed_classes
        self.transform = transform
        
        # Default: include all classes 0-11, exclude only 13
        if self.allowed_classes is None:
            self.allowed_classes = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        
        # Load file paths
        self.file_paths = self._load_file_paths()
        
        # Create class mapping
        self.class_to_idx, self.idx_to_class = self._create_class_mapping()
        self.num_classes = len(self.class_to_idx)
        
        print(f"✅ TrueCityDataset initialized: {len(self.file_paths)} files, {self.num_classes} classes")
        print(f"   Class mapping: {self.class_to_idx}")
    
    def _load_file_paths(self) -> List[str]:
        """Load all data file paths for the given split"""
        possible_paths = [
            os.path.join(self.data_path, self.split),
            self.data_path,
        ]
        
        file_paths = []
        for base_path in possible_paths:
            if os.path.exists(base_path):
                patterns = [
                    os.path.join(base_path, f"{self.split}*.npy"),
                    os.path.join(base_path, f"{self.split}_*.npy"),
                    os.path.join(base_path, "*.npy"),
                ]
                
                for pattern in patterns:
                    found = glob.glob(pattern)
                    file_paths.extend(found)
        
        if len(file_paths) == 0:
            raise ValueError(f"No files found for split '{self.split}' in {self.data_path}")
        
        return sorted(list(set(file_paths)))
    
    def _create_class_mapping(self) -> tuple:
        """Create class mapping from allowed classes"""
        # Filter to only allowed classes
        allowed_set = set(self.allowed_classes)
        sorted_classes = sorted(allowed_set)
        
        class_to_idx = {cls: i for i, cls in enumerate(sorted_classes)}
        idx_to_class = {i: cls for cls, i in class_to_idx.items()}
        
        return class_to_idx, idx_to_class
    
    def _load_file(self, file_path: str) -> np.ndarray:
        """Load a single NPY file"""
        data = np.load(file_path, allow_pickle=True)
        
        if len(data.shape) != 2 or data.shape[1] < 4:
            raise ValueError(f"Invalid data shape: {data.shape}, expected [N, 4+]")
        
        return data
    
    def _filter_classes(self, labels: np.ndarray) -> np.ndarray:
        """Filter labels to only include allowed classes and remap to contiguous indices"""
        allowed_set = set(self.allowed_classes)
        mask = np.isin(labels, list(allowed_set))
        
        # Remap to contiguous indices
        filtered_labels = labels[mask].copy()
        for original_cls, new_idx in self.class_to_idx.items():
            filtered_labels[filtered_labels == original_cls] = new_idx
        
        return filtered_labels, mask
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return a point cloud sample"""
        file_path = self.file_paths[idx]
        data = self._load_file(file_path)
        
        # Extract points and labels
        points = data[:, :3].astype(np.float32)  # [N, 3]
        labels = data[:, 3].astype(np.float32)   # [N]
        
        # Filter classes
        filtered_labels, mask = self._filter_classes(labels)
        points = points[mask]
        
        if len(points) == 0:
            # Fallback: use center point with first valid class
            center = np.mean(data[:, :3], axis=0, keepdims=True)
            points = center.astype(np.float32)
            filtered_labels = np.array([0], dtype=np.float32)
        
        # Convert to torch tensors
        pos = torch.from_numpy(points)
        y = torch.from_numpy(filtered_labels.astype(np.int64))
        
        # Create basic Data object (will be processed by transforms)
        data_obj = Data(
            pos=pos,
            y=y,
            file_path=file_path,
        )
        
        if self.transform is not None:
            data_obj = self.transform(data_obj)
        
        return data_obj


class TrueCityDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for TrueCity/Ingolstadt dataset
    Compatible with Superpoint Transformer preprocessing pipeline
    """
    
    def __init__(
        self,
        data_dir: str = '/home/stud/nguyenti/storage/user/EARLy/datav2_final',
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        allowed_classes: Optional[List[float]] = None,
        num_classes: int = 12,
        pre_transform: Optional[Any] = None,
        train_transform: Optional[Any] = None,
        val_transform: Optional[Any] = None,
        test_transform: Optional[Any] = None,
        **kwargs
    ):
        """
        Args:
            data_dir: Root directory containing train/val/test splits
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            persistent_workers: Whether to keep workers alive
            allowed_classes: List of allowed class labels (whitelist)
            num_classes: Number of classes (after filtering)
            pre_transform: Transform applied during preprocessing
            train_transform: Transform applied during training
            val_transform: Transform applied during validation
            test_transform: Transform applied during testing
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        # Default: include all classes 0-11
        if allowed_classes is None:
            allowed_classes = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        self.allowed_classes = allowed_classes
        self.num_classes = num_classes
        
        self.pre_transform = pre_transform
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        
        # Datasets (will be created in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each split"""
        if stage == 'fit' or stage is None:
            # Create train dataset
            train_data_path = self.data_dir
            self.train_dataset = TrueCityDataset(
                data_path=train_data_path,
                split='train',
                allowed_classes=self.allowed_classes,
                transform=self.train_transform,
            )
            
            # Create val dataset
            self.val_dataset = TrueCityDataset(
                data_path=train_data_path,
                split='val',
                allowed_classes=self.allowed_classes,
                transform=self.val_transform,
            )
            
            # Verify class consistency
            train_classes = set(self.train_dataset.class_to_idx.keys())
            val_classes = set(self.val_dataset.class_to_idx.keys())
            
            if train_classes != val_classes:
                print(f"⚠️ Warning: Class mismatch detected!")
                print(f"   Train classes: {sorted(train_classes)}")
                print(f"   Val classes: {sorted(val_classes)}")
                print(f"   Using intersection: {sorted(train_classes & val_classes)}")
                
                # Use intersection
                common_classes = sorted(train_classes & val_classes)
                self.allowed_classes = [float(c) for c in common_classes]
                self.num_classes = len(common_classes)
                
                # Recreate datasets with common classes
                self.train_dataset = TrueCityDataset(
                    data_path=train_data_path,
                    split='train',
                    allowed_classes=self.allowed_classes,
                    transform=self.train_transform,
                )
                self.val_dataset = TrueCityDataset(
                    data_path=train_data_path,
                    split='val',
                    allowed_classes=self.allowed_classes,
                    transform=self.val_transform,
                )
            
            print(f"✅ Training setup complete: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
            print(f"   Number of classes: {self.num_classes}")
        
        if stage == 'test' or stage is None:
            # Create test dataset
            test_data_path = self.data_dir
            self.test_dataset = TrueCityDataset(
                data_path=test_data_path,
                split='test',
                allowed_classes=self.allowed_classes,
                transform=self.test_transform,
            )
            
            print(f"✅ Test setup complete: {len(self.test_dataset)} test samples")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )


