"""
Data processing utilities with global normalization support
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import json
from augmentations import default_augmentation


class ToTensor:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        return torch.from_numpy(pointcloud)


class Normalize:
    """Original per-batch normalization"""
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        # Convert DataFrame to numpy if needed
        if hasattr(pointcloud, 'to_numpy'):
            pointcloud = pointcloud.to_numpy()

        # Ensure float32
        pointcloud = pointcloud.astype(np.float32)

        # Separate coordinates and label
        xyz = pointcloud[:, :3]  # X, Y, Z
        label = pointcloud[:, 3:4]  # label

        # Normalize the coordinates
        norm_xyz = xyz - np.mean(xyz, axis=0)
        norm_xyz /= np.max(np.linalg.norm(norm_xyz, axis=1)) + 1e-6

        # Combine normalized coordinates with original label
        norm_pointcloud = np.hstack((norm_xyz, label))

        return norm_pointcloud


def create_test_dataloader_with_original_coords(config):
    """
    Create test dataloader that preserves original coordinates for proper denormalization
    """
    # Load test blocks
    test_blocks = load_preprocessed_data(config.test_data_root)
    
    # Load normalization stats for denormalization
    stats_path = os.path.join(config.save_path, "normalization_stats.json")
    if os.path.exists(stats_path):
        norm_stats = load_normalization_stats(stats_path)
        config.normalization_stats = norm_stats
    else:
        print("Warning: No normalization stats found for denormalization")
        norm_stats = None
    
    # Store original coordinates separately
    original_coords_list = []
    test_data = []
    
    train_transforms = default_transforms(train=False, config=None, use_global_norm=False, norm_stats=None)  # Use local normalization
    
    for block in test_blocks:
        # Store original coordinates
        original_coords_list.append(block[:, :3].copy())  # Store original xyz
        
        # Apply local normalization for model input
        normalized_block = train_transforms(pd.DataFrame(block, columns=["X", "Y", "Z", "cla"]))
        test_data.append(normalized_block)
    
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    return test_loader, original_coords_list
    """Original per-batch normalization"""
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        # Convert DataFrame to numpy if needed
        if hasattr(pointcloud, 'to_numpy'):
            pointcloud = pointcloud.to_numpy()

        # Ensure float32
        pointcloud = pointcloud.astype(np.float32)

        # Separate coordinates and label
        xyz = pointcloud[:, :3]  # X, Y, Z
        label = pointcloud[:, 3:4]  # label

        # Normalize the coordinates
        norm_xyz = xyz - np.mean(xyz, axis=0)
        norm_xyz /= np.max(np.linalg.norm(norm_xyz, axis=1)) + 1e-6

        # Combine normalized coordinates with original label
        norm_pointcloud = np.hstack((norm_xyz, label))

        return norm_pointcloud


class GlobalNormalize:
    """Global normalization using pre-computed statistics"""
    def __init__(self, mean, max_norm):
        self.mean = np.array(mean)
        self.max_norm = max_norm
        
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        # Convert DataFrame to numpy if needed
        if hasattr(pointcloud, 'to_numpy'):
            pointcloud = pointcloud.to_numpy()

        # Ensure float32
        pointcloud = pointcloud.astype(np.float32)

        # Separate coordinates and label
        xyz = pointcloud[:, :3]  # X, Y, Z
        label = pointcloud[:, 3:4]  # label

        # Apply global normalization
        norm_xyz = (xyz - self.mean) / (self.max_norm + 1e-6)

        # Combine normalized coordinates with original label
        norm_pointcloud = np.hstack((norm_xyz, label))

        return norm_pointcloud


def load_normalization_stats(stats_path):
    """Load normalization statistics from file"""
    if stats_path.endswith('.json'):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        stats['mean'] = np.array(stats['mean'])
    else:  # assume pickle
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
    
    return stats


def compute_global_normalization_stats_preprocessed(train_data_root, real_ratio):
    """
    Compute global normalization statistics from preprocessed data
    """
    print("Computing global normalization statistics from preprocessed data...")
    
    if not os.path.exists(train_data_root):
        raise ValueError(f"Training data folder not found: {train_data_root}")
    
    # Load all .npy files in the folder
    npy_files = [f for f in os.listdir(train_data_root) if f.endswith('.npy')]
    if not npy_files:
        raise ValueError(f"No .npy files found in {train_data_root}")
    
    print(f"Found {len(npy_files)} preprocessed files")
    
    all_xyz = []
    for i, npy_file in enumerate(sorted(npy_files)):
        file_path = os.path.join(train_data_root, npy_file)
        data = np.load(file_path)  # Shape: [2048, 4] -> [x, y, z, label]
        xyz = data[:, :3]  # X, Y, Z columns
        all_xyz.append(xyz)
        
        # Progress indicator for large datasets
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(npy_files)} files processed")
    
    # Combine all training data
    all_xyz = np.vstack(all_xyz)
    print(f"Total training points: {len(all_xyz):,}")
    
    # Compute global statistics
    global_mean = np.mean(all_xyz, axis=0)
    centered_xyz = all_xyz - global_mean
    norms = np.linalg.norm(centered_xyz, axis=1)
    global_max_norm = np.max(norms)
    
    stats = {
        'mean': global_mean,
        'max_norm': global_max_norm
    }
    
    print(f"Global mean: [{global_mean[0]:.6f}, {global_mean[1]:.6f}, {global_mean[2]:.6f}]")
    print(f"Global max norm: {global_max_norm:.6f}")
    
    return stats


def save_normalization_stats(stats, save_path):
    """Save normalization statistics to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as pickle
    pkl_path = save_path.replace('.json', '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(stats, f)
    
    # Save as JSON (for human readability)
    stats_json = {
        'mean': stats['mean'].tolist(),
        'max_norm': float(stats['max_norm'])
    }
    with open(save_path, 'w') as f:
        json.dump(stats_json, f, indent=2)
    
    print(f"Normalization stats saved to:")
    print(f"  - {save_path}")
    print(f"  - {pkl_path}")


def default_transforms(use_global_norm=False, norm_stats=None, train=False, config=None):
    """Create transform pipeline - supports both old and new signatures"""
    # Determine which signature is being used based on whether config is provided
    # New signature: train, config (config is not None)
    if config is not None:
        return transforms.Compose([
            Normalize(),
            (lambda pc: (
                (lambda arr: np.hstack((default_augmentation(arr[:, :3].astype(np.float32), config)[0], arr[:, 3:4])).astype(np.float32))
                (pc.to_numpy() if hasattr(pc, 'to_numpy') else pc)
            )) if train else (lambda pc: (pc.to_numpy() if hasattr(pc, 'to_numpy') else pc).astype(np.float32)),
            ToTensor()
        ])
    
    # Old signature: use_global_norm, norm_stats
    if use_global_norm and norm_stats is not None:
        return transforms.Compose([
            GlobalNormalize(norm_stats['mean'], norm_stats['max_norm']),
            ToTensor()
        ])
    else:
        # Default: per-batch normalization
        return transforms.Compose([
            Normalize(),
        ToTensor()
    ])


def collate_fn(batch):
    if not batch:
        raise ValueError("Empty batch")

    for i, item in enumerate(batch):
        if item.shape[1] < 4:
            raise ValueError(f"Item {i} has insufficient columns: {item.shape[1]}")

    # coord, feat, label
    coord = [item[:, 0:3].to(torch.float32) for item in batch]
    feat = [item[:, 0:3].to(torch.float32) for item in batch]
    label = [item[:, 3].to(torch.int64) for item in batch]

    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)

    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def load_data(config):
    """Load original (non-preprocessed) data"""
    root_folder = config.base_data_root
    real_ratio = config.real_ratio

    # Load training data
    lxyz_train = []
    train_files = [f"train1_{real_ratio}.npy", f"train2_{real_ratio}.npy"]

    for f in train_files:
        file_path = os.path.join(root_folder, 'train', f)
        if os.path.exists(file_path):
            data_train_building = np.load(file_path)
            data_train_building = pd.DataFrame(data_train_building, columns=["X", "Y", "Z", "cla"])
            data_train_building["cla"] = data_train_building["cla"].astype(int)
            lxyz_train.append(data_train_building)
        else:
            print(f"Warning: {file_path} not found!")

    lxyz_train = pd.concat(lxyz_train).dropna()

    # Load test data
    lxyz_test = []
    for f in os.listdir(os.path.join(root_folder, 'test')):
        if f.endswith(".npy"):
            data_test_building = np.load(os.path.join(root_folder, 'test', f))
            data_test_building = pd.DataFrame(data_test_building, columns=["X", "Y", "Z", "cla"])
            data_test_building["cla"] = data_test_building["cla"].astype(int)
            lxyz_test.append(data_test_building)
    lxyz_test = pd.concat(lxyz_test).dropna()

    # Load validation data
    lxyz_validation = []
    for f in os.listdir(os.path.join(root_folder, 'val')):
        if f.endswith(".npy"):
            data_validation_building = np.load(os.path.join(root_folder, 'val', f))
            data_validation_building = pd.DataFrame(data_validation_building, columns=["X", "Y", "Z", "cla"])
            data_validation_building["cla"] = data_validation_building["cla"].astype(int)
            lxyz_validation.append(data_validation_building)
    lxyz_validation = pd.concat(lxyz_validation).dropna()

    print(f"Training samples: {len(lxyz_train)}")
    print(f"Test samples: {len(lxyz_test)}")
    print(f"Validation samples: {len(lxyz_validation)}")

    return lxyz_train, lxyz_validation, lxyz_test


def dataloader_tumfacade(lxyz, config, is_train=False, shuffle=False, use_global_norm=False, norm_stats=None):
    """Create dataloader for original data"""
    if config.sample_ratio < 1.0:
        lxyz = lxyz.sample(frac=config.sample_ratio, random_state=42).reset_index(drop=True)
        print(f"Using {config.sample_ratio*100}% of data: {len(lxyz)} points")

    train_transforms = default_transforms(train=is_train, config=config, use_global_norm=use_global_norm, norm_stats=norm_stats)
    batch_num = int(len(lxyz) / config.npoints_batch)

    print(f"Creating {batch_num} batches with {config.npoints_batch} points each")

    _data = lxyz[0:(config.npoints_batch * batch_num)]
    data_temp = np.split(_data, batch_num)
    data_trans = [train_transforms(test) for test in data_temp]

    data_loader = DataLoader(
        dataset=data_trans,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return data_loader, _data


def calculate_class_weights(data, num_classes):
    labelweights = data['cla'].value_counts().sort_index().values
    print("Class distribution:", labelweights)

    labelweights = labelweights / np.sum(labelweights)
    weights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
    print("Class weights:", weights)

    return weights


def load_preprocessed_data(split_root):
    """Load preprocessed block data"""
    if not os.path.exists(split_root):
        raise ValueError(f"Split directory not found: {split_root}")

    block_files = [f for f in os.listdir(split_root) if f.endswith('.npy')]
    if not block_files:
        raise ValueError(f"No .npy files found in {split_root}")

    blocks = []
    for block_file in sorted(block_files):
        block_path = os.path.join(split_root, block_file)
        block_data = np.load(block_path)  # [2048, 4] -> [x, y, z, label]
        blocks.append(block_data)

    print(f"Loaded {len(blocks)} blocks from {split_root}")
    return blocks


def create_preprocessed_dataloaders(config):
    """Create dataloaders for preprocessed data with training-compatible normalization"""
    
    # Check if normalization stats exist, if not compute them
    stats_path = os.path.join(config.save_path, "normalization_stats.json")
    
    if not os.path.exists(stats_path):
        print("Computing and saving global normalization statistics...")
        # Compute global stats from preprocessed training data
        norm_stats = compute_global_normalization_stats_preprocessed(
            config.train_data_root, config.real_ratio
        )
        save_normalization_stats(norm_stats, stats_path)
    else:
        print("Loading existing normalization statistics...")
        norm_stats = load_normalization_stats(stats_path)
    
    # Store stats in config for later use
    config.normalization_stats = norm_stats
    
    # Load preprocessed data
    train_blocks = load_preprocessed_data(config.train_data_root)
    val_blocks = load_preprocessed_data(config.val_data_root)
    test_blocks = load_preprocessed_data(config.test_data_root)

    if config.calculate_class_weights:
        train_labels = []
        for block in train_blocks:
            train_labels.extend(block[:, 3].astype(int))
        train_df = pd.DataFrame({'cla': train_labels})
        config.class_weights = calculate_class_weights(train_df, config.num_classes)

    # IMPORTANT: Use the SAME normalization as during training
    # Since your model was trained with per-batch normalization, we use that for compatibility
    print("Using per-batch normalization (training-compatible mode)")
    train_transforms = default_transforms(train=True, config=config, use_global_norm=False, norm_stats=None)
    eval_transforms = default_transforms(train=False, config=config, use_global_norm=False, norm_stats=None)

    train_data = [train_transforms(pd.DataFrame(block, columns=["X", "Y", "Z", "cla"]))
                  for block in train_blocks]
    val_data = [eval_transforms(pd.DataFrame(block, columns=["X", "Y", "Z", "cla"]))
                for block in val_blocks]
    test_data = [eval_transforms(pd.DataFrame(block, columns=["X", "Y", "Z", "cla"]))
                 for block in test_blocks]

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, test_blocks


def create_dataloaders(config):
    """Main function to create dataloaders based on config"""
    if getattr(config, 'use_preprocessed', False):
        return create_preprocessed_dataloaders(config)
    else:
        # For original data, try to load or compute normalization stats
        stats_path = os.path.join(config.save_path, "normalization_stats.json")
        
        if not os.path.exists(stats_path):
            print("Warning: No normalization statistics found for original data.")
            print("Using per-batch normalization (original behavior).")
            norm_stats = None
        else:
            print("Loading existing normalization statistics...")
            norm_stats = load_normalization_stats(stats_path)
            config.normalization_stats = norm_stats
        
        lxyz_train, lxyz_validation, lxyz_test = load_data(config)
        if config.calculate_class_weights:
            config.class_weights = calculate_class_weights(lxyz_train, config.num_classes)

        use_global = norm_stats is not None
        train_loader, _ = dataloader_tumfacade(lxyz_train, config, is_train=True, shuffle=True, 
                                             use_global_norm=use_global, norm_stats=norm_stats)
        valid_loader, _ = dataloader_tumfacade(lxyz_validation, config, is_train=False, shuffle=False, 
                                             use_global_norm=use_global, norm_stats=norm_stats)
        test_loader, test_dataset = dataloader_tumfacade(lxyz_test, config, is_train=False, shuffle=False, 
                                                       use_global_norm=use_global, norm_stats=norm_stats)

        return train_loader, valid_loader, test_loader, test_dataset