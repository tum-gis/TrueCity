"""
Data processing utilities (updated for config.py integration)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
from augmentations import default_augmentation


class ToTensor:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        return torch.from_numpy(pointcloud)


class Normalize:
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


def default_transforms(train=False, config=None):
    ops = [Normalize()]
    if train:
        def _apply_aug(pc):
            arr = pc.to_numpy() if hasattr(pc, 'to_numpy') else pc
            xyz = arr[:, :3].astype(np.float32)
            lbl = arr[:, 3:4]
            aug_xyz, _, _ = default_augmentation(xyz, config)
            return np.hstack((aug_xyz, lbl)).astype(np.float32)
        ops += [_apply_aug]
    ops += [ToTensor()]
    return transforms.Compose(ops)


def collate_fn(batch, voxel_size=0.01):
    coords, feats, labels, offsets, grid_coords = [], [], [], [], []
    count = 0
    if not batch:
        raise ValueError("Empty batch")

    # Allow passing config.voxel_size via torch.utils.data default collate args by closure or use default
    vx = voxel_size
    if hasattr(collate_fn, 'voxel_size_override') and collate_fn.voxel_size_override is not None:
        vx = float(collate_fn.voxel_size_override)

    for i, item in enumerate(batch):
        if item.shape[1] < 4:
            raise ValueError(f"Item {i} has insufficient columns: {item.shape[1]}")

        coord = item[:, 0:3].to(torch.float32)
        feat = coord.clone()
        label = item[:, 3].to(torch.int64)

        # Shift coordinates per-sample to ensure non-negative grid indices
        coord_min = coord.min(dim=0).values
        coord_shifted = coord - coord_min
        grid_coord = torch.floor(coord_shifted / vx).to(torch.int32)

        # Add a dummy batch dimension for uniqueness check (per-sample loop is already isolated)
        full_grid = torch.cat([grid_coord, torch.zeros_like(grid_coord[:, :1])], dim=1)  # (N, 4)

        unique_full_grid, unique_idx = np.unique(full_grid.cpu().numpy(), axis=0, return_index=True)
        unique_idx = torch.from_numpy(unique_idx).long().to(coord.device)

        grid_coords.append(grid_coord[unique_idx])
        coords.append(coord[unique_idx])
        feats.append(feat[unique_idx])
        labels.append(label[unique_idx])

        count += len(unique_idx)
        offsets.append(count)

    grid_coords = torch.cat(grid_coords, dim=0)
    coords = torch.cat(coords, dim=0)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    offsets = torch.IntTensor(offsets)

    # Removed second normalization of coords to avoid distribution shift
    return grid_coords, coords, feats, labels, offsets

def load_data(config):
    root_folder = config.data_root
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


def dataloader_tumfacade(lxyz, config, shuffle=False):
    if config.sample_ratio < 1.0:
        lxyz = lxyz.sample(frac=config.sample_ratio, random_state=42).reset_index(drop=True)
        print(f"Using {config.sample_ratio*100}% of data: {len(lxyz)} points")

    point_transforms = default_transforms(train=shuffle, config=config)
    batch_num = int(len(lxyz) / config.npoints_batch)

    print(f"Creating {batch_num} batches with {config.npoints_batch} points each")

    _data = lxyz[0:(config.npoints_batch * batch_num)]
    data_temp = np.split(_data, batch_num)
    data_trans = [point_transforms(test) for test in data_temp]

    data_loader = DataLoader(
        dataset=data_trans,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return data_loader, _data


def _apply_label_mapping(blocks, num_classes, src_lbl=None, dst_lbl=None):
    for arr in blocks:
        # ensure integer labels
        arr[:, 3] = arr[:, 3].astype(np.int32)
        if src_lbl is not None and dst_lbl is not None:
            arr[:, 3][arr[:, 3] == int(src_lbl)] = int(dst_lbl)
        # clamp any unexpected labels into valid range
        arr[:, 3] = np.clip(arr[:, 3], 0, int(num_classes) - 1)


def _log_label_stats(name, blocks, num_classes):
    import numpy as _np
    labels = _np.concatenate([blk[:, 3].astype(_np.int32) for blk in blocks]) if blocks else _np.array([], dtype=_np.int32)
    if labels.size > 0:
        uniq, cnt = _np.unique(labels, return_counts=True)
        print(f"[LABEL] {name}: unique {uniq.tolist()} (len={len(uniq)})")
        counts = _np.bincount(labels, minlength=int(num_classes)).astype(_np.int64)
        print(f"[LABEL] {name}: bincount len={len(counts)} -> {counts.tolist()}")
    else:
        print(f"[LABEL] {name}: no labels")


def calculate_class_weights(data, num_classes):
    labels = data['cla'].to_numpy().astype(np.int32)
    labels = np.clip(labels, 0, int(num_classes) - 1)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float64)
    print("Class distribution:", counts.astype(np.int64))
    counts = counts / np.sum(counts)
    weights = np.power(np.amax(counts) / np.maximum(counts, 1e-12), 1 / 3.0)
    print("Class weights (len={}):".format(int(num_classes)), weights)
    return weights


def load_preprocessed_data(split_root):
    if not os.path.exists(split_root):
        raise ValueError(f"Split directory not found: {split_root}")

    block_files = [f for f in os.listdir(split_root) if f.endswith('.npy')]
    if not block_files:
        raise ValueError(f"No .npy files found in {split_root}")

    print(f"[IO] {os.path.basename(split_root)}: found {len(block_files)} .npy blocks")
    start_t = time.time()
    blocks = []
    for block_file in tqdm(block_files, desc=f"Load {os.path.basename(split_root)}", dynamic_ncols=True, unit='blk', mininterval=0.5):
        block_path = os.path.join(split_root, block_file)
        block_data = np.load(block_path)  # [2048, 4] -> [x, y, z, label]
        blocks.append(block_data)

    elapsed = time.time() - start_t
    rate = len(blocks) / elapsed if elapsed > 0 else 0.0
    print(f"[IO] Loaded {len(blocks)} blocks from {split_root} in {elapsed:.1f}s ({rate:.1f} blk/s)")
    return blocks

def create_preprocessed_dataloaders(config):
    train_blocks = load_preprocessed_data(config.train_data_root)
    val_blocks = load_preprocessed_data(config.val_data_root)
    test_blocks = load_preprocessed_data(config.test_data_root)

    # Merge rare class and clamp labels
    src_lbl = getattr(config, 'merge_label_from', None)
    dst_lbl = getattr(config, 'merge_label_to', None)
    _apply_label_mapping(train_blocks, config.num_classes, src_lbl, dst_lbl)
    _apply_label_mapping(val_blocks, config.num_classes, src_lbl, dst_lbl)
    _apply_label_mapping(test_blocks, config.num_classes, src_lbl, dst_lbl)

    # Log label stats after mapping
    _log_label_stats('train', train_blocks, config.num_classes)
    _log_label_stats('val', val_blocks, config.num_classes)
    _log_label_stats('test', test_blocks, config.num_classes)

    if config.calculate_class_weights:
        train_labels = []
        for block in train_blocks:
            train_labels.extend(block[:, 3].astype(int))
        train_df = pd.DataFrame({'cla': train_labels})
        config.class_weights = calculate_class_weights(train_df, config.num_classes)

    # Train-time augmentation (via augmentations.py), val/test clean
    train_transforms = default_transforms(train=True, config=config)

    print(f"[IO] Transform train: {len(train_blocks)} blocks")
    t0 = time.time()
    train_data = [
        train_transforms(pd.DataFrame(block, columns=["X", "Y", "Z", "cla"]))
        for block in tqdm(train_blocks, desc="Transform train", dynamic_ncols=True, unit='blk', mininterval=0.5)
    ]
    t1 = time.time()
    print(f"[IO] Transform train done in {t1 - t0:.1f}s")

    print(f"[IO] Transform val: {len(val_blocks)} blocks")
    t2 = time.time()
    val_data = [
        default_transforms(train=False, config=config)(pd.DataFrame(block, columns=["X", "Y", "Z", "cla"]))
        for block in tqdm(val_blocks, desc="Transform val", dynamic_ncols=True, unit='blk', mininterval=0.5)
    ]
    t3 = time.time()
    print(f"[IO] Transform val done in {t3 - t2:.1f}s")

    print(f"[IO] Transform test: {len(test_blocks)} blocks")
    t4 = time.time()
    test_data = [
        default_transforms(train=False, config=config)(pd.DataFrame(block, columns=["X", "Y", "Z", "cla"]))
        for block in tqdm(test_blocks, desc="Transform test", dynamic_ncols=True, unit='blk', mininterval=0.5)
    ]
    t5 = time.time()
    print(f"[IO] Transform test done in {t5 - t4:.1f}s")

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=getattr(config, 'workers', 0), pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=getattr(config, 'workers', 0), pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=getattr(config, 'workers', 0), pin_memory=True)

    # inject voxel size override into collate function instances
    collate_fn.voxel_size_override = getattr(config, 'voxel_size', None)

    return train_loader, val_loader, test_loader, test_blocks

def create_dataloaders(config):
    if getattr(config, 'use_preprocessed', False):
        return create_preprocessed_dataloaders(config)
    else:
        lxyz_train, lxyz_validation, lxyz_test = load_data(config)
        if config.calculate_class_weights:
            config.class_weights = calculate_class_weights(lxyz_train, config.num_classes)

        train_loader, _ = dataloader_tumfacade(lxyz_train, config, shuffle=True)
        valid_loader, _ = dataloader_tumfacade(lxyz_validation, config, shuffle=False)
        test_loader, test_dataset = dataloader_tumfacade(lxyz_test, config, shuffle=False)

        return train_loader, valid_loader, test_loader, test_dataset
