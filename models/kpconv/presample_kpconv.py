#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from typing import Tuple, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pre-sample point clouds for KPConv')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Input dataset root containing train/ val/ test/ with .npy files [x,y,z,label]')
    parser.add_argument('--out_root', type=str, required=True,
                        help='Output dataset root to write pre-sampled files (mirrors train/ val/ test/)')
    parser.add_argument('--target_points', type=int, default=16384,
                        help='Target number of points per output sample')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    np.random.seed(seed)


def load_cloud(npy_path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(npy_path)
    assert arr.ndim == 2 and arr.shape[1] >= 4, f'Invalid array shape: {arr.shape} in {npy_path}'
    xyz = arr[:, :3].astype(np.float32)
    labels = arr[:, 3].astype(np.int64)
    return xyz, labels


def save_cloud(out_path: str, xyz: np.ndarray, labels: np.ndarray) -> None:
    assert xyz.shape[0] == labels.shape[0], 'Points and labels length mismatch'
    out_arr = np.concatenate([xyz.astype(np.float32), labels.reshape(-1, 1).astype(np.int64)], axis=1)
    out_dir = os.path.dirname(out_path)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, out_arr)


RATIO_BASE = 16384


def list_split_files(root: str) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    # Case 1: flat directory of .npy files -> treat as 'train'
    flat_npy = [fname for fname in sorted(os.listdir(root)) if fname.endswith('.npy')]
    if len(flat_npy) > 0:
        for fname in flat_npy:
            files.append(('train', os.path.join(root, fname)))
        return files
    # Case 2: expected split subdirectories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith('.npy'):
                files.append((split, os.path.join(split_dir, fname)))
    return files


def partition_indices(num_points: int, target_points: int, seed: int) -> List[np.ndarray]:
    # Number of partitions to reach ~target_points per chunk
    k = int(np.ceil(num_points / float(max(target_points, 1))))
    k = max(k, 1)

    # Compute sizes that sum exactly to num_points
    base = num_points // k
    rem = num_points % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    # Deterministic shuffle per file based on seed
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_points).astype(np.int64)
    # Split permutation into contiguous chunks
    out: List[np.ndarray] = []
    start = 0
    for sz in sizes:
        out.append(perm[start:start+sz])
        start += sz
    return out


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)

    assert os.path.isdir(args.data_root), f'Input root not found: {args.data_root}'
    if not os.path.isdir(args.out_root):
        os.makedirs(args.out_root, exist_ok=True)
    for split in ['train', 'val', 'test']:
        out_split = os.path.join(args.out_root, split)
        if not os.path.isdir(out_split):
            os.makedirs(out_split, exist_ok=True)

    in_files = list_split_files(args.data_root)
    assert len(in_files) > 0, f'No .npy files found under {args.data_root}'

    print(f'Found {len(in_files)} files. Partitioning into ~{args.target_points} points per sample (non-overlapping).')

    processed = 0
    total_original_points = 0
    total_preprocessed_samples = 0
    for split, fpath in in_files:
        xyz, labels = load_cloud(fpath)
        base = os.path.splitext(os.path.basename(fpath))[0]
        # Partition into non-overlapping chunks that sum to original size
        file_seed = abs(hash(base)) % (2**31)
        parts = partition_indices(xyz.shape[0], args.target_points, args.seed + file_seed)
        total_original_points += int(xyz.shape[0])
        total_preprocessed_samples += len(parts)
        for pidx, idx in enumerate(parts):
            xyz_s = xyz[idx]
            labels_s = labels[idx]
            out_name = f'{base}_part-{pidx:02d}_of-{len(parts):02d}.npy'
            out_path = os.path.join(args.out_root, split, out_name)
            save_cloud(out_path, xyz_s, labels_s)
        processed += 1
        if processed % 50 == 0 or processed == len(in_files):
            print(f'Processed {processed}/{len(in_files)} files', flush=True)

    ratio_process = total_original_points / float(max(total_preprocessed_samples * RATIO_BASE, 1))
    print(f'Done. ratio-process = {ratio_process:.6f} (total_original_points={total_original_points}, preprocessed_samples={total_preprocessed_samples}, base={RATIO_BASE})')


if __name__ == '__main__':
    main() 