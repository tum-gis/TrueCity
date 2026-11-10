#!/usr/bin/env python3
"""Create real-synthetic point cloud mixes.

For each training split, the real point cloud is cut along one axis into
N_PARTS contiguous segments of equal 2D occupancy area. The synthetic point
cloud is cut at the same coordinates, so real and synthetic segments line up
seamlessly. Mix k (k = 0..N_PARTS) takes the first k segments from the real
cloud and the remaining ones from the synthetic cloud.

Inputs (in --data_dir):
    {split}_real.npy, {split}_synthetic.npy     arrays of shape (N, 4): x, y, z, class

Outputs:
    {data_dir}/mix/{split}_{ratio}.npy          ratio = percentage of real segments (0, 25, ..., 100)
    {data_dir}/cut/cut_coordinate.npy           cut coordinates, shape (len(SPLIT_AXIS), N_PARTS - 1)
    {data_dir}/cut/{split}_{source}_{i}.npy     individual segments (only with --save_cuts)

With --label_rs, a fifth column marks each point's source: 1 = synthetic, 0 = real.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterator, Optional

import numpy as np

SPLIT_AXIS = {"train1": "y", "train2": "x"}  # axis along which each split is cut
AXIS_IDX = {"x": 0, "y": 1}
N_PARTS = 4
GRID_SIZE = 0.1  # occupancy grid cell size, in point cloud units
SYNTHETIC_LABEL = 1
REAL_LABEL = 0
MIX_DIR = "mix"
CUT_DIR = "cut"


def load_points(path: str, source_label: Optional[int] = None) -> np.ndarray:
    """Load an (N, 4) x, y, z, class array, optionally appending a source label column."""
    points = np.load(path)[:, :4]
    if source_label is not None:
        points = np.column_stack([points, np.full(len(points), source_label, dtype=points.dtype)])
    return points


def find_area_boundaries(points: np.ndarray, axis: str, n_parts: int,
                         grid_size: float = GRID_SIZE) -> np.ndarray:
    """Return n_parts - 1 coordinates along `axis` that cut `points` into parts of equal 2D occupancy area.

    Occupancy area is the number of occupied grid cells in the xy plane. Points are
    swept along the axis, and each cell counts toward the part in which it first appears.
    """
    a = AXIS_IDX[axis]
    order = np.argsort(points[:, a], kind="stable")

    xy = points[order, :2]
    ij = np.floor((xy - xy.min(axis=0)) / grid_size).astype(np.int64)
    cell_ids = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]

    _, first_seen = np.unique(cell_ids, return_index=True)
    is_new_cell = np.zeros(len(order), dtype=bool)
    is_new_cell[first_seen] = True
    cum_area = np.cumsum(is_new_cell)

    targets = cum_area[-1] * np.arange(1, n_parts) / n_parts
    cut_idx = np.searchsorted(cum_area, targets)
    return points[order[cut_idx], a]


def split_by_boundaries(points: np.ndarray, axis: str, boundaries: np.ndarray) -> list[np.ndarray]:
    """Split points into len(boundaries) + 1 parts. A point on a boundary goes to the part after it."""
    part = np.searchsorted(boundaries, points[:, AXIS_IDX[axis]], side="right")
    return [points[part == k] for k in range(len(boundaries) + 1)]


def build_mixes(real_parts: list[np.ndarray],
                synth_parts: list[np.ndarray]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (real_percentage, points) with the first k parts real and the rest synthetic."""
    n = len(real_parts)
    for k in range(n + 1):
        yield round(100 * k / n), np.concatenate(real_parts[:k] + synth_parts[k:])


def process_split(data_dir: str, split: str, axis: str, label_rs: bool, save_cuts: bool) -> np.ndarray:
    """Cut and mix one split. Returns the cut coordinates."""
    print(f"[{split}] loading point clouds")
    real = load_points(os.path.join(data_dir, f"{split}_real.npy"), REAL_LABEL if label_rs else None)
    synth = load_points(os.path.join(data_dir, f"{split}_synthetic.npy"), SYNTHETIC_LABEL if label_rs else None)

    print(f"[{split}] finding equal-area cuts along {axis}")
    boundaries = find_area_boundaries(real, axis, N_PARTS)
    print(f"[{split}] cut coordinates: {boundaries}")

    real_parts = split_by_boundaries(real, axis, boundaries)
    synth_parts = split_by_boundaries(synth, axis, boundaries)
    del real, synth

    if save_cuts:
        cut_dir = os.path.join(data_dir, CUT_DIR)
        for source, parts in (("real", real_parts), ("synthetic", synth_parts)):
            for i, part in enumerate(parts):
                np.save(os.path.join(cut_dir, f"{split}_{source}_{i}.npy"), part)

    mix_dir = os.path.join(data_dir, MIX_DIR)
    for ratio, mix in build_mixes(real_parts, synth_parts):
        print(f"[{split}] saving {ratio}% real mix ({len(mix)} points)")
        np.save(os.path.join(mix_dir, f"{split}_{ratio}.npy"), mix)

    return boundaries


def main():
    parser = argparse.ArgumentParser(description="Create real-synthetic data mix.")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing {split}_real.npy and {split}_synthetic.npy")
    parser.add_argument("-l", "--label_rs", action="store_true",
                        help=f"Append a source column (synthetic={SYNTHETIC_LABEL}, real={REAL_LABEL})")
    parser.add_argument("--save_cuts", action="store_true",
                        help="Also save the individual real/synthetic segments")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.data_dir, MIX_DIR), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, CUT_DIR), exist_ok=True)

    cut_coords = [
        process_split(args.data_dir, split, axis, args.label_rs, args.save_cuts)
        for split, axis in SPLIT_AXIS.items()
    ]
    np.save(os.path.join(args.data_dir, CUT_DIR, "cut_coordinate.npy"), np.stack(cut_coords))


if __name__ == "__main__":
    main()
