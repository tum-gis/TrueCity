import os
import os.path as osp
import glob
import numpy as np
import torch
import logging
from typing import List, Dict, Optional

from src.data import Data, InstanceData
from src.datasets.base import BaseDataset
from src.datasets.tumdilab_config import (
    CLASS_NAMES, TUMDILAB_NUM_CLASSES, CLASS_COLORS, STUFF_CLASSES
)

log = logging.getLogger(__name__)

# Hack to solve dataloader issues on some machines
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

__all__ = ["Tumdilab", "MiniTumdilab"]


def _read_one_block_npy(npy_path: str) -> Data:
    """
    Read a single preprocessed block stored as numpy array.

    Expected format: [N, 4] with columns [x, y, z, label].
    Returns a Data object with fields: pos, y, pos_offset, obj (dummy).
    """
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Invalid npy shape at {npy_path}: {arr.shape}")

    # positions
    pos = torch.from_numpy(arr[:, :3].astype(np.float32))
    data = Data(pos=pos, pos_offset=torch.zeros(3, dtype=torch.float32))

    # labels -> map out-of-range to ignored (num_classes)
    ignored = TUMDILAB_NUM_CLASSES
    if arr.shape[1] >= 4:
        y_np = arr[:, 3].astype(np.int64)
        mask_valid = (y_np >= 0) & (y_np < TUMDILAB_NUM_CLASSES)
        y_np[~mask_valid] = ignored
        data.y = torch.from_numpy(y_np)
    else:
        data.y = torch.full((pos.shape[0],), ignored, dtype=torch.long)

    # dummy instance info (kept for robustness in transforms)
    idx = torch.arange(pos.shape[0], dtype=torch.long)
    obj = torch.zeros(pos.shape[0], dtype=torch.long)   # single instance 0
    count = torch.ones_like(obj)
    data.obj = InstanceData(idx, obj, count, data.y, dense=True)

    return data


class Tumdilab(BaseDataset):
    """
    Tumdilab dataset for point cloud semantic segmentation.

    Raw layout:
        {root}/raw/
          ├─ train/
          │    └─ datav2_{real_ratio}_octree_fps/*.npy
          ├─ val/*.npy
          └─ test/*.npy

    Each .npy file contains a point cloud block with shape [N, 4+]
    where columns are [x, y, z, label, ...].
    """

    # ---------------- Initialization ----------------
    def __init__(self,
                 root: str,
                 stage: str = 'train',
                 transform=None,
                 pre_transform=None,
                 on_device_transform=None,
                 # dataset-specific options (forwarded by BaseDataModule)
                 real_ratio: Optional[int] = None,
                 train_subdir_tpl: Optional[str] = None,
                 fallback_to_plain_train: bool = True,
                 **kwargs):
        """
        Store dataset-specific options and PRECOMPUTE cloud ids BEFORE
        calling parent __init__, because BaseDataset.__init__ will call
        check_cloud_ids() early.
        """
        self.real_ratio = real_ratio
        self.train_subdir_tpl = train_subdir_tpl
        self.fallback_to_plain_train = bool(fallback_to_plain_train)

        # Precompute raw_dir by mimicking BaseDataset's root layout:
        # BaseDataset will later use root = os.path.join(root, data_subdir_name)
        dataset_root = osp.join(root, self.data_subdir_name)
        pre_raw_dir = osp.join(dataset_root, "raw")

        # Precompute the lists of .npy files for each split
        self._all_base_cloud_ids_cache = self._scan_all_base_cloud_ids(pre_raw_dir)

        # Now call parent init (this will run check_cloud_ids, which uses our cache)
        super().__init__(
            root=root,
            stage=stage,
            transform=transform,
            pre_transform=pre_transform,
            on_device_transform=on_device_transform,
            **kwargs
        )

    # ---------------- Helpers ----------------
    def _scan_all_base_cloud_ids(self, raw_dir: str) -> Dict[str, List[str]]:
        """
        Scan the filesystem to collect .npy files per split using the given raw_dir.
        This runs BEFORE InMemoryDataset sets self.raw_dir, so we pass raw_dir explicitly.
        """
        out = {'train': [], 'val': [], 'test': []}
        base = raw_dir  # expected {root}/{data_subdir}/raw

        # Resolve training directories based on dataset-specific options
        rr = self.real_ratio
        tpl = self.train_subdir_tpl
        fallback = self.fallback_to_plain_train

        train_dirs: List[str] = []
        if rr is not None and tpl:
            sub = tpl.format(real_ratio=rr) if "{real_ratio}" in tpl else tpl
            train_specific = osp.join(base, "train", sub)
            if osp.isdir(train_specific):
                train_dirs.append(train_specific)
                log.info(f"Using training directory: {train_specific}")

        if not train_dirs and fallback:
            train_plain = osp.join(base, "train")
            if osp.isdir(train_plain):
                npy_files = glob.glob(osp.join(train_plain, "*.npy"))
                if npy_files:
                    train_dirs.append(train_plain)
                    log.info(f"Using fallback training directory: {train_plain}")
                else:
                    subdirs = [d for d in glob.glob(osp.join(train_plain, "*"))
                               if osp.isdir(d)]
                    train_dirs.extend(subdirs)
                    log.info(f"Found {len(subdirs)} training subdirectories")

        def _list_npy(directory: str) -> List[str]:
            """List all .npy files in a directory (as relative paths from raw_dir)."""
            pattern = osp.join(directory, "*.npy")
            files = sorted(glob.glob(pattern))
            return [osp.relpath(f, base) for f in files]

        # Train
        for d in train_dirs:
            files = _list_npy(d)
            out['train'].extend(files)
            log.info(f"Found {len(files)} training files in {d}")

        # Val
        val_dir = osp.join(base, "val")
        if osp.isdir(val_dir):
            out['val'] = _list_npy(val_dir)
            log.info(f"Found {len(out['val'])} validation files")

        # Test
        test_dir = osp.join(base, "test")
        if osp.isdir(test_dir):
            out['test'] = _list_npy(test_dir)
            log.info(f"Found {len(out['test'])} test files")

        if len(out['train']) == 0:
            raise RuntimeError(
                f"No training blocks found! Searched in: {train_dirs}\n"
                f"Base directory: {base}\n"
                f"Parameters: real_ratio={rr}, train_subdir_tpl={tpl}, "
                f"fallback={fallback}"
            )
        return out

    # ---------------- Dataset Properties ----------------
    @property
    def class_names(self) -> List[str]:
        """List of class names including 'ignored' as the last class."""
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        """Number of valid classes (excluding 'ignored')."""
        return TUMDILAB_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        """
        List of 'stuff' class indices for instance/panoptic segmentation.
        For pure semantic segmentation, this can be empty, but we keep
        the config value for compatibility.
        """
        return STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        """RGB colors for each class for visualization."""
        return CLASS_COLORS

    # IMPORTANT: do NOT override `all_cloud_ids` here.
    # Use the BaseDataset implementation which expands tiling if needed.

    @property
    def all_base_cloud_ids(self) -> Dict[str, List[str]]:
        """
        Return cached lists of raw .npy relative paths per split.
        This is safe to call during BaseDataset.__init__.
        """
        return self._all_base_cloud_ids_cache

    # ---------------- File I/O Methods ----------------
    def download_dataset(self) -> None:
        """No automatic download for Tumdilab."""
        log.info("Tumdilab dataset should be manually placed in the data directory")
        pass

    @property
    def raw_file_structure(self) -> str:
        """Human-readable description of expected raw file structure."""
        return f"""
{self.root}/
  └── raw/
      ├── train/
      │    └── datav2_{{real_ratio}}_octree_fps/
      │          ├── block_000001.npy
      │          ├── block_000002.npy
      │          └── ...
      ├── val/
      │    ├── val_000001.npy
      │    └── ...
      └── test/
           ├── test_000001.npy
           └── ...
"""

    def id_to_relative_raw_path(self, id: str) -> str:
        """
        Map a cloud id (possibly with tiling suffix) to the raw relative path.

        IMPORTANT: we must strip the tiling suffix if present, otherwise
        raw existence checks will fail when tiling is enabled.
        """
        return self.id_to_base_id(id)  # keep the original ".npy" id

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        """
        Read a single raw cloud file (.npy) and convert it to Data.
        """
        return _read_one_block_npy(raw_cloud_path)

    def processed_to_raw_path(self, processed_path: str) -> str:
        """
        Convert a processed path (<processed>/<stage>/<hash>/<cloud_id>.h5)
        back to the corresponding raw .npy path under self.raw_dir.
        """
        # Remove the <processed_dir>/ prefix
        rel = osp.relpath(processed_path, self.processed_dir)
        # Drop the first two components: <stage>/<hash>/..., keep the remainder
        # Use maxsplit=2 to preserve the full remainder path (which may include subdirs)
        parts = rel.split(os.sep, 2)
        remainder = parts[2] if len(parts) == 3 else parts[-1]
        # Remove trailing '.h5'
        if remainder.endswith('.h5'):
            remainder = remainder[:-3]
        # Join back under raw_dir
        raw_path = osp.join(self.raw_dir, remainder)
        return raw_path


class MiniTumdilab(Tumdilab):
    """
    A mini version of Tumdilab with only a few samples for quick testing.
    """
    _NUM_MINI = 8

    @property
    def all_base_cloud_ids(self) -> Dict[str, List[str]]:
        """Return only a subset of files for quick testing."""
        base = super().all_base_cloud_ids
        return {k: v[:self._NUM_MINI] for k, v in base.items()}

    @property
    def data_subdir_name(self) -> str:
        """Use the parent class name as subdir to share the same cache."""
        return self.__class__.__bases__[0].__name__.lower()

    def process(self) -> None:
        """Process the mini dataset by delegating to parent."""
        super().process()

    def download(self) -> None:
        """No download needed."""
        super().download()
