import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import logging
from pathlib import Path

from sklearn.neighbors import KDTree
import time

VERBOSE = os.environ.get('DILAB_VERBOSE', '0') == '1'
K_MAX = int(os.environ.get('DILAB_MAX_NEIGHBORS', '64'))

# Setup logging to file for deadlock detection
_LOG_DIR = Path(os.environ.get('DILAB_LOG_DIR', '/tmp'))
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / f'kpconv_worker_{os.getpid()}.log'
_logger = logging.getLogger(f'kpconv_worker_{os.getpid()}')
_logger.setLevel(logging.DEBUG)
_handler = logging.FileHandler(_LOG_FILE)
_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
_logger.addHandler(_handler)
_logger.info(f"Worker {os.getpid()} started, logging to {_LOG_FILE}")

# Runtime config provided by the training script
_RUNTIME_CFG = None


def set_runtime_config(cfg):
    global _RUNTIME_CFG
    _RUNTIME_CFG = cfg


def _voxel_grid_subsample(points: np.ndarray, lengths: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    subsampled_list: List[np.ndarray] = []
    new_lengths: List[int] = []
    start = 0
    for ln in lengths:
        cloud = points[start:start+ln]
        if voxel_size <= 0:
            subsampled = cloud
        else:
            vox = np.floor(cloud / voxel_size).astype(np.int64)
            keys = vox.view([('', vox.dtype)] * vox.shape[1])
            _, idx = np.unique(keys, return_index=True)
            subsampled = cloud[np.sort(idx)]
        subsampled_list.append(subsampled)
        new_lengths.append(len(subsampled))
        start += ln
    return np.concatenate(subsampled_list, axis=0) if subsampled_list else np.zeros((0, 3), dtype=np.float32), np.asarray(new_lengths, dtype=np.int32)


def _radius_neighbors(queries: np.ndarray, supports: np.ndarray, q_lengths: np.ndarray, s_lengths: np.ndarray, radius: float) -> np.ndarray:
    t0 = time.time()
    total_queries = queries.shape[0]
    total_supports = supports.shape[0]
    _logger.info(f"_radius_neighbors START: queries={total_queries} supports={total_supports} radius={radius:.4f}")
    
    results: List[List[int]] = []
    offsets: List[int] = []
    max_k = 0
    q_start = 0
    s_start = 0
    num_batches = len(q_lengths)
    
    # Thresholds for logging (reduce noise for small/fast operations)
    LOG_POINT_THRESHOLD = 10000  # Only log if points > this
    LOG_TIME_THRESHOLD = 0.1  # Only log if operation takes > this many seconds
    
    for qi, (q_ln, s_ln) in enumerate(zip(q_lengths, s_lengths)):
        batch_start_time = time.time()
        _logger.info(f"Processing batch {qi+1}/{num_batches}: q_len={q_ln} s_len={s_ln}")
        
        q = queries[q_start:q_start+q_ln]
        s = supports[s_start:s_start+s_ln]
        
        if s_ln == 0:
            _logger.warning(f"Batch {qi+1}: Empty support set, skipping KDTree")
            neigh = [[] for _ in range(q_ln)]
        else:
            # Only log for significant operations
            should_log = s_ln > LOG_POINT_THRESHOLD or q_ln > LOG_POINT_THRESHOLD
            
            kdtree_start = time.time()
            if should_log:
                _logger.info(f"Batch {qi+1}: Building KDTree for {s_ln} points...")
            
            tree = KDTree(s)
            kdtree_build_time = time.time() - kdtree_start
            
            if should_log and kdtree_build_time > LOG_TIME_THRESHOLD:
                _logger.info(f"Batch {qi+1}: KDTree built in {kdtree_build_time:.2f}s")
            
            query_start = time.time()
            if should_log:
                _logger.info(f"Batch {qi+1}: Starting query_radius for {q_ln} queries with radius {radius:.4f}...")
            
            inds = tree.query_radius(q, r=radius)
            
            query_time = time.time() - query_start
            total_batch_time = time.time() - batch_start_time
            
            # Don't print batch-level messages - too noisy
            # Only log to file for debugging
            
            if should_log:
                _logger.info(f"Batch {qi+1}: query_radius completed in {query_time:.2f}s, processing {len(inds)} results")
            
            # Process results
            process_start = time.time()
            neigh = []
            for idx, ind in enumerate(inds):
                if len(ind) > K_MAX:
                    neigh.append(ind[:K_MAX].tolist())
                else:
                    neigh.append(ind.tolist())
            
            process_time = time.time() - process_start
            if should_log and process_time > LOG_TIME_THRESHOLD:
                _logger.info(f"Batch {qi+1}: Processed all neighbors in {process_time:.2f}s")
        
        max_k = max(max_k, max((len(x) for x in neigh), default=0))
        results.extend(neigh)
        offsets.extend([s_start] * q_ln)
        q_start += q_ln
        s_start += s_ln
        
        batch_time = time.time() - batch_start_time
        _logger.info(f"Batch {qi+1}/{num_batches} completed in {batch_time:.2f}s")
    
    # Padding phase
    pad_start = time.time()
    _logger.info(f"Padding results: {len(results)} results, max_k={max_k}")
    shadow_index = supports.shape[0] - 1 if supports.shape[0] > 0 else 0
    padded = np.full((len(results), max_k if max_k > 0 else 1), shadow_index, dtype=np.int64)
    for i, lst in enumerate(results):
        if lst:
            padded[i, :len(lst)] = (np.asarray(lst, dtype=np.int64) + offsets[i])
    pad_time = time.time() - pad_start
    _logger.info(f"Padding completed in {pad_time:.2f}s")
    
    total_time = time.time() - t0
    _logger.info(f"_radius_neighbors COMPLETE: took {total_time:.2f}s (max_k={max_k})")
    
    if VERBOSE:
        print(f'[DiLab] radius_neighbors: queries={total_queries} supports={total_supports} max_k={max_k} r={radius:.4f} took {1000*total_time:.1f}ms', flush=True)
    return padded


class DiLabDataset(Dataset):
    def __init__(self, config, set='training'):
        self.config = config
        self.set = set

        self.label_to_names = {
            0: 'Road Surface',
            1: 'Ground surface',
            2: 'Road installations',
            3: 'Vehicle',
            4: 'Pedestrian',
            5: 'Wall surface',
            6: 'Roof surface',
            7: 'Door',
            8: 'Window',
            9: 'Building installation',
            10: 'Tree',
            11: 'Noise',
        }
        self.ignored_labels = np.array([])
        # Expose class count on the dataset for validation
        self.num_classes = len(self.label_to_names)

        base = getattr(config, 'data_root', None) or os.environ.get('DILAB_DATA_ROOT') or '/home/stud/nguyenti/storage/user/EARLy/datav2_final'
        
        # Validate base path exists
        if not os.path.exists(base):
            raise FileNotFoundError(
                f"Data root directory not found: {base}\n"
                f"Please check the --data_path argument or DILAB_DATA_ROOT environment variable."
            )
        
        if not os.path.isdir(base):
            raise NotADirectoryError(f"Data root is not a directory: {base}")
        
        split_dir = {'training': 'train', 'validation': 'val', 'test': 'test'}[self.set]
        self.path = os.path.join(base, split_dir)
        
        # Validate split directory exists before accessing
        if not os.path.isdir(self.path):
            raise FileNotFoundError(
                f"Data directory not found: {self.path}\n"
                f"Preprocess data first: python presample_kpconv.py --data_root <input> --out_root {base} --target_points 16384"
            )
        
        # Safe file listing with error handling
        try:
            self.files = sorted([
                os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.npy')
            ])
        except OSError as e:
            raise FileNotFoundError(
                f"Cannot access directory {self.path}: {e}\n"
                f"Please ensure the directory exists and is readable."
            )
        
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {self.path}")

        config.num_classes = len(self.label_to_names)
        config.dataset_task = 'cloud_segmentation'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cloud = np.load(self.files[idx])
        xyz = cloud[:, :3].astype(np.float32)
        labels = cloud[:, 3].astype(np.int64)

        # Merge external Noise label 13 into in-vocabulary Noise label 11
        if np.any(labels == 13):
            labels[labels == 13] = 11
        # Keep labels strictly within 0..11
        labels = np.clip(labels, 0, len(self.label_to_names) - 1)
        # Final validation - ensure all labels are in valid range
        assert labels.min() >= 0, f"Labels below 0 found: {labels.min()}"
        assert labels.max() < len(self.label_to_names), f"Labels above {len(self.label_to_names)-1} found: {labels.max()}"

        features = np.ones((xyz.shape[0], 1), dtype=np.float32)
        return {'points': xyz, 'features': features, 'labels': labels, 'cloud_inds': np.array([idx] * xyz.shape[0], dtype=np.int32)}


class KPConvBatch:
    def __init__(self,
                 points_per_layer: List[torch.Tensor],
                 neighbors: List[torch.Tensor],
                 pools: List[torch.Tensor],
                 upsamples: List[torch.Tensor],
                 stack_lengths: List[torch.Tensor],
                 features: torch.Tensor,
                 labels: torch.Tensor):
        self.points = points_per_layer
        self.neighbors = neighbors
        self.pools = pools
        self.upsamples = upsamples
        self.lengths = stack_lengths
        self.features = features
        self.labels = labels

    def to(self, device):
        self.points = [p.to(device) for p in self.points]
        self.neighbors = [n.to(device) for n in self.neighbors]
        self.pools = [p.to(device) for p in self.pools]
        self.upsamples = [u.to(device) for u in self.upsamples]
        self.lengths = [l.to(device) for l in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        return self


_collate_printed_once = False

def DiLabCollate(batch_list: List[dict]):
    global _collate_printed_once
    t_all = time.time()
    _logger.info(f"DiLabCollate START: batch_size={len(batch_list)}")
    
    # Stack base level
    _logger.info("Stacking base level points/features/labels...")
    base_points = np.concatenate([b['points'] for b in batch_list], axis=0).astype(np.float32)
    base_features = np.concatenate([b['features'] for b in batch_list], axis=0).astype(np.float32)
    base_labels = np.concatenate([b['labels'] for b in batch_list], axis=0).astype(np.int64)
    stack_lengths = np.asarray([b['points'].shape[0] for b in batch_list], dtype=np.int32)
    _logger.info(f"Base level stacked: {base_points.shape[0]} points, lengths={stack_lengths}")

    # Use runtime config if provided, else fallback to a local default
    cfg = _RUNTIME_CFG
    if cfg is None:
        from kpconv.config import DiLabConfig as _Cfg
        cfg = _Cfg()

    # Build pyramid
    r_normal = cfg.first_subsampling_dl * cfg.conv_radius
    arch = cfg.architecture

    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_stack_lengths = []

    stacked_points = base_points
    stacked_lengths = stack_lengths
    layer_blocks: List[str] = []
    collate_start_time = time.time()

    if VERBOSE and not _collate_printed_once:
        print(f'[DiLab] Collate start: B={len(batch_list)} N={base_points.shape[0]} r0={r_normal:.4f}', flush=True)
        _collate_printed_once = True
        _LAST_PRINT_TIME = time.time()  # Initialize rate limiter

    for li, block in enumerate(arch):
        if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
            layer_blocks.append(block)
            continue

        t_layer = time.time()
        deform_layer = any('deformable' in bl for bl in layer_blocks)
        radius = r_normal * (cfg.deform_radius / cfg.conv_radius if deform_layer else 1.0)
        
        layer_points = stacked_points.shape[0]
        _logger.info(f"Layer {li} ({block}): Starting conv neighbors, stacked_points={layer_points}, radius={radius:.4f}")
        
        # Don't print layer start - too noisy, will print on completion instead
        
        if layer_blocks:
            conv_i = _radius_neighbors(stacked_points, stacked_points, stacked_lengths, stacked_lengths, radius)
            _logger.info(f"Layer {li} ({block}): Conv neighbors completed")
        else:
            conv_i = np.zeros((0, 1), dtype=np.int64)
            _logger.info(f"Layer {li} ({block}): No layer blocks, skipping conv neighbors")

        if 'pool' in block or 'strided' in block:
            dl = 2 * r_normal / cfg.conv_radius
            _logger.info(f"Layer {li} ({block}): Subsampling with dl={dl:.4f}")
            pool_p, pool_b = _voxel_grid_subsample(stacked_points, stacked_lengths, voxel_size=dl)
            _logger.info(f"Layer {li} ({block}): Subsampled to {pool_p.shape[0]} points")
            r_pool = r_normal * (cfg.deform_radius / cfg.conv_radius if deform_layer else 1.0)
            
            _logger.info(f"Layer {li} ({block}): Starting pool neighbors")
            pool_i = _radius_neighbors(pool_p, stacked_points, pool_b, stacked_lengths, r_pool)
            _logger.info(f"Layer {li} ({block}): Pool neighbors completed")
            
            _logger.info(f"Layer {li} ({block}): Starting upsample neighbors")
            up_i = _radius_neighbors(stacked_points, pool_p, stacked_lengths, pool_b, 2 * r_normal)
            _logger.info(f"Layer {li} ({block}): Upsample neighbors completed")
        else:
            pool_i = np.zeros((0, 1), dtype=np.int64)
            pool_p = np.zeros((0, 3), dtype=np.float32)
            pool_b = np.zeros((0,), dtype=np.int32)
            up_i = np.zeros((0, 1), dtype=np.int64)

        input_points.append(torch.from_numpy(stacked_points.copy()))
        input_neighbors.append(torch.from_numpy(conv_i.astype(np.int64)))
        input_pools.append(torch.from_numpy(pool_i.astype(np.int64)))
        input_upsamples.append(torch.from_numpy(up_i.astype(np.int64)))
        input_stack_lengths.append(torch.from_numpy(stacked_lengths.copy()))

        stacked_points = pool_p
        stacked_lengths = pool_b
        # Don't print layer completion - too noisy, only print collate summary
        r_normal *= 2
        layer_blocks = []

        if 'global' in block or 'upsample' in block:
            break

    _logger.info("Converting to tensors...")
    features_t = torch.from_numpy(base_features)
    labels_t = torch.from_numpy(base_labels)

    total_time = time.time() - t_all
    _logger.info(f"DiLabCollate COMPLETE: took {total_time:.2f}s")
    
    # Don't print - training script already shows progress

    return KPConvBatch(
        points_per_layer=input_points,
        neighbors=input_neighbors,
        pools=input_pools,
        upsamples=input_upsamples,
        stack_lengths=input_stack_lengths,
        features=features_t,
        labels=labels_t,
    )