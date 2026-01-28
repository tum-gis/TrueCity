"""
Configuration settings for point cloud segmentation
"""

import os
import time

class Config:
    # Data settings
    base_data_root = '../../dataset/2d-area-based-split-v2/'
    real_ratio = 100  # unused when base_data_root points to a specific datav2_XXX_octree_fps dir
    sample_ratio = 0.0001  # 1.0 means use all data
    npoints_batch = 2048
    use_preprocessed = True   # True = use 2048-blocks, False = cut from raw point cloud
    fast_debug = False
    # Label mapping: merge rare 13th label (id=12) into class id 11
    merge_label_from = 12
    merge_label_to = 11
    # Voxel size used in collate (meters)
    voxel_size = 0.02

    # Model settings
    model_name = 'point_transformer_v3'
    feature_dim = 3
    num_classes = 12
    # Serialized transformer regularization
    drop_path = 0.1
    enable_flash = False

    # Training settings
    epochs = 100
    batch_size = 32
    base_lr = 0.001
    momentum = 0.9
    weight_decay = 0.01
    learning_rate_clip = 1e-5
    workers = 4
    label_smoothing = 0.0
    head_dropout = 0.1

    # Scheduler settings
    scheduler_type = 'multistep'
    milestones = [int(epochs * 0.6), int(epochs * 0.8)]
    gamma = 0.1

    # Other settings
    ignore_label = 255
    print_freq = 1
    save_freq = 1
    eval_freq = 1
    evaluate = True
    manual_seed = 7777

    # Class names
    class_names = [
        'RoadSurface', 'GroundSurface', 'RoadInstallations',
        'Vehicle', 'Pedestrian', 'WallSurface', 'RoofSurface', 
        'Door', 'Window', 'BuildingInstallation', 'Tree', 'Noise'
    ]

    # Class weights
    calculate_class_weights = True
    use_class_weights = False
    class_weights = None

    # Evaluation-time label merging (e.g., treat Vehicle/Pedestrian as Noise)
    eval_merge_label_from = [3, 4]
    eval_merge_label_to = 11

    def __init__(self, **kwargs):
        passed_keys = set(kwargs.keys())
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # allow dynamic attrs like save_root
                setattr(self, key, value)

        # Default base_data_root if not provided via kwargs
        if 'base_data_root' not in passed_keys:
            candidates = [
                '/home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree',  # tum-di-lab structure
                '/home/stud/nguyenti/storage/user/tum-di-lab/datav2_final',  # tum-di-lab alternative
                '/home/stud/nguyenti/storage/user/EARLy/datav2_final',  # EARLy structure
                '../EARLy/datav2_final',          # when running from repo root
                '../../EARLy/datav2_final',       # when running from point_transformer_v3 dir
            ]
            chosen = None
            for c in candidates:
                if os.path.isdir(c):
                    chosen = c
                    break
            self.base_data_root = chosen or candidates[0]

        # Resolve dataset split directories
        if self.use_preprocessed:
            self.sample_ratio = 1.0
            self.npoints_batch = 2048

            # Case 0: base_data_root already points to a specific dataset root that contains train/val/test
            direct_train = os.path.join(self.base_data_root, 'train')
            direct_val = os.path.join(self.base_data_root, 'val')
            direct_test = os.path.join(self.base_data_root, 'test')
            if os.path.isdir(direct_train) and os.path.isdir(direct_val) and os.path.isdir(direct_test):
                self.train_data_root = direct_train
                self.val_data_root = direct_val
                self.test_data_root = direct_test
            else:
                # Support multiple layouts:
                # 1) base/train/datav2_XXX_octree_fps (datav2_final structure)
                # 2) base/datav2_XXX_octree_fps/train (datav2_final alternative)
                # 3) base/data_XXX_octree/train (datav2_octree structure)
                dataset_dirname_fps = f"datav2_{self.real_ratio}_octree_fps"
                dataset_dirname_octree = f"data_{self.real_ratio}_octree"
                
                # Try datav2_final structure first
                candidate_a_train = os.path.join(self.base_data_root, 'train', dataset_dirname_fps)
                candidate_b_root = os.path.join(self.base_data_root, dataset_dirname_fps)
                candidate_b_train = os.path.join(candidate_b_root, 'train')
                
                # Try datav2_octree structure
                candidate_c_root = os.path.join(self.base_data_root, dataset_dirname_octree)
                candidate_c_train = os.path.join(candidate_c_root, 'train')
                
                if os.path.isdir(candidate_c_train):
                    # Structure 3: datav2_octree/data_{ratio}_octree/train
                    self.train_data_root = candidate_c_train
                    self.val_data_root = os.path.join(candidate_c_root, 'val')
                    self.test_data_root = os.path.join(candidate_c_root, 'test')
                elif os.path.isdir(candidate_b_train):
                    # Structure 2: base/datav2_{ratio}_octree_fps/train
                    self.train_data_root = candidate_b_train
                    self.val_data_root = os.path.join(candidate_b_root, 'val')
                    self.test_data_root = os.path.join(candidate_b_root, 'test')
                else:
                    # Structure 1: base/train/datav2_{ratio}_octree_fps (fallback)
                    self.train_data_root = candidate_a_train
                    self.val_data_root = os.path.join(self.base_data_root, 'val')
                    self.test_data_root = os.path.join(self.base_data_root, 'test')
        else:
            self.train_data_root = os.path.join(self.base_data_root, 'train')
            self.val_data_root = os.path.join(self.base_data_root, 'val')
            self.test_data_root = os.path.join(self.base_data_root, 'test')

        # Allow overriding split roots explicitly via kwargs
        if 'train_data_root' in passed_keys:
            self.train_data_root = kwargs['train_data_root']
        if 'val_data_root' in passed_keys:
            self.val_data_root = kwargs['val_data_root']
        if 'test_data_root' in passed_keys:
            self.test_data_root = kwargs['test_data_root']

        # Save root: prefer kwarg save_root, then env SAVE_ROOT, then default to TrueCity results
        provided_root = getattr(self, 'save_root', None)
        env_root = os.environ.get('SAVE_ROOT', None)
        save_root = provided_root or env_root or '/home/stud/nguyenti/storage/user/TrueCity/results'
        # Derive dataset name from train_data_root (e.g., .../datav2_100_octree_fps/train -> datav2_100_octree_fps)
        train_root_norm = os.path.normpath(self.train_data_root)
        base_train_dir = os.path.basename(train_root_norm)
        if base_train_dir.lower() == 'train':
            dataset_name = os.path.basename(os.path.dirname(train_root_norm))
        else:
            dataset_name = base_train_dir
        ts = time.strftime('%Y%m%d_%H%M%S')
        run_dirname = f"ptv3_{dataset_name}_{ts}"
        self.save_path = os.path.join(save_root, run_dirname)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'model'), exist_ok=True)


def get_config(**kwargs):
    return Config(**kwargs)