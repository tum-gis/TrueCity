"""
Configuration settings for point cloud segmentation
"""

import os

class Config:
    # Data settings
    base_data_root = '../../dataset/2d-area-based-split-v2/'
    real_ratio = 75
    sample_ratio = 0.0001  # 1.0 means use all data
    npoints_batch = 2048
    use_preprocessed = True   # True = use 2048-blocks, False = cut from raw point cloud
    fast_debug = False  # Set to True only for quick testing (limits to 2 batches)

    # Model settings
    model_name = 'point_transformer_v1'
    feature_dim = 3
    num_classes = 12

    # Training settings
    epochs = 100
    batch_size = 8  # Reduced from 32 to save memory (Point Transformer is very memory-intensive)
                     # With batch_size=8: 8 * 2048 = 16,384 points per batch (much more manageable)
                     # Use --batch_size to override if you have more GPU memory
    base_lr = 0.1
    momentum = 0.9
    weight_decay = 0.0001
    learning_rate_clip = 1e-5

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

    def __init__(self, **kwargs):
        # Apply any overrides passed in (e.g. base_data_root, real_ratio)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # If base_data_root was not overridden, fall back to the default
        if "base_data_root" not in kwargs:
        self.base_data_root = '../../dataset/datav2_final'

        if self.use_preprocessed:
            self.sample_ratio = 1.0
            self.npoints_batch = 2048
            # Check multiple possible structures:
            # 1) datav2_final/train/datav2_{ratio}_octree_fps/ (original structure)
            # 2) datav2_octree/data_{ratio}_octree/ (tum-di-lab structure)
            # 3) base_data_root is already pointing to data_{ratio}_octree/
            
            potential_fps_dir = os.path.join(
                self.base_data_root, 'train', f"datav2_{self.real_ratio}_octree_fps"
            )
            potential_data_ratio_dir = os.path.join(
                self.base_data_root, f"data_{self.real_ratio}_octree"
            )
            
            # Also check if base_data_root itself is datav2_octree and we need to go into it
            potential_datav2_octree = os.path.join(self.base_data_root, 'datav2_octree') if 'datav2_octree' not in self.base_data_root else self.base_data_root
            if os.path.isdir(potential_datav2_octree) and potential_datav2_octree != self.base_data_root:
                potential_data_ratio_dir_v2 = os.path.join(
                    potential_datav2_octree, f"data_{self.real_ratio}_octree"
                )
            else:
                potential_data_ratio_dir_v2 = None
            
            if os.path.exists(potential_fps_dir):
                # Structure 1: base_data_root/train/datav2_{ratio}_octree_fps/
                self.train_data_root = potential_fps_dir
                self.val_data_root = os.path.join(self.base_data_root, 'val')
                self.test_data_root = os.path.join(self.base_data_root, 'test')
            elif os.path.exists(potential_data_ratio_dir):
                # Structure 2: base_data_root/data_{ratio}_octree/ (tum-di-lab when base_data_root=datav2_octree)
                self.train_data_root = os.path.join(potential_data_ratio_dir, 'train')
                self.val_data_root = os.path.join(potential_data_ratio_dir, 'val')
                self.test_data_root = os.path.join(potential_data_ratio_dir, 'test')
            elif potential_data_ratio_dir_v2 and os.path.exists(potential_data_ratio_dir_v2):
                # Structure 2 variant: base_data_root/datav2_octree/data_{ratio}_octree/
                self.train_data_root = os.path.join(potential_data_ratio_dir_v2, 'train')
                self.val_data_root = os.path.join(potential_data_ratio_dir_v2, 'val')
                self.test_data_root = os.path.join(potential_data_ratio_dir_v2, 'test')
            else:
                # Fallback: assume base_data_root is already data_{ratio}_octree/ or points to train/val/test directly
                self.train_data_root = os.path.join(self.base_data_root, 'train')
            self.val_data_root = os.path.join(self.base_data_root, 'val')
            self.test_data_root = os.path.join(self.base_data_root, 'test')
        else:
            self.train_data_root = os.path.join(self.base_data_root, 'train')
            self.val_data_root = os.path.join(self.base_data_root, 'val')
            self.test_data_root = os.path.join(self.base_data_root, 'test')

        self.save_path = f"experiments/{self.model_name}_logs/model_{self.real_ratio}"
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'model'), exist_ok=True)

def get_config(**kwargs):
    return Config(**kwargs)