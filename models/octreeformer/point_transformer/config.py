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
    fast_debug = True

    # Model settings
    model_name = 'point_transformer_v1'
    feature_dim = 3
    num_classes = 12

    # Training settings
    epochs = 100
    batch_size = 32
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
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.base_data_root = '../../dataset/datav2_final'

        if self.use_preprocessed:
            self.sample_ratio = 1.0
            self.npoints_batch = 2048
            self.train_data_root = os.path.join(
                self.base_data_root, 'train', f"datav2_{self.real_ratio}_octree_fps"
            )
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