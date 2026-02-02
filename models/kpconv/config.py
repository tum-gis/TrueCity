# filepath: utils/config.py

import os
import json
import sys

# Ensure upstream KPConv config base is importable
sys.path.append('/home/stud/nguyenti/storage/user/tum-di-lab/EARLy_notebooks/kpconv/_kpconv_upstream')
from utils.config import Config as UpstreamConfig


class DiLabConfig(UpstreamConfig):
    # Dataset parameters
    dataset = 'DiLab'
    num_classes = 12  # Corrected to match actual dataset (labels 1-12 mapped to 0-11)
    in_features_dim = 1

    # Training parameters
    max_epoch = 2
    batch_num = 2  # Reduced from 32 to prevent OOM (CUDA memory limit ~15GB)
    input_threads = 1  # Reduced from 4 to prevent OOM (KDTree is memory-intensive)
    learning_rate = 1e-3  # Default value, will be overridden by --lr argument
    lr_decays = {50: 0.1, 80: 0.01}
    saving = True
    saving_path = None
    checkpoint_gap = 10
    grad_clip_norm = 10.0  # Reduced gradient clipping for stability
    deform_lr_factor = 0.01  # Reduced deformation learning rate
    momentum = 0.98
    weight_decay = 1e-3

    # Deformation parameters for numerical stability
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 0.1  # Reduced deformation power for stability
    repulse_extent = 1.2
    # Denser, more stable neighborhoods
    first_subsampling_dl = 0.12  # Increased from 0.06 to reduce memory (fewer points in first layer)
    max_in_points = 200000  # Limit points per batch to prevent OOM (0 = no limit)
    KP_extent = 1.5
    conv_radius = 3.0
    deform_radius = 5.0
    deform_warmup_epochs = 3  # Disable deformable updates for first N epochs
    
    # Batch normalization for stability
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Model parameters
    architecture = [
        'simple', 'resnetb', 'resnetb_strided', 'resnetb',
        'resnetb', 'resnetb_strided', 'resnetb', 'resnetb',
        'resnetb_strided', 'resnetb_deformable', 'resnetb_deformable',
        'resnetb_deformable_strided', 'resnetb_deformable', 'resnetb_deformable',
        'nearest_upsample', 'unary', 'nearest_upsample', 'unary',
        'nearest_upsample', 'unary', 'nearest_upsample', 'unary'
    ]

    def __init__(self):
        super().__init__()

    def save(self):
        if self.saving_path is None:
            return
        cfg_path = os.path.join(self.saving_path, 'config.json')
        with open(cfg_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)