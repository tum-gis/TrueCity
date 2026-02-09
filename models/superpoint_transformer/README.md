# Superpoint Transformer for Tumdilab Dataset

This repository contains code for training Superpoint Transformer on the tumdilab dataset for semantic segmentation of urban 3D point clouds.

**Based on:** [https://github.com/drprojects/superpoint_transformer](https://github.com/drprojects/superpoint_transformer)

We gratefully acknowledge the original authors for their excellent work on the Superpoint Transformer architecture.

## Installation

Run the installation script to set up the conda environment and install all dependencies:

```bash
./install.sh
```

This creates a conda environment named `spt` with all required dependencies.

**Optional:** If you need sparse convolution support (for advanced models), use:

```bash
./install.sh with_torchsparse
```

## Data Preparation

### Directory Structure

Place your data in the `data/tumdilab/raw/` directory with the following structure:

```
data/tumdilab/raw/
├── train/
│   └── datav2_100_octree_fps/    # Training data (100% real data ratio)
│       ├── block_000001.npy
│       ├── block_000002.npy
│       └── ...
├── val/
│   ├── val_000001.npy
│   ├── val_000002.npy
│   └── ...
└── test/
    ├── test_000001.npy
    ├── test_000002.npy
    └── ...
```

**Note:** The training subdirectory name follows the pattern `datav2_{real_ratio}_octree_fps`, where `real_ratio` can be adjusted in the configuration (see below).

### Data Format

Each `.npy` file should be a NumPy array with shape `[N, 4+]` where:
- `N` is the number of points
- Columns: `[X, Y, Z, semantic_label, optional_features...]`
- **semantic_label**: Integer values 0-11 for the 12 semantic classes (values outside this range are treated as "ignored")

### Semantic Classes

The tumdilab dataset supports 12 semantic classes:

| Class ID | Class Name           |
|----------|----------------------|
| 0        | RoadSurface          |
| 1        | GroundSurface        |
| 2        | RoadInstallations    |
| 3        | Vehicle              |
| 4        | Pedestrian           |
| 5        | WallSurface          |
| 6        | RoofSurface          |
| 7        | Door                 |
| 8        | Window               |
| 9        | BuildingInstallation |
| 10       | Tree                 |
| 11       | Noise                |

## Configuration

Training parameters can be configured in `configs/datamodule/semantic/tumdilab.yaml`. Key parameters you may want to adjust:

### Data Selection
- **`real_ratio`**: Percentage of real data (default: 100). This affects the training directory name (`datav2_{real_ratio}_octree_fps`)

### Preprocessing
- **`voxel`**: Voxel size for downsampling (default: 0.05)
- **`pcp_regularization`**: Regularization for partition levels (default: [0.1, 0.2, 0.3])
- **`pcp_spatial_weight`**: Spatial importance in partition (default: [1e-1, 1e-2, 1e-3])
- **`pcp_cutoff`**: Minimum superpoint size (default: [10, 30, 100])

### Training
- **`dataloader.batch_size`**: Training batch size (default: 4)
- **`sample_graph_r`**: Radius for spherical sampling during training (default: 50)
- **`sample_graph_k`**: Number of spherical samples per batch (default: 4)

### Augmentation
- **`pos_jitter`**: Position jittering magnitude (default: 0.05)
- **`tilt_n_rotate_phi`**: Tilt angle range (default: 0.1)
- **`tilt_n_rotate_theta`**: Rotation angle range in degrees (default: 180)
- **`anisotropic_scaling`**: Anisotropic scaling factor (default: 0.2)

## Training

To start training on the tumdilab dataset:

```bash
python src/train.py experiment=semantic/tumdilab
```

### Training Outputs

All training outputs are saved in timestamped directories under `logs/train/runs/`:

```
logs/train/runs/YYYY-MM-DD_HH-MM-SS/
├── checkpoints/
│   ├── last.ckpt           # Most recent checkpoint
│   └── epoch_XXX.ckpt      # Best checkpoint (highest validation mIoU)
├── train.log               # Training log file
└── wandb/                  # Weights & Biases logs
```

- **Checkpoints**: The best model is automatically selected based on validation mIoU
- **Training logs**: Detailed logs are saved to `train.log`
- **WandB**: Training metrics are synced to Weights & Biases under the project name `spt_tumdilab`

## Evaluation

To evaluate a trained model on the test set:

```bash
python src/eval.py experiment=semantic/tumdilab ckpt_path=/path/to/checkpoint.ckpt
```

Example with a specific checkpoint:

```bash
python src/eval.py experiment=semantic/tumdilab ckpt_path=logs/train/runs/2026-02-09_14-30-00/checkpoints/last.ckpt
```
