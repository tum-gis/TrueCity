# Point Transformer v1 (TrueCity Integration)

This directory contains the **Point Transformer v1** implementation for 3D point cloud segmentation, migrated from the `tum-di-lab` repository into the TrueCity project as a standalone model.

It is designed to reproduce the reported results:

| Model                | 100S-0R mIoU / OA | 75S-25R mIoU / OA | 50S-50R mIoU / OA | 25S-75R mIoU / OA | 0S-100R mIoU / OA |
|----------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| Point Transformer v1 | 16.30 / 57.54     | 19.79 / 60.29     | 23.43 / 67.54     | 24.66 / 68.70     | 28.89 / 67.98     |

## Structure

```
models/point_transformer/
├── train.py                     # Training entrypoint
├── test.py                      # Evaluation script
├── test_new.py                  # Additional eval / export utilities
├── model.py                     # Model definition + metrics helpers
├── config.py                    # Config class and get_config()
├── data_utils.py                # Dataloaders, transforms, normalization
├── compute_normalization_stats.py
├── augmentations.py             # Geometric/data augmentations
├── libs/                        # Custom CUDA/C++ ops
│   ├── pointops/
│   ├── pointops2/
│   ├── pointgroup_ops/
│   └── pointseg/
├── environment.yml              # Conda environment (this module only)
└── README.md
```

## Environment Setup

Create and activate the dedicated environment:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer

conda env create -f environment.yml
conda activate point_transformer
```

### Build Custom CUDA Ops

Inside the `point_transformer` environment, build the required CUDA/C++ extensions:

```bash
conda activate point_transformer

cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer/libs/pointops
python setup.py install

cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer/libs/pointops2
python setup.py install

cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer/libs/pointgroup_ops
python setup.py install

# Only needed if you use pointseg-specific functionality
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer/libs/pointseg
python setup.py install
```

> Note: These builds require a working CUDA toolchain (nvcc) compatible with the PyTorch version in the environment.

## Data Expectations

Point Transformer v1 expects preprocessed octree-FPS blocks under the **EARLy datav2_final** directory:

```
/home/stud/nguyenti/storage/user/EARLy/datav2_final/
├── train/
│   ├── datav2_0_octree_fps/
│   ├── datav2_25_octree_fps/
│   ├── datav2_50_octree_fps/
│   ├── datav2_75_octree_fps/
│   └── datav2_100_octree_fps/
├── val/
└── test/
```

Each `.npy` file has shape `[N, 4]`:
- Columns 0–2: `x, y, z` coordinates (`float32`)
- Column 3: class label

## Training Commands (5 Regimes)

All commands are run from:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer
conda activate point_transformer
```

The `train.py` script supports overriding the base data root and `real_ratio` via CLI arguments. The defaults still match the original code; these overrides only change paths.

### Arguments

- `--base_data_root`: Path to `datav2_final` root (EARLy data).
- `--real_ratio`: Real/synthetic ratio (0, 25, 50, 75, 100).

### 100S-0R (100% Synthetic, 0% Real)

```bash
python train.py \
  --base_data_root /home/stud/nguyenti/storage/user/EARLy/datav2_final \
  --real_ratio 100
```

### 75S-25R

```bash
python train.py \
  --base_data_root /home/stud/nguyenti/storage/user/EARLy/datav2_final \
  --real_ratio 75
```

### 50S-50R

```bash
python train.py \
  --base_data_root /home/stud/nguyenti/storage/user/EARLy/datav2_final \
  --real_ratio 50
```

### 25S-75R

```bash
python train.py \
  --base_data_root /home/stud/nguyenti/storage/user/EARLy/datav2_final \
  --real_ratio 25
```

### 0S-100R (0% Synthetic, 100% Real)

```bash
python train.py \
  --base_data_root /home/stud/nguyenti/storage/user/EARLy/datav2_final \
  --real_ratio 0
```

The configuration logic in `config.py` then maps these to:

```python
train_data_root = os.path.join(base_data_root, "train", f"datav2_{real_ratio}_octree_fps")
val_data_root   = os.path.join(base_data_root, "val")
test_data_root  = os.path.join(base_data_root, "test")
save_path       = f"experiments/{model_name}_logs/model_{real_ratio}"
```

## Evaluation

After training finishes, use `test.py` to evaluate a specific checkpoint:

```bash
conda activate point_transformer

python test.py \
  --checkpoint experiments/point_transformer_v1_logs/model_75/model/model_best.pth
```

This writes `test_log.txt` next to the checkpoint with per-class IoU/accuracy and overall metrics (mIoU, OA).

## Results & Logs

Runs are stored under:

```
experiments/point_transformer_v1_logs/model_<real_ratio>/
└── model/
    ├── model_best.pth
    ├── model_last.pth
    ├── training_log.txt
    └── test_log.txt
```

These logs contain the final **mIoU** and **OA** used to populate the result table at the top of this file.

## Notes

- Logic is preserved from the original `tum-di-lab/point_transformer` code; only paths, environment, and CLI overrides have been added for easier use inside TrueCity.
- Custom CUDA ops must be built inside the `point_transformer` environment before training.
- Random seed and training configuration are controlled via `config.py` and remain unchanged by this migration.


