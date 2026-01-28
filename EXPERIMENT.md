# Experiment Guide

## 📋 Table of Contents

1. [Project Architecture](#project-architecture)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Training PointNet](#training-pointnet)
5. [Training PointNet2++](#training-pointnet2)
6. [Training OctreeFormer](#training-octreeformer)
7. [Evaluation](#evaluation)
8. [Advanced Configuration](#advanced-configuration)
9. [Troubleshooting](#troubleshooting)

## 🏗️ Project Architecture

### Project Structure

```
TrueCity/
├── models/                 # Model implementations (each model is self-contained)
│   ├── pointnet/          # PointNet and PointNet2++ (standalone)
│   │   ├── src/           # Model source code
│   │   │   ├── models/     # Model definitions (pointnet.py, pointnet2.py)
│   │   │   ├── training/  # Training logic (pointnet_trainer.py, pointnet2_trainer.py)
│   │   │   └── metrics.py # Evaluation metrics
│   │   └── scripts/       # Training scripts
│   │       ├── train_pointnet.py
│   │       └── train_pointnet2.py
│   ├── point_transformer/ # Point Transformer v1 (standalone)
│   │   ├── train.py       # Training script
│   │   ├── test.py        # Testing script
│   │   ├── model.py       # Model definition
│   │   ├── config.py      # Configuration
│   │   ├── data_utils.py  # Data loading utilities
│   │   ├── libs/          # CUDA operations (pointops, pointops2, etc.)
│   │   └── environment.yml # Conda environment
│   └── octreeformer/      # OctreeFormer (standalone, separate environment)
│       ├── train.py       # Training script
│       ├── test.py        # Testing script
│       ├── model.py       # Model wrapper
│       ├── octformer.py   # Core OctFormer
│       ├── octformerseg.py # Segmentation head
│       ├── point_transformer/ # Point transformer utilities
│       └── environment.yml # Conda environment
├── shared/                 # Shared utilities (used by pointnet models)
│   ├── data/              # Shared data utilities
│   │   ├── dataloader.py  # DataLoader creation
│   │   └── dataset.py     # Dataset class
│   └── utils/             # Shared utilities
│       ├── logging.py     # Logging setup
│       └── fps.py         # FPS sampling functions
├── scripts/               # Root-level scripts (experimental/legacy)
│   └── train_superpoint.py # Superpoint Transformer (experimental, requires external dependencies)
├── tools/                 # Utility scripts
├── examples/              # Example usage
├── docs/                  # Additional documentation
├── config/                 # Configuration files
├── results/               # Training results and checkpoints
├── README.md
├── EXPERIMENT.md
└── requirements.txt
```

### Data Flow

#### Training Pipeline

1. **Data Loading** (`shared/data/dataloader.py` for PointNet, model-specific for others)
   - Load point cloud files from disk
   - Apply FPS sampling (if not precomputed)
   - Normalize coordinates
   - Filter invalid classes
   - Create PyTorch DataLoaders

2. **Model Creation** (`models/*/src/models/` for PointNet, model-specific for others)
   - Initialize model architecture
   - Setup loss function
   - Move to GPU if available

3. **Training Loop** (`models/*/src/training/*_trainer.py` for PointNet, model-specific for others)
   - Forward pass through model
   - Compute loss
   - Backward pass and optimization
   - Validation evaluation
   - Checkpoint saving

4. **Evaluation** (`models/*/src/training/metrics.py` for PointNet, model-specific for others)
   - Compute comprehensive metrics
   - Per-class IoU, accuracy, precision, recall, F1
   - Overall accuracy and mean IoU

### Model Architectures

#### PointNet (`models/pointnet/src/models/pointnet.py`)
- **Encoder**: STN3d → Conv layers → Max pooling
- **Decoder**: Feature concatenation → Conv layers → Classification
- **Features**: Feature transformation for rotation invariance

#### PointNet2++ (`models/pointnet/src/models/pointnet2.py`)
- **Set Abstraction**: Hierarchical point sampling and grouping
- **Feature Propagation**: Upsampling with interpolation
- **Features**: Multi-scale feature learning

#### Point Transformer v1 (`models/point_transformer/`)
- **Architecture**: Transformer-based point cloud segmentation
- **Features**: Self-attention mechanisms, relative position encoding
- **Environment**: Separate conda environment (`point_transformer`)
- **CUDA Ops**: Custom CUDA operations for efficient computation

#### OctreeFormer (`models/octreeformer/`)
- **Backbone**: OctFormer with hierarchical transformer blocks on octree structure
- **Head**: Segmentation head with FPN-style feature fusion
- **Features**: Octree-based hierarchical processing with attention mechanisms
- **Environment**: Separate conda environment (`octreeformer`)

### Module Dependencies

**PointNet/PointNet2++**:
```
models/pointnet/scripts/
  └── train_*.py
      ├── models/pointnet/src/training/*_trainer.py
      │   ├── models/pointnet/src/models/*.py
      │   ├── shared/data/dataloader.py
      │   ├── models/pointnet/src/training/metrics.py
      │   └── shared/utils/logging.py
      └── shared/utils/logging.py
```

**Point Transformer & OctreeFormer**:
- Standalone models with local imports only
- No dependencies on root-level `src/` or `shared/` folders

### Key Design Decisions

1. **Self-Contained Models**: Each model is in its own directory with its own dependencies
2. **Shared Utilities**: PointNet models use `shared/` folder for common utilities
3. **Separate Environments**: Point Transformer and OctreeFormer use separate conda environments
4. **Preserved Logic**: All original functionality maintained during migration
5. **Extensible**: Easy to add new models as standalone directories

### Import Paths

**PointNet/PointNet2++**:
- Scripts use: `from src.training.pointnet_trainer import ...` (relative to `models/pointnet/`)
- Trainers use: `from shared.data.dataloader import ...` (from project root)
- Models use: `from src.models.pointnet import ...` (relative to `models/pointnet/`)

**Point Transformer & OctreeFormer**:
- All imports are local: `from config import ...`, `from model import ...`

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM (32GB+ recommended for large datasets)
- Conda or Miniconda (for conda environment setup)

### Option 1: Conda Environment (Recommended)

The easiest way to set up TrueCity is using the provided conda environment:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity

# Create conda environment from environment.yml
conda env create -n truecity -f environment.yml

# Activate the environment
conda activate truecity

# Verify installation
python -c "import torch; import pytorch_lightning; import hydra; print('All dependencies OK!')"
```

**Note**: If you need CUDA support, you may need to install PyTorch with CUDA separately after creating the environment:

```bash
conda activate truecity
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**To use the environment:**
```bash
# Always activate before running training
conda activate truecity

# Run training scripts
cd models/pointnet
python scripts/train_pointnet.py --mode train --data_path /path/to/data
python scripts/train_pointnet2.py --mode train --data_path /path/to/data

# Note: Superpoint Transformer is experimental and may require additional setup
# python ../scripts/train_superpoint.py datamodule.data_dir=/path/to/data experiment=semantic/truecity
```

### Option 2: Pip Installation

Alternatively, you can install dependencies using pip:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity

# Install core dependencies
pip install -r requirements.txt

# For GPU support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Superpoint Transformer dependencies (if using conda env)
conda activate truecity
python -c "import pytorch_lightning; import hydra; print('Superpoint Transformer dependencies OK!')"
```

## 📊 Data Preparation

### Data Structure

Organize your data in the following structure:

```
data/
├── train/
│   ├── file1.npy
│   ├── file2.npy
│   └── ...
├── val/
│   ├── file1.npy
│   └── ...
└── test/
    ├── file1.npy
    └── ...
```

### Data Format

Each `.npy` file should contain a point cloud array with shape `[N, 4]`:
- Columns 0-2: `[x, y, z]` coordinates (float32)
- Column 3: `object_class` label (int or float)

Example:
```python
import numpy as np
# Shape: [N, 4]
data = np.array([
    [1.2, 3.4, 0.1, 0],  # Point with class 0
    [2.1, 1.8, 0.3, 1],  # Point with class 1
    [-0.5, 2.1, 1.2, 0], # Point with class 0
    ...
], dtype=np.float32)
np.save('train/file1.npy', data)
```

### Data Preprocessing Setup

Before training, verify your data is properly preprocessed:

1. **Check Data Availability**:
```bash
# Base data directory
EARLY_DATA="/home/stud/nguyenti/storage/user/EARLy/datav2_final"

# Verify data splits exist
ls -la $EARLY_DATA/datav2_*_octree_fps
```

2. **Verify Data Structure**:
Each data split should have `train/`, `val/`, and `test/` subdirectories (or files matching these patterns).

3. **Class Mapping**:
The training automatically uses a fixed 12-class mapping: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` (all classes except 13) to ensure consistency across all splits.

### Optional: Precompute FPS Samples

For faster training, precompute FPS samples:

```python
# FPS preprocessing is handled automatically by the data loaders
# For custom preprocessing, see shared/utils/fps.py
```

Then use the precomputed data path for training.

## 🚀 Training PointNet

### Exact Training Commands (From Successful Experiments)

These are the exact hyperparameters used to achieve the results in the paper table:

#### 100S-0R (100% Synthetic, 0% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet

python scripts/train_pointnet.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_100_octree_fps \
    --n_points 2048 \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --model_path pointnet_datav2_100.pth
```

#### 75S-25R (75% Synthetic, 25% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_75_octree_fps \
    --n_points 2048 \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --model_path pointnet_datav2_75.pth
```

#### 50S-50R (50% Synthetic, 50% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_50_octree_fps \
    --n_points 2048 \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --model_path pointnet_datav2_50.pth
```

#### 25S-75R (25% Synthetic, 75% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_25_octree_fps \
    --n_points 2048 \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --model_path pointnet_datav2_25.pth
```

#### 0S-100R (0% Synthetic, 100% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_0_octree_fps \
    --n_points 2048 \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --model_path pointnet_datav2_0.pth
```

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--n_points` | `2048` | Number of points per sample |
| `--batch_size` | `8` | Training batch size |
| `--num_epochs` | `50` | Number of training epochs |
| `--learning_rate` | `0.001` | Learning rate |
| `--sample_multiplier` | `1.0` | Sample multiplier (default) |

### Resume Training

If a checkpoint exists, training will automatically resume:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py \
    --mode train \
    --data_path /path/to/data \
    --model_path pointnet_model.pth \
    --num_epochs 100  # Will resume from last checkpoint
```

### Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `train` | Mode: `train`, `eval`, `test`, `analyze`, `checkpoints` |
| `--data_path` | Required | Path to data directory |
| `--model_path` | `ingolstadt_pointnet_segmentation.pth` | Path to save/load model |
| `--n_points` | `2048` | Number of points per sample |
| `--batch_size` | `8` | Training batch size |
| `--num_epochs` | `50` | Number of training epochs |
| `--learning_rate` | `0.001` | Learning rate |
| `--sample_multiplier` | `1.0` | Sample multiplier (1.0=normal, 2.5=high diversity) |
| `--ignore_labels` | `None` | Labels to ignore (e.g., `--ignore_labels 3 4`) |

## 🚀 Training PointNet2++

### Exact Training Commands (From Successful Experiments)

These are the exact hyperparameters used to achieve the results in the paper table:

#### 100S-0R (100% Synthetic, 0% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet

python scripts/train_pointnet2.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_100_octree_fps \
    --n_points 2048 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.00005 \
    --model_path pointnet2_datav2_100.pth
```

#### 75S-25R (75% Synthetic, 25% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet2.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_75_octree_fps \
    --n_points 2048 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.00005 \
    --model_path pointnet2_datav2_75.pth
```

#### 50S-50R (50% Synthetic, 50% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet2.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_50_octree_fps \
    --n_points 2048 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.00005 \
    --model_path pointnet2_datav2_50.pth
```

#### 25S-75R (25% Synthetic, 75% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet2.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_25_octree_fps \
    --n_points 2048 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.00005 \
    --model_path pointnet2_datav2_25.pth
```

#### 0S-100R (0% Synthetic, 100% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet2.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_0_octree_fps \
    --n_points 2048 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.00005 \
    --model_path pointnet2_datav2_0.pth
```

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--n_points` | `2048` | Number of points per sample |
| `--batch_size` | `32` | Training batch size |
| `--num_epochs` | `100` | Number of training epochs |
| `--learning_rate` | `0.00005` | Learning rate (5e-5) |

### Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `train` | Mode: `train`, `eval`, `test` |
| `--data_path` | Required | Path to data directory |
| `--model_path` | `pointnet2_segmentation.pth` | Path to save/load model |
| `--n_points` | `2048` | Number of points per sample |
| `--batch_size` | `32` | Training batch size |
| `--num_epochs` | `100` | Number of training epochs |
| `--learning_rate` | `0.00005` | Learning rate (lower than PointNet) |
| `--sample_multiplier` | `1.0` | Sample multiplier |
| `--ignore_labels` | `None` | Labels to ignore |

### PointNet2++ Specific Notes

- Uses precomputed FPS data by default (`is_precomputed=True`)
- Learning rate is typically lower (0.0001 vs 0.001)
- Creates experiment directories in `./log/pointnet2_sem_seg/`
- Saves checkpoints every 5 epochs

## 🚀 Training Point Transformer v1

### Overview

Point Transformer v1 is a transformer-based architecture for 3D point cloud segmentation. It uses self-attention mechanisms with relative position encoding to process point clouds efficiently.

**Key Features**:
- Transformer-based architecture with self-attention
- Relative position encoding
- Custom CUDA operations for efficient computation
- Separate conda environment for isolation

### Installation

Point Transformer v1 uses a separate conda environment:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate point_transformer

# Build CUDA operations
bash build_cuda_ops.sh
```

**Dependencies**:
- Python 3.8
- PyTorch (>=1.12.0)
- Custom CUDA ops: pointops, pointops2, pointgroup_ops, pointseg
- numpy, pandas, tqdm, scipy, scikit-learn

### Training Commands

#### Basic Training

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer
conda activate point_transformer

python train.py \
  --base_data_root /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree \
  --real_ratio 0 \
  --batch_size 8 \
  --accum_steps 4
```

#### Training for All Data Regimes

```bash
# 100S-0R (100% Synthetic, 0% Real)
python train.py --base_data_root /path/to/datav2_octree --real_ratio 100 --batch_size 8 --accum_steps 4

# 75S-25R
python train.py --base_data_root /path/to/datav2_octree --real_ratio 75 --batch_size 8 --accum_steps 4

# 50S-50R
python train.py --base_data_root /path/to/datav2_octree --real_ratio 50 --batch_size 8 --accum_steps 4

# 25S-75R
python train.py --base_data_root /path/to/datav2_octree --real_ratio 25 --batch_size 8 --accum_steps 4

# 0S-100R (0% Synthetic, 100% Real)
python train.py --base_data_root /path/to/datav2_octree --real_ratio 0 --batch_size 8 --accum_steps 4
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_data_root` | Required | Base path to data directory (e.g., `datav2_octree`) |
| `--real_ratio` | `75` | Real data ratio (0, 25, 50, 75, 100) |
| `--batch_size` | `8` | Training batch size (reduced for memory efficiency) |
| `--accum_steps` | `1` | Gradient accumulation steps (use 4 to simulate batch_size=32) |
| `--epochs` | `100` | Number of training epochs |
| `--base_lr` | `0.1` | Base learning rate |
| `--weight_decay` | `0.0001` | Weight decay for optimizer |

### Training Options

Additional options available:

| Parameter | Description |
|-----------|-------------|
| `--fast_debug` | Enable fast debug mode (2 batches, 2 epochs) |
| `--npoints_batch` | Number of points per batch item (default: 2048) |

### Model Architecture

Point Transformer v1 consists of:
- **Encoder**: Transformer blocks with self-attention
- **Decoder**: Feature propagation with interpolation
- **Input**: Point coordinates + features
- **Output**: Per-point class predictions

### Data Format

Point Transformer v1 expects:
- `.npy` files with shape `[N, 4]`
- Columns: `[x, y, z, label]`
- Data organized in `train/`, `val/`, `test/` subdirectories
- Supports both `datav2_octree/data_{ratio}_octree/` and `datav2_final/train/datav2_{ratio}_octree_fps/` structures

### Results and Checkpoints

Results are saved to:
```
experiments/point_transformer_v1_logs/model_{ratio}/
├── model/
│   ├── model_best.pth      # Best model checkpoint
│   ├── model_last.pth      # Last epoch checkpoint
│   ├── training_log.txt    # Training log
│   └── test_log.txt        # Test results
```

### Testing/Evaluation

```bash
conda activate point_transformer

python test.py \
  --checkpoint /path/to/model_best.pth \
  --base_data_root /path/to/datav2_octree \
  --real_ratio 0
```

### Point Transformer v1 Specific Notes

- **Separate Environment**: Uses its own conda environment (`point_transformer`)
- **Memory Intensive**: Architecture is memory-intensive; use smaller batch sizes with gradient accumulation
- **CUDA Operations**: Requires custom CUDA ops to be built (see `build_cuda_ops.sh`)
- **Gradient Accumulation**: Use `--accum_steps 4` with `--batch_size 8` to simulate `batch_size=32`
- **Python 3.8 Compatible**: Code has been updated for Python 3.8 compatibility

### Troubleshooting Point Transformer v1

**CUDA Ops Build Issues**:
```bash
# Load CUDA module if available
module load cuda/12.1

# Or set CUDA_HOME
export CUDA_HOME=/usr/local/cuda

# Rebuild CUDA ops
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer
bash build_cuda_ops.sh
```

**Memory Issues**:
- Reduce `--batch_size` (default is 8)
- Use `--accum_steps` for gradient accumulation
- Clear CUDA cache: `torch.cuda.empty_cache()` is called automatically

**Data Path Issues**:
- The config automatically detects different data directory structures
- Ensure `base_data_root` points to the correct base directory

## 🚀 Training Point Transformer v3

### Overview

Point Transformer v3 (PTv3) is a transformer-based architecture for 3D point cloud segmentation using serialized attention mechanisms and sparse convolutions. It uses efficient serialization (z-order, hilbert curves) and sparse 3D convolutions for memory-efficient processing.

**Key Features**:
- Serialized attention with multiple orderings (z-order, hilbert)
- Sparse convolution (spconv) for efficient 3D processing
- Encoder-decoder architecture with multi-scale features
- Separate conda environment for isolation

### Installation

Point Transformer v3 uses a separate conda environment:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer_v3

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate point_transformer_v3

# Install spconv (may require CUDA setup)
pip install spconv-cu118  # For CUDA 11.8
# Or build from source if needed
```

**Dependencies**:
- Python 3.8
- PyTorch >= 2.0.0
- spconv (sparse convolution) - requires CUDA
- torch-scatter
- timm (for DropPath)
- addict (for Dict)
- numpy, pandas, scipy, tqdm, scikit-learn

### Training Commands

#### Basic Training

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer_v3
conda activate point_transformer_v3

python train.py \
  --base_data_root /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree \
  --real_ratio 0 \
  --batch_size 16 \
  --epochs 100
```

#### Training for All Data Regimes

```bash
# 100S-0R (100% Synthetic, 0% Real)
python train.py --base_data_root /path/to/datav2_octree --real_ratio 100 --batch_size 16

# 75S-25R
python train.py --base_data_root /path/to/datav2_octree --real_ratio 75 --batch_size 16

# 50S-50R
python train.py --base_data_root /path/to/datav2_octree --real_ratio 50 --batch_size 16

# 25S-75R
python train.py --base_data_root /path/to/datav2_octree --real_ratio 25 --batch_size 16

# 0S-100R (0% Synthetic, 100% Real)
python train.py --base_data_root /path/to/datav2_octree --real_ratio 0 --batch_size 16
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_data_root` | Auto-detected | Base path to data directory |
| `--real_ratio` | `100` | Real data ratio (0, 25, 50, 75, 100) |
| `--batch_size` | `32` | Training batch size |
| `--epochs` | `100` | Number of training epochs |
| `--base_lr` | `0.001` | Base learning rate |
| `--weight_decay` | `0.01` | Weight decay for optimizer |
| `--drop_path` | `0.1` | Drop path rate for regularization |
| `--voxel_size` | `0.02` | Voxel size for collate function (meters) |
| `--enable_flash` | `False` | Enable flash attention (requires flash_attn) |

### Training Options

Additional options available:

| Parameter | Description |
|-----------|-------------|
| `--head_dropout` | Dropout rate for classification head (default: 0.1) |
| `--label_smoothing` | Label smoothing factor (default: 0.0) |
| `--workers` | Number of data loading workers (default: 4) |

### Model Architecture

Point Transformer v3 consists of:
- **Embedding**: Initial feature embedding layer
- **Encoder**: Multi-stage encoder with serialized attention blocks
- **Decoder**: Multi-stage decoder with feature propagation
- **Head**: Classification head for segmentation

Key features:
- Serialized attention with multiple orderings (z-order, hilbert curves)
- Sparse convolution for efficient 3D processing
- Drop path regularization
- Optional flash attention support

### Data Format

Point Transformer v3 expects:
- `.npy` files with shape `[N, 4]`
- Columns: `[x, y, z, label]`
- Data organized in `train/`, `val/`, `test/` subdirectories
- Supports both `datav2_octree/data_{ratio}_octree/` and `datav2_final/train/datav2_{ratio}_octree_fps/` structures

### Results and Checkpoints

Results are saved to:
```
/home/stud/nguyenti/storage/user/TrueCity/results/ptv3_{dataset_name}_{timestamp}/
├── model/
│   ├── model_best.pth      # Best model checkpoint
│   ├── model_last.pth      # Last epoch checkpoint
│   └── training_log.txt    # Training log
```

### Testing/Evaluation

```bash
conda activate point_transformer_v3

python test.py \
  --checkpoint /path/to/model_best.pth \
  --base_data_root /path/to/datav2_octree \
  --real_ratio 0
```

### Point Transformer v3 Specific Notes

- **Separate Environment**: Uses its own conda environment (`point_transformer_v3`) to avoid dependency conflicts
- **spconv Requirement**: Requires spconv (sparse convolution) which needs CUDA setup
- **Serialized Attention**: Uses efficient serialization mechanisms for memory-efficient attention
- **Memory Efficient**: Handles large point clouds efficiently through sparse convolutions
- **Python 3.8 Compatible**: Code has been updated for Python 3.8 compatibility
- **Label Merging**: Class 12 is automatically merged into class 11 during training

### Troubleshooting Point Transformer v3

**spconv Installation Issues**:
```bash
# Ensure CUDA is available
nvcc --version

# Set CUDA_HOME if needed
export CUDA_HOME=/usr/local/cuda

# Try installing spconv
conda activate point_transformer_v3
pip install spconv-cu118

# If that fails, try building from source
pip install spconv-cu118 --no-build-isolation
```

**Memory Issues**:
- Reduce `--batch_size` (default is 32, try 16 or 8)
- Reduce `--voxel_size` to process fewer points
- Use gradient accumulation if needed

**Import Errors**:
- Ensure you're in the `point_transformer_v3` directory when running scripts
- Verify all dependencies: `pip list | grep -E "spconv|torch-scatter|timm|addict"`
- Check that `serialization/` module is in the same directory

## 🚀 Training OctreeFormer

### Overview

OctreeFormer is an octree-based transformer architecture for 3D point cloud segmentation. It uses hierarchical octree structures to efficiently process large point clouds with attention mechanisms.

**Key Features**:
- Octree-based hierarchical processing
- Transformer attention mechanisms
- Efficient memory usage for large point clouds
- Separate conda environment for isolation

### Installation

OctreeFormer uses a separate conda environment to avoid conflicts with other models:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/octreeformer

# Create the environment (already done if migrated)
conda env create -f environment.yml

# Activate the environment
conda activate octreeformer
```

**Dependencies**:
- Python 3.8
- PyTorch (>=1.12.0)
- ocnn (Octree CNN library) - automatically installed
- numpy, pandas, tqdm, scipy

### Training Commands

#### Basic Training

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/octreeformer
conda activate octreeformer

python train.py \
  --data_path /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree/data_0_octree \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.1 \
  --weight_decay 0.0001 \
  --save_root /home/stud/nguyenti/storage/user/TrueCity/results
```

#### Training with Logging

To save training output to a log file:

```bash
python train.py \
  --data_path /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree/data_0_octree \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.1 \
  --weight_decay 0.0001 \
  --save_root /home/stud/nguyenti/storage/user/TrueCity/results \
  > /home/stud/nguyenti/storage/user/TrueCity/results/octformer_data_0_octree_20250813_172119/octreeformer_data_0_octree_run.log 2>&1
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | Required | Path to data directory (must have train/val/test subdirectories) |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Training batch size |
| `--lr` | `0.1` | Learning rate |
| `--weight_decay` | `0.0001` | Weight decay for optimizer |
| `--save_root` | `/home/stud/nguyenti/storage/user/tum-di-lab/results` | Root directory for saving results |
| `--run_name` | Auto-generated | Custom run directory name |
| `--drop_path` | `None` | Stochastic depth drop path rate |
| `--dropout` | `None` | Head dropout rate |
| `--accum_steps` | `1` | Gradient accumulation steps |
| `--max_points_per_item` | `None` | Optional downsampling cap per item |

### Training Options

Additional options available:

| Parameter | Description |
|-----------|-------------|
| `--test_only` | Only run evaluation on test set |
| `--checkpoint` | Path to checkpoint for evaluation |
| `--run_dir` | Existing run directory to load checkpoint from |
| `--no_dwconv` | Disable CUDA dwconv extension |
| `--use_dwconv` | Enable CUDA dwconv extension explicitly |

### Model Architecture

OctreeFormer consists of:
- **Backbone**: OctFormer with hierarchical transformer blocks
- **Head**: Segmentation head with FPN-style feature fusion
- **Input**: Point coordinates + features (PL: Position + Label)
- **Output**: Per-point class predictions

### Data Format

OctreeFormer expects the same data format as other models:
- `.npy` files with shape `[N, 4]`
- Columns: `[x, y, z, label]`
- Data organized in `train/`, `val/`, `test/` subdirectories

### Results and Checkpoints

Results are saved to:
```
/home/stud/nguyenti/storage/user/TrueCity/results/octformer_<data>_<timestamp>/
├── model/
│   ├── model_best.pth      # Best model checkpoint
│   ├── model_last.pth      # Last epoch checkpoint
│   ├── training_log.txt    # Training configuration
│   └── test_log.txt        # Test results
└── octreeformer_<data>_octree_run.log  # Full training log
```

### Testing/Evaluation

```bash
conda activate octreeformer

python test.py \
  --checkpoint /path/to/model_best.pth \
  --data_path /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree/data_0_octree \
  --batch_size 32
```

### OctreeFormer Specific Notes

- **Separate Environment**: Uses its own conda environment (`octreeformer`) to avoid dependency conflicts
- **Octree Processing**: Automatically builds octrees from point clouds
- **Memory Efficient**: Handles large point clouds efficiently through hierarchical processing
- **Python 3.8 Compatible**: Code has been updated for Python 3.8 compatibility
- **Migration**: Successfully migrated from `tum-di-lab` repository with preserved logic

### Troubleshooting OctreeFormer

**Environment Issues**:
```bash
# If environment doesn't exist
conda env create -f /home/stud/nguyenti/storage/user/TrueCity/models/octreeformer/environment.yml

# If ocnn is missing
conda activate octreeformer
pip install ocnn
```

**Import Errors**:
- Ensure you're in the `octreeformer` directory when running scripts
- Check that `point_transformer` module is in the same directory
- Verify `augmentations.py` is in the root of `octreeformer/`

**Memory Issues**:
- Use `--max_points_per_item` to cap points per sample
- Reduce `--batch_size` if running out of memory
- Use `--accum_steps` for gradient accumulation with smaller batches

## 📈 Evaluation

### Test Evaluation

```bash
# PointNet
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py \
    --mode test \
    --model_path pointnet_model.pth \
    --data_path /path/to/data \
    --n_points 2048 \
    --batch_size 32

# PointNet2++
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet2.py \
    --mode test \
    --model_path pointnet2_model.pth \
    --data_path /path/to/data \
    --n_points 2048 \
    --batch_size 32
```

### Evaluation Metrics

The evaluation outputs:
- **Overall Accuracy**: Per-point classification accuracy
- **Mean IoU (mIoU)**: Mean Intersection over Union across all classes
- **Per-class IoU**: IoU for each individual class
- **Per-class Accuracy**: Accuracy for each class
- **Precision, Recall, F1**: Per-class metrics (PointNet2++)

### Dataset Analysis

Analyze your dataset before training:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py \
    --mode analyze \
    --data_path /path/to/data \
    --max_files 100  # Optional: limit analysis to first 100 files
```

This provides:
- File count and point statistics
- Coordinate bounds
- Class distribution
- Sampling impact analysis

## ⚙️ Advanced Configuration

### Batch Size Optimization

Adjust batch size based on GPU memory:

```bash
# For 24GB GPU (RTX 3090, A5000)
--batch_size 16 --n_points 2048

# For 48GB GPU (A100)
--batch_size 32 --n_points 4096

# For 8GB GPU (RTX 3060)
--batch_size 4 --n_points 1024
```

### Learning Rate Scheduling

PointNet uses StepLR scheduler:
- Step size: 20 epochs
- Gamma: 0.7

PointNet2++ uses custom decay:
- Step size: 10 epochs
- Decay: 0.7
- Minimum LR: 1e-5

### Class Weighting

For imbalanced datasets, modify the loss function in the trainer files:

```python
# In pointnet_trainer.py or pointnet2_trainer.py
# Calculate class weights based on dataset
class_weights = compute_class_weights(dataset)
criterion = SemanticSegmentationLoss(weight=class_weights)
```

### Multi-GPU Training

For multi-GPU setups, modify the trainer to use DataParallel:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 🐛 Troubleshooting

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch_size 4`
2. Reduce points per sample: `--n_points 1024`
3. Use gradient accumulation
4. Enable mixed precision training

### No Data Found

**Symptoms**: `ValueError: No data files found for split 'train'`

**Solutions**:
1. Check data directory structure (must have `train/`, `val/`, `test/` subdirectories)
2. Verify file format (`.npy` files with `[N, 4]` shape)
3. Check file naming (should match split name or be in subdirectory)
4. Run dataset analysis to verify: `--mode analyze`

### Import Errors

**Symptoms**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
1. Ensure you're in the project root directory
2. Install dependencies: `pip install -r requirements.txt`
3. Check Python path: `python -c "import sys; print(sys.path)"`
4. Verify project structure exists

### Poor Training Performance

**Symptoms**: Low accuracy, unstable training, slow convergence

**Solutions**:
1. Increase `n_points` for better spatial coverage
2. Adjust learning rate (try 0.0005 or 0.002)
3. Use precomputed FPS data for consistent sampling
4. Check class distribution (may need class weighting)
5. Increase training epochs
6. Verify data quality and labels

### Checkpoint Issues

**Symptoms**: Cannot resume training, checkpoint errors

**Solutions**:
1. Check checkpoint file exists and is readable
2. Verify checkpoint contains required keys:
   - `model_state_dict`
   - `optimizer_state_dict`
   - `epoch`
   - `class_to_idx`
3. List available checkpoints: `--mode checkpoints`

### Model Architecture Mismatch

**Symptoms**: `RuntimeError: Error(s) in loading state_dict`

**Solutions**:
1. Ensure model architecture matches checkpoint
2. Check number of classes matches dataset
3. Verify model file corresponds to correct architecture (PointNet vs PointNet2++)

## 📝 Training Logs

Training logs are saved to:
- **PointNet**: `./logs/pointnet_training_YYYY-MM-DD_HH-MM.txt`
- **PointNet2++**: `./log/pointnet2_sem_seg/YYYY-MM-DD_dataset/logs/`
- **OctreeFormer**: `./results/octformer_<data>_<timestamp>/octreeformer_<data>_octree_run.log`

Logs include:
- Training progress and metrics
- Validation results
- Model checkpoints
- Final evaluation results

## 💡 Best Practices

1. **Start Small**: Begin with small datasets and fewer epochs to verify setup
2. **Monitor Training**: Watch for NaN losses, memory issues, or convergence problems
3. **Save Checkpoints**: Models are saved automatically, but keep backups
4. **Validate Early**: Check validation metrics to catch overfitting
5. **Use Precomputed Data**: For faster iteration, precompute FPS samples
6. **Experiment Tracking**: Use experiment directories (PointNet2++) for organization

## 🔗 Quick Reference

### Data Verification

Before training, verify your data setup:

```bash
# Check data availability
EARLY_DATA="/home/stud/nguyenti/storage/user/EARLy/datav2_final"
ls -la $EARLY_DATA/datav2_*_octree_fps

# Verify data structure for a specific split
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py \
    --mode analyze \
    --data_path $EARLY_DATA/datav2_0_octree_fps
```

### Common Commands

```bash
# Analyze dataset
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py --mode analyze --data_path /path/to/data

# Train PointNet (0S-100R example)
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_0_octree_fps \
    --n_points 2048 \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.001

# Train PointNet2++ (0S-100R example)
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet2.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_0_octree_fps \
    --n_points 2048 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.00005

# Train Point Transformer v1 (0S-100R example)
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer
conda activate point_transformer
python train.py \
  --base_data_root /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree \
  --real_ratio 0 \
  --batch_size 8 \
  --accum_steps 4

# Train Point Transformer v3 (0S-100R example)
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer_v3
conda activate point_transformer_v3
python train.py \
  --base_data_root /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree \
  --real_ratio 0 \
  --batch_size 16 \
  --epochs 100

# Train OctreeFormer (0S-100R example)
cd /home/stud/nguyenti/storage/user/TrueCity/models/octreeformer
conda activate octreeformer
python train.py \
  --data_path /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree/data_0_octree \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.1 \
  --weight_decay 0.0001 \
  --save_root /home/stud/nguyenti/storage/user/TrueCity/results

# Evaluate PointNet
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py --mode test --model_path model.pth --data_path /path/to/data

# Evaluate Point Transformer v1
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer
conda activate point_transformer
python test.py --checkpoint /path/to/model_best.pth --base_data_root /path/to/data --real_ratio 0

# Evaluate Point Transformer v3
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer_v3
conda activate point_transformer_v3
python test.py --checkpoint /path/to/model_best.pth --base_data_root /path/to/data --real_ratio 0

# Evaluate OctreeFormer
cd /home/stud/nguyenti/storage/user/TrueCity/models/octreeformer
conda activate octreeformer
python test.py --checkpoint /path/to/model_best.pth --data_path /path/to/data

# List checkpoints
cd /home/stud/nguyenti/storage/user/TrueCity/models/pointnet
python scripts/train_pointnet.py --mode checkpoints --model_path model.pth
```

### Quick Setup Script

Run the setup script to verify data and generate training commands:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity
bash setup_training.sh
```

This will:
- Check all data splits availability
- Verify data structure
- Generate training commands for all available splits
- Create a `training_commands.sh` file with ready-to-use functions

### File Locations

- **Models**: Saved to current directory or specified `--model_path`
- **Logs**: `./logs/` (PointNet) or `./log/pointnet2_sem_seg/` (PointNet2++)
- **Checkpoints**: Same as model path, with `_checkpoint_epoch_N.pth` suffix

---

For more details, see:
- `README.md` - Project overview

