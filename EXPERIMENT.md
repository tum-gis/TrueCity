# Experiment Guide

## 📋 Table of Contents

1. [Project Architecture](#project-architecture)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Training PointNet](#training-pointnet)
5. [Training PointNet2++](#training-pointnet2)
6. [Evaluation](#evaluation)
7. [Advanced Configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)

## 🏗️ Project Architecture

### Project Structure

```
TrueCity/
├── src/                    # Source code
│   ├── models/            # Model definitions
│   │   ├── __init__.py
│   │   ├── pointnet.py    # PointNet model
│   │   └── pointnet2.py   # PointNet2++ model
│   ├── data/              # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py     # IngolstadtDataset class
│   │   └── dataloader.py  # DataLoader creation and analysis
│   ├── training/          # Training logic
│   │   ├── __init__.py
│   │   ├── pointnet_trainer.py    # PointNet training
│   │   ├── pointnet2_trainer.py   # PointNet2++ training
│   │   └── metrics.py     # Evaluation metrics
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── logging.py     # Logging setup
│       └── fps.py         # FPS sampling functions
├── scripts/               # Executable scripts
│   ├── train_pointnet.py
│   └── train_pointnet2.py
├── tools/                 # Utility scripts
├── examples/              # Example usage
├── docs/                  # Additional documentation
├── config/                 # Configuration files
├── README.md
├── Experiment.md
└── requirements.txt
```

### Data Flow

#### Training Pipeline

1. **Data Loading** (`src/data/dataloader.py`)
   - Load point cloud files from disk
   - Apply FPS sampling (if not precomputed)
   - Normalize coordinates
   - Filter invalid classes
   - Create PyTorch DataLoaders

2. **Model Creation** (`src/models/`)
   - Initialize PointNet or PointNet2++ architecture
   - Setup loss function
   - Move to GPU if available

3. **Training Loop** (`src/training/*_trainer.py`)
   - Forward pass through model
   - Compute loss
   - Backward pass and optimization
   - Validation evaluation
   - Checkpoint saving

4. **Evaluation** (`src/training/metrics.py`)
   - Compute comprehensive metrics
   - Per-class IoU, accuracy, precision, recall, F1
   - Overall accuracy and mean IoU

### Model Architectures

#### PointNet (`src/models/pointnet.py`)
- **Encoder**: STN3d → Conv layers → Max pooling
- **Decoder**: Feature concatenation → Conv layers → Classification
- **Features**: Feature transformation for rotation invariance

#### PointNet2++ (`src/models/pointnet2.py`)
- **Set Abstraction**: Hierarchical point sampling and grouping
- **Feature Propagation**: Upsampling with interpolation
- **Features**: Multi-scale feature learning

### Module Dependencies

```
scripts/
  └── train_*.py
      ├── src/training/*_trainer.py
      │   ├── src/models/*.py
      │   ├── src/data/dataloader.py
      │   ├── src/training/metrics.py
      │   └── src/utils/logging.py
      └── src/utils/logging.py
```

### Key Design Decisions

1. **Modular Structure**: Separated models, data, training, and utilities
2. **Preserved Logic**: All original functionality maintained during refactoring
3. **Unified Interfaces**: Consistent API across PointNet and PointNet2++
4. **Extensible**: Easy to add new models or datasets

### Import Paths

All imports use relative paths within the `src/` package:
- `from ..models.pointnet import ...`
- `from ..data.dataloader import ...`
- `from .metrics import ...`

Scripts add the project root to `sys.path` for absolute imports:
- `from src.training.pointnet_trainer import ...`

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
python scripts/train_pointnet.py --mode train --data_path /path/to/data
python scripts/train_pointnet2.py --mode train --data_path /path/to/data
python scripts/train_superpoint.py datamodule.data_dir=/path/to/data experiment=semantic/truecity
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
from src.data.preprocessing import precompute_fps_dataset

precompute_fps_dataset(
    source_path='/path/to/raw/data',
    output_path='/path/to/precomputed/data',
    n_points=2048,
    sample_multiplier=1.0,
    gpu_batch_size=8
)
```

Then use the precomputed data path for training.

## 🚀 Training PointNet

### Exact Training Commands (From Successful Experiments)

These are the exact hyperparameters used to achieve the results in the paper table:

#### 100S-0R (100% Synthetic, 0% Real)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity

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
cd /home/stud/nguyenti/storage/user/TrueCity

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

## 📈 Evaluation

### Test Evaluation

```bash
# PointNet
python scripts/train_pointnet.py \
    --mode test \
    --model_path pointnet_model.pth \
    --data_path /path/to/data \
    --n_points 2048 \
    --batch_size 32

# PointNet2++
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
python scripts/train_pointnet.py \
    --mode analyze \
    --data_path $EARLY_DATA/datav2_0_octree_fps
```

### Common Commands

```bash
# Analyze dataset
python scripts/train_pointnet.py --mode analyze --data_path /path/to/data

# Train PointNet (0S-100R example)
python scripts/train_pointnet.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_0_octree_fps \
    --n_points 2048 \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.001

# Train PointNet2++ (0S-100R example)
python scripts/train_pointnet2.py \
    --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/datav2_final/datav2_0_octree_fps \
    --n_points 2048 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.00005

# Evaluate model
python scripts/train_pointnet.py --mode test --model_path model.pth --data_path /path/to/data

# List checkpoints
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

