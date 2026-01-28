# PointNet and PointNet2++ for Semantic Segmentation

This directory contains implementations of PointNet and PointNet2++ for semantic segmentation on the Ingolstadt point cloud dataset.

## Structure

```
models/pointnet/
├── src/
│   ├── models/
│   │   ├── pointnet.py          # PointNet model implementation
│   │   └── pointnet2.py         # PointNet2++ model implementation
│   └── training/
│       ├── pointnet_trainer.py  # PointNet training logic
│       ├── pointnet2_trainer.py # PointNet2++ training logic
│       └── metrics.py            # Shared evaluation metrics
├── scripts/
│   ├── train_pointnet.py        # PointNet training script
│   └── train_pointnet2.py       # PointNet2++ training script
├── environment.yml              # Conda environment definition
├── requirements.txt             # Pip requirements
└── README.md                    # This file
```

## Dependencies

Both PointNet and PointNet2++ share the same dependencies since they use the same preprocessing pipeline.

### Installation

#### Option 1: Conda Environment (Recommended)

```bash
cd models/pointnet
conda env create -f environment.yml
conda activate truecity_pointnet
```

#### Option 2: Pip Installation

```bash
cd models/pointnet
pip install -r requirements.txt
```

**Note**: If you need CUDA support, install PyTorch with CUDA separately:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Data Requirements

The scripts expect data in the following format:
- **Data path**: `/home/stud/nguyenti/storage/user/EARLy/data` (default)
- **Format**: NPY files with columns `[x, y, z, object_class]`
- **Structure**: Organized in `train/`, `val/`, and `test/` subdirectories

You can specify a different data path using the `--data_path` argument.

## Usage

### PointNet

#### Training

```bash
cd models/pointnet
python scripts/train_pointnet.py --mode train \
    --data_path /path/to/data \
    --n_points 2048 \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --model_path pointnet_model.pth
```

#### Evaluation

```bash
python scripts/train_pointnet.py --mode eval \
    --model_path pointnet_model.pth \
    --data_path /path/to/data \
    --n_points 2048 \
    --batch_size 32
```

#### Test Evaluation

```bash
python scripts/train_pointnet.py --mode test \
    --model_path pointnet_model.pth \
    --data_path /path/to/data
```

#### Dataset Analysis

```bash
python scripts/train_pointnet.py --mode analyze \
    --data_path /path/to/data
```

#### List Checkpoints

```bash
python scripts/train_pointnet.py --mode checkpoints \
    --model_path pointnet_model.pth
```

### PointNet2++

#### Training

```bash
cd models/pointnet
python scripts/train_pointnet2.py --mode train \
    --data_path /path/to/data \
    --n_points 2048 \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 0.0001 \
    --model_path pointnet2_model.pth
```

#### Evaluation

```bash
python scripts/train_pointnet2.py --mode eval \
    --model_path pointnet2_model.pth \
    --data_path /path/to/data \
    --n_points 2048 \
    --batch_size 32
```

#### Test Evaluation

```bash
python scripts/train_pointnet2.py --mode test \
    --model_path pointnet2_model.pth \
    --data_path /path/to/data
```

## Command-Line Arguments

### Common Arguments (Both Models)

- `--mode`: Operation mode
  - `train`: Train a new model or resume from checkpoint
  - `eval`: Evaluate on validation set
  - `test`: Comprehensive test evaluation
  - `analyze`: Analyze dataset statistics (PointNet only)
  - `checkpoints`: List available checkpoints (PointNet only)

- `--data_path`: Path to data directory (default: `/home/stud/nguyenti/storage/user/EARLy/data`)
- `--model_path`: Path to save/load model checkpoint
- `--n_points`: Number of points per sample (default: 2048)
- `--batch_size`: Batch size for training/evaluation
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate
- `--sample_multiplier`: Multiplier for samples per file (default: 1.0)
- `--ignore_labels`: Labels to ignore during training (space-separated, e.g., `3 4`)

### PointNet-Specific Arguments

- `--max_files`: Maximum number of files to analyze in analyze mode

## Model Checkpoints

Both models automatically:
- **Save checkpoints** during training (best model and latest epoch)
- **Resume training** if a checkpoint exists at `--model_path`
- **Save best model** based on validation accuracy

Checkpoint format:
```python
{
    'epoch': int,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'val_accuracy': float,
    'best_val_accuracy': float,
    ...
}
```

## Training Tips

### PointNet
- **Recommended settings**: `n_points=2048`, `batch_size=8`, `learning_rate=0.001`
- Lower batch size if running out of memory
- Training automatically resumes from checkpoint if model file exists

### PointNet2++
- **Recommended settings**: `n_points=2048`, `batch_size=32`, `learning_rate=0.0001`
- PointNet2++ typically requires more memory than PointNet
- Uses hierarchical feature extraction with set abstraction layers

## Evaluation Metrics

Both models compute:
- **Overall Accuracy (OA)**: Percentage of correctly classified points
- **Mean IoU (mIoU)**: Mean Intersection over Union across all classes
- **Per-class IoU**: IoU for each individual class
- **Per-class Accuracy**: Accuracy for each individual class

## Shared Preprocessing

Both models use the same preprocessing pipeline from `shared/`:
- **FPS Sampling**: Farthest Point Sampling for point cloud downsampling
- **Data Loading**: `IngolstadtDataset` and `create_ingolstadt_dataloaders`
- **Logging**: Unified logging utilities

## Examples

### Quick Training Run (PointNet)

```bash
cd models/pointnet
conda activate truecity_pointnet
python scripts/train_pointnet.py --mode train \
    --n_points 1024 \
    --batch_size 16 \
    --num_epochs 10 \
    --sample_multiplier 0.2
```

### Full Training Run (PointNet2++)

```bash
cd models/pointnet
conda activate truecity_pointnet
python scripts/train_pointnet2.py --mode train \
    --data_path /home/stud/nguyenti/storage/user/EARLy/data \
    --n_points 2048 \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 0.0001 \
    --model_path pointnet2_segmentation.pth
```

### Evaluate Existing Model

```bash
cd models/pointnet
conda activate truecity_pointnet
python scripts/train_pointnet.py --mode test \
    --model_path ingolstadt_pointnet_segmentation.pth \
    --data_path /home/stud/nguyenti/storage/user/EARLy/data \
    --n_points 2048 \
    --batch_size 32
```

## Troubleshooting

### Out of Memory Errors
- Reduce `--batch_size`
- Reduce `--n_points`
- Use `--sample_multiplier 0.5` or lower

### Import Errors
Make sure you're in the correct directory and have activated the conda environment:
```bash
cd models/pointnet
conda activate truecity_pointnet
```

### Data Path Issues
Verify your data path structure:
```
data/
├── train/
│   ├── file1.npy
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## Notes

- Both models share the same codebase structure and preprocessing
- Training automatically saves best model based on validation accuracy
- Models can be resumed from checkpoints automatically
- All logging is saved to `logs/` directory
- Function signatures match the original EARLy repository implementation

