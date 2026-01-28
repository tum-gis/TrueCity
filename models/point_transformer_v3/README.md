# Point Transformer v3

Point Transformer v3 (PTv3) is a transformer-based architecture for 3D point cloud segmentation using serialized attention mechanisms and sparse convolutions.

## Overview

Point Transformer v3 uses:
- **Serialized Attention**: Efficient attention mechanisms with serialization (z-order, hilbert curves)
- **Sparse Convolutions**: spconv for efficient 3D sparse tensor operations
- **Hierarchical Processing**: Encoder-decoder architecture with multi-scale features

## Structure

```
models/point_transformer_v3/
├── train.py                     # Training entrypoint
├── test.py                      # Evaluation script
├── test_new.py                  # Additional eval utilities
├── model_v3.py                  # Model definition
├── config.py                    # Config class and get_config()
├── data_utils.py                # Dataloaders, transforms
├── augmentations.py             # Geometric/data augmentations
├── serialization/               # Serialization utilities (z_order, hilbert, default)
├── environment.yml              # Conda environment (this module only)
└── README.md
```

## Environment Setup

Create and activate the dedicated environment:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/point_transformer_v3

conda env create -f environment.yml
conda activate point_transformer_v3
```

### Install spconv (Sparse Convolution)

**Important**: spconv requires CUDA and may need to be built from source depending on your CUDA version.

#### Option A: Pre-built wheel (CUDA 11.8)
```bash
conda activate point_transformer_v3
pip install spconv-cu118
```

#### Option B: Build from source (if pre-built doesn't work)
```bash
conda activate point_transformer_v3

# Load CUDA module if available
module load cuda/12.1  # or your CUDA version

# Or set CUDA_HOME
export CUDA_HOME=/usr/local/cuda

# Install spconv
pip install spconv-cu118 --no-build-isolation
```

### Verify Installation

```bash
conda activate point_transformer_v3
python -c "import torch; import spconv; import torch_scatter; import timm; import addict; print('All dependencies OK!')"
```

## Data Expectations

Point Transformer v3 supports multiple data directory structures:

1. **datav2_octree structure**: `datav2_octree/data_{ratio}_octree/train/`
2. **datav2_final structure**: `datav2_final/train/datav2_{ratio}_octree_fps/` or `datav2_final/datav2_{ratio}_octree_fps/train/`

The config automatically detects the structure based on the `base_data_root` path.

Each `.npy` file has shape `[N, 4]`:
- Columns 0–2: `x, y, z` coordinates (`float32`)
- Column 3: class label

## Dependencies

- Python 3.8
- PyTorch >= 2.0.0
- spconv (sparse convolution) - requires CUDA
- torch-scatter
- timm (for DropPath)
- addict (for Dict)
- numpy, pandas, scipy, tqdm, scikit-learn

## Model Architecture

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

## Results

Results are saved to:
```
/home/stud/nguyenti/storage/user/TrueCity/results/ptv3_{dataset_name}_{timestamp}/
├── model/
│   ├── model_best.pth      # Best model checkpoint
│   ├── model_last.pth      # Last epoch checkpoint
│   └── training_log.txt    # Training log
```

## Troubleshooting

### spconv Installation Issues

If spconv fails to install:
1. Ensure CUDA is available: `nvcc --version`
2. Set `CUDA_HOME` environment variable
3. Try building from source: `pip install spconv-cu118 --no-build-isolation`
4. Check CUDA version compatibility (spconv-cu118 requires CUDA 11.8)

### Import Errors

- Ensure you're in the `point_transformer_v3` directory when running scripts
- Verify all dependencies are installed: `pip list | grep -E "spconv|torch-scatter|timm|addict"`
- Check that `serialization/` module is in the same directory

## Notes

- Point Transformer v3 uses a separate conda environment from Point Transformer v1
- The model uses serialized attention which is more memory-efficient than standard attention
- Flash attention can be enabled for faster training if `flash_attn` is installed
- Label merging: Class 12 is automatically merged into class 11 during training
- For training commands and detailed usage, see `EXPERIMENT.md` in the TrueCity root directory
