# TrueCity Experiments

This document tracks experiments and model configurations for the TrueCity project.

## Data Preprocessing

**IMPORTANT**: Before training any model, you must preprocess your data to match the format used in EARLy or tum-di-lab.

### Required Data Format

The data should be organized in the following structure:
```
data_root/
├── train/
│   ├── file1.npy  # Shape: [N, 4] -> [x, y, z, label]
│   ├── file2.npy
│   └── ...
├── val/
│   ├── file1.npy
│   └── ...
└── test/
    ├── file1.npy
    └── ...
```

Each `.npy` file should contain point clouds with 4 columns: `[x, y, z, label]`

### Preprocessing Options

#### Option 1: KPConv-Specific Preprocessing (Required for KPConv)

**For KPConv, you MUST use `presample_kpconv.py` to preprocess your data.** This preprocessing splits large point cloud files into manageable chunks of ~16k points each, which is essential to avoid OOM errors and enable efficient training with larger batch sizes.

```bash
# Activate environment
conda activate kpconv

# Navigate to script directory
cd /home/stud/nguyenti/storage/user/TrueCity/models/kpconv

# Run preprocessing
python presample_kpconv.py \
  --data_root /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree/data_25_octree \
  --out_root /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree_part16k/data_25_octree_parts16k \
  --target_points 16384 \
  --seed 0
```

**Parameters:**
- `--data_root`: Input dataset root containing train/val/test/ with .npy files (e.g., `datav2_octree/data_25_octree`)
- `--out_root`: Output dataset root to write pre-sampled files (e.g., `datav2_octree_part16k/data_25_octree_parts16k`)
  - **Important**: Output goes to `datav2_octree_part16k/` directory (not `datav2_octree/`)
- `--target_points`: Target number of points per output sample (default: 16384, creates ~16k point chunks)
- `--seed`: Random seed for reproducibility

**What this does:**
- Splits each input file into multiple chunks of approximately 16,384 points each
- Creates many more files (e.g., 185 original files → 4941 preprocessed files)
- Enables training with batch_size=16 and workers=4 instead of batch_size=1-2
- Output directory structure: `datav2_octree_part16k/data_25_octree_parts16k/` (contains train/, val/, test/)

#### Option 2: Octree FPS Preprocessing (For other models)

Use the TrueCity preprocessing tool to create FPS-sampled data for other models:

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/tools
python precompute_octree_fps.py \
  --source_path /path/to/your/raw/data \
  --output_path /path/to/output/preprocessed/data \
  --n_points 2048 \
  --sample_multiplier 0.2 \
  --gpu_batch_size 16
```

**Parameters:**
- `--source_path`: Path to your raw data (should have train/val/test subdirectories)
- `--output_path`: Where to save preprocessed data
- `--n_points`: Number of points per sample (default: 2048)
- `--sample_multiplier`: Multiplier for samples (0.2=fast training, 1.0=full dataset)
- `--gpu_batch_size`: GPU batch size for FPS processing

### After Preprocessing

Once preprocessing is complete, use the preprocessed data path in your training commands (see model-specific sections below).

### Complete Example: Full Workflow for KPConv

Here's the complete step-by-step workflow that was successfully used:

#### Step 1: Activate Environment
```bash
conda activate kpconv
```

#### Step 2: Preprocess Data (REQUIRED for KPConv)
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/kpconv
python presample_kpconv.py \
  --data_root /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree/data_25_octree \
  --out_root /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree_part16k/data_25_octree_parts16k \
  --target_points 16384 \
  --seed 0
```

**Note**: The output will be in `datav2_octree_part16k/` directory (not `datav2_octree/`). Wait for preprocessing to complete - this may take some time.

#### Step 3: Train KPConv with Preprocessed Data
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/kpconv
python train_dilab_with_sample.py \
  --data_path /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree_part16k/data_25_octree_parts16k \
  --save_dir /home/stud/nguyenti/storage/user/TrueCity/results \
  --batch_size 16 \
  --ebatch_size 32 \
  --workers 4 \
  --epochs 10 \
  --gpu 0
```

**Important**: Use the correct preprocessed data path: `datav2_octree_part16k/data_25_octree_parts16k` (not `datav2_octree/data_25_octree_parts16k`)

**Expected results after preprocessing:**
- Original: 185 train files → Preprocessed: ~4941 train files (files split into ~16k point chunks)
- This enables training with `batch_size=16` and `workers=4` instead of `batch_size=1-2`

**Note**: The preprocessing step is **mandatory** for KPConv. Without it, you will encounter:
- `FileNotFoundError` when trying to train (data directory doesn't exist)
- OOM errors even with batch_size=1 (if you try to use raw data)

---

## RandLA-Net

### Location
`/home/stud/nguyenti/storage/user/TrueCity/models/randla_net/`

### Setup
1. Create conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate randla_net
   ```

2. **Preprocess your data first** (see Data Preprocessing section above)

3. Training command:
   ```bash
   cd /home/stud/nguyenti/storage/user/TrueCity/models/randla_net
   python train.py \
     --data_root /path/to/preprocessed/data \
     --epochs 10 \
     --batch_size 8 \
     --lr 1e-4
   ```
   
   **Note**: Use the preprocessed data path from the preprocessing step.

### Configuration
- Default batch size: 8
- Learning rate: 1e-4
- Number of epochs: 10
- Input points per batch: 1024

### Results
Results are saved to `TrueCity/results/` directory.

---

## KPConv

### Location
`/home/stud/nguyenti/storage/user/TrueCity/models/kpconv/`

### Setup
1. Create conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate kpconv
   ```

2. **Preprocess your data first** (see Data Preprocessing section above)
   
   **For KPConv, you MUST use `presample_kpconv.py` (Option 1 in preprocessing section).** This creates the `*_parts16k` format which is required for efficient training.

3. Training command:
   ```bash
   # Activate environment
   conda activate kpconv
   
   # Navigate to training script directory
   cd /home/stud/nguyenti/storage/user/TrueCity/models/kpconv
   
   # Run training
   python train_dilab_with_sample.py \
     --data_path /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree_part16k/data_25_octree_parts16k \
     --save_dir /home/stud/nguyenti/storage/user/TrueCity/results \
     --batch_size 16 \
     --ebatch_size 32 \
     --workers 4 \
     --epochs 10 \
     --gpu 0
   ```
   
   **Parameters**:
   - `--data_path`: Path to preprocessed data (must use `datav2_octree_part16k/data_25_octree_parts16k` format)
   - `--save_dir`: Directory to save results and checkpoints
   - `--batch_size 16`: Actual batch size per step (requires preprocessed data)
   - `--ebatch_size 32`: Effective batch size via gradient accumulation
   - `--workers 4`: Number of DataLoader workers (requires preprocessed data)
   - `--epochs 10`: Number of training epochs
   - `--gpu 0`: GPU device ID
   
   **Note**: 
   - Use the preprocessed data path with `_parts16k` suffix
   - The correct path format is: `datav2_octree_part16k/data_25_octree_parts16k` (output from `presample_kpconv.py`)
   - Without preprocessing, you'll get `FileNotFoundError` - preprocessing is mandatory

### Configuration
- Default batch size: 16 (with proper preprocessing using `presample_kpconv.py`)
- Workers: 4 (with proper preprocessing)
- Effective batch size: 32 (via gradient accumulation with `--ebatch_size 32`)
- Learning rate: 1e-3
- Number of epochs: 10
- `first_subsampling_dl`: 0.12 (increased from 0.06 to reduce points in first layer)
- `max_in_points`: 200000 (limits points per batch to prevent OOM)

**Important**: These settings work **only with properly preprocessed data** (using `presample_kpconv.py` with `--target_points 16384`). Without preprocessing, you will need to use batch_size=1-2 and workers=1, and may still encounter OOM errors.

### Notes
- **Preprocessing is mandatory**: Use `presample_kpconv.py` to create `*_parts16k` format before training
- Preprocessing splits large files into ~16k point chunks, enabling efficient training with larger batch sizes
- After preprocessing, expect many more files (e.g., 185 original files → 4941 preprocessed files)
- KDTree operations are memory-intensive, but preprocessing makes them manageable
- Use gradient accumulation (`--ebatch_size`) to maintain effective batch size
- Worker logs are saved to `/tmp/kpconv_worker_*.log` for debugging

### Troubleshooting

If you encounter OOM errors even with preprocessed data:

1. **Reduce batch size**: Try `--batch_size 8` or `--batch_size 4`
2. **Reduce workers**: Try `--workers 2` or `--workers 1`
3. **Use PyTorch memory allocator optimization**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
4. **Increase `first_subsampling_dl`** in config.py (e.g., 0.15 or 0.18)
5. **Reduce `max_in_points`** in config.py (e.g., 100000 or 150000)

### Results
Results are saved to `TrueCity/results/kp_conv_<dataset>_<timestamp>_log/`

---

## Notes
- **Data Preprocessing is Required**: All models expect preprocessed data in the format described above (train/val/test splits with .npy files containing [x, y, z, label])
- Preprocessing must be done before training - use the tools in `TrueCity/tools/` or model-specific preprocessing scripts
- All models use the TrueCity dataset structure with train/val/test splits
- Results are centralized in `TrueCity/results/` directory
- Model checkpoints and logs are saved per experiment

