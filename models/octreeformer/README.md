# OctreeFormer Model

This directory contains the OctreeFormer implementation for 3D point cloud segmentation, migrated from the tum-di-lab repository.

## Structure

```
octreeformer/
├── model.py              # Model wrapper and utilities
├── octformer.py          # Core OctFormer implementation
├── octformerseg.py       # Segmentation head
├── train.py              # Training script
├── test.py               # Testing/evaluation script
├── augmentations.py      # Data augmentation utilities
├── point_transformer/    # Point transformer utilities (config, data loaders)
│   ├── config.py
│   ├── data_utils.py
│   └── libs/            # Point operations libraries
├── environment.yml       # Conda environment specification
└── README.md            # This file
```

## Dependencies

The model requires:
- PyTorch (>=1.12.0)
- ocnn (Octree CNN library)
- numpy, pandas, tqdm, scipy
- point_transformer utilities (included in this directory)

## Environment Setup

The conda environment has been created. Activate it:

```bash
conda activate octreeformer
```

If you need to recreate the environment:
```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/octreeformer
conda env create -f environment.yml
conda activate octreeformer
```

Note: `ocnn` is automatically installed via pip when creating the environment.

## Usage

### Training

```bash
cd /home/stud/nguyenti/storage/user/TrueCity/models/octreeformer
conda activate octreeformer
python train.py \
  --data_path /home/stud/nguyenti/storage/user/tum-di-lab/datav2_octree/data_0_octree \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.1 \
  --weight_decay 0.0001 \
  --save_root /home/stud/nguyenti/storage/user/TrueCity/results \
  > /home/stud/nguyenti/storage/user/TrueCity/results/octformer_data_0_octree_20250813_172119/octreeformer_data_0_octree_run.log 2>&1
```

### Testing

```bash
python test.py --checkpoint /path/to/checkpoint.pth --data_path /path/to/data
```

## Results

Model checkpoints and logs are stored in:
```
/home/stud/nguyenti/storage/user/TrueCity/results/octformer_data_0_octree_20250813_172119/model/
```

This includes:
- `model_best.pth` - Best model checkpoint
- `model_last.pth` - Last epoch checkpoint
- `training_log.txt` - Training configuration and logs
- `test_log.txt` - Test results

## Notes

- The code logic has been preserved during migration
- Import paths have been updated to work in the new location
- This model uses a separate environment from the main TrueCity environment

