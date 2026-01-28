#!/usr/bin/env python
"""
Training script for Superpoint Transformer on TrueCity/Ingolstadt dataset
Uses Hydra for configuration management and PyTorch Lightning for training

NOTE: This script is EXPERIMENTAL and requires the root src/ folder which has been removed.
To use Superpoint Transformer, you need to restore src/data/superpoint_datamodule.py or
install superpoint_transformer as a package. This script may not work in the current
cleaned repository structure.
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Try to import superpoint_transformer components
# The SemanticSegmentationModule should be in the superpoint_transformer source code
# Check multiple possible locations
semantic_module = None
possible_paths = [
    os.path.join(project_root, 'src', 'models', 'semantic.py'),  # In TrueCity
    os.path.join(project_root, '..', 'EARLy', 'superpoint_transformer', 'src', 'models', 'semantic.py'),  # In EARLy
    os.path.join(project_root, '..', 'superpoint_transformer', 'src', 'models', 'semantic.py'),  # Alternative location
]

# Try to find and import
for path in possible_paths:
    if os.path.exists(path):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(path))))
        try:
            from src.models.semantic import SemanticSegmentationModule
            semantic_module = SemanticSegmentationModule
            print(f"✓ Found SemanticSegmentationModule at: {path}")
            break
        except ImportError:
            continue

# Import TrueCity DataModule
from src.data.superpoint_datamodule import TrueCityDataModule

# Check if we found the semantic module
if semantic_module is None:
    print("=" * 80)
    print("ERROR: Could not find SemanticSegmentationModule")
    print("=" * 80)
    print("\nThe Superpoint Transformer source code is required.")
    print("Please ensure one of the following:")
    print("  1. superpoint_transformer is installed as a package: pip install superpoint-transformer")
    print("  2. Source code is available at: ../EARLy/superpoint_transformer/src/")
    print("  3. Source code is in the TrueCity project: src/models/semantic.py")
    print("\nSearched paths:")
    for path in possible_paths:
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {path}")
    print("\nIf you have the source code elsewhere, add it to PYTHONPATH:")
    print("  export PYTHONPATH=/path/to/superpoint_transformer:$PYTHONPATH")
    sys.exit(1)

SemanticSegmentationModule = semantic_module


@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(cfg: DictConfig) -> None:
    """
    Main training function using Hydra configuration
    
    Args:
        cfg: Hydra configuration object
    """
    print("=" * 80)
    print("Superpoint Transformer Training - TrueCity Dataset")
    print("=" * 80)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seeds for reproducibility
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)
    
    # Setup data module
    print("\n" + "=" * 80)
    print("Setting up DataModule...")
    print("=" * 80)
    
    # Extract datamodule config
    datamodule_cfg = cfg.datamodule
    
    # Create datamodule
    datamodule = TrueCityDataModule(
        data_dir=datamodule_cfg.get('data_dir', '/home/stud/nguyenti/storage/user/EARLy/datav2_final'),
        batch_size=datamodule_cfg.dataloader.batch_size,
        num_workers=datamodule_cfg.dataloader.get('num_workers', 4),
        pin_memory=datamodule_cfg.dataloader.get('pin_memory', True),
        persistent_workers=datamodule_cfg.dataloader.get('persistent_workers', True),
        allowed_classes=datamodule_cfg.get('allowed_classes', [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
        num_classes=datamodule_cfg.num_classes,
    )
    
    # Setup datasets
    datamodule.setup(stage='fit')
    
    # Get actual number of classes from dataset
    actual_num_classes = datamodule.train_dataset.num_classes
    print(f"\n✅ DataModule setup complete")
    print(f"   Train samples: {len(datamodule.train_dataset)}")
    print(f"   Val samples: {len(datamodule.val_dataset)}")
    print(f"   Number of classes: {actual_num_classes}")
    
    # Setup model
    print("\n" + "=" * 80)
    print("Setting up Model...")
    print("=" * 80)
    
    model_cfg = cfg.model
    model_cfg.num_classes = actual_num_classes  # Use actual number from dataset
    
    # Create model
    model = SemanticSegmentationModule(
        num_classes=actual_num_classes,
        **{k: v for k, v in model_cfg.items() if k != 'num_classes'}
    )
    
    print(f"✅ Model setup complete")
    print(f"   Number of classes: {actual_num_classes}")
    
    # Setup logger
    print("\n" + "=" * 80)
    print("Setting up Logger...")
    print("=" * 80)
    
    logger = None
    if cfg.logger is not None and cfg.logger.get('_target_') is not None:
        logger_cfg = cfg.logger
        logger_type = logger_cfg.get('_target_', '').split('.')[-1]
        
        if 'WandbLogger' in logger_type:
            logger = WandbLogger(**{k: v for k, v in logger_cfg.items() if k != '_target_'})
        elif 'TensorBoardLogger' in logger_type:
            logger = TensorBoardLogger(**{k: v for k, v in logger_cfg.items() if k != '_target_'})
        elif 'CSVLogger' in logger_type:
            logger = CSVLogger(**{k: v for k, v in logger_cfg.items() if k != '_target_'})
    
    if logger is None:
        print("   Using default CSV logger")
        logger = CSVLogger(save_dir=hydra.utils.get_original_cwd() + "/logs")
    
    print(f"✅ Logger setup complete: {type(logger).__name__}")
    
    # Setup callbacks
    print("\n" + "=" * 80)
    print("Setting up Callbacks...")
    print("=" * 80)
    
    callbacks = []
    
    # Model checkpoint
    if cfg.callbacks.get('model_checkpoint') is not None:
        checkpoint_cfg = cfg.callbacks.model_checkpoint
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_cfg.get('dirpath', 'checkpoints'),
                filename=checkpoint_cfg.get('filename', 'epoch_{epoch:03d}'),
                monitor=checkpoint_cfg.get('monitor', cfg.optimized_metric),
                mode=checkpoint_cfg.get('mode', 'max'),
                save_last=checkpoint_cfg.get('save_last', True),
            )
        )
    
    # Early stopping
    if cfg.callbacks.get('early_stopping') is not None:
        early_stop_cfg = cfg.callbacks.early_stopping
        callbacks.append(
            EarlyStopping(
                monitor=early_stop_cfg.get('monitor', cfg.optimized_metric),
                mode=early_stop_cfg.get('mode', 'max'),
                patience=early_stop_cfg.get('patience', 10),
            )
        )
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    print(f"✅ Callbacks setup complete: {len(callbacks)} callbacks")
    
    # Setup trainer
    print("\n" + "=" * 80)
    print("Setting up Trainer...")
    print("=" * 80)
    
    trainer_cfg = cfg.trainer
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get('max_epochs', 100),
        min_epochs=trainer_cfg.get('min_epochs', 1),
        accelerator=trainer_cfg.get('accelerator', 'gpu'),
        devices=trainer_cfg.get('devices', 1),
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=trainer_cfg.get('check_val_every_n_epoch', 1),
        deterministic=trainer_cfg.get('deterministic', False),
        default_root_dir=trainer_cfg.get('default_root_dir', None),
    )
    
    print(f"✅ Trainer setup complete")
    print(f"   Max epochs: {trainer_cfg.get('max_epochs', 100)}")
    print(f"   Accelerator: {trainer_cfg.get('accelerator', 'gpu')}")
    print(f"   Devices: {trainer_cfg.get('devices', 1)}")
    
    # Train model
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
    if cfg.train:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get('ckpt_path', None))
    
    # Test model
    if cfg.test:
        print("\n" + "=" * 80)
        print("Starting Testing...")
        print("=" * 80)
        trainer.test(model, datamodule=datamodule)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    train()

