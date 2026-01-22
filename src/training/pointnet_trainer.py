"""
PointNet training and evaluation functions
Core training logic for PointNet semantic segmentation
"""

import os
import glob
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from ..data.dataloader import create_ingolstadt_dataloaders
from ..models.pointnet import create_pointnet_segmentation
from .metrics import calculate_metrics
from ..utils.logging import setup_logger, get_log_string_function


def train_ingolstadt_segmentation(data_path='/home/stud/nguyenti/storage/user/EARLy/data', 
                                 n_points=2048, batch_size=8, num_epochs=50, 
                                 learning_rate=0.001, save_path='ingolstadt_pointnet_segmentation.pth',
                                 sample_multiplier=1.0, ignore_labels=None, log_func=None):
    """
    Complete training pipeline for PointNet Semantic Segmentation on Ingolstadt dataset
    
    Args:
        data_path: str - path to Ingolstadt data folder
        n_points: int - number of points to sample per point cloud using FPS
        batch_size: int - training batch size  
        num_epochs: int - number of training epochs
        learning_rate: float - learning rate for optimizer
        save_path: str - path to save/resume model (automatically resumes if exists)
        sample_multiplier: float - multiplier for samples (1.0=normal, 0.2=faster training)
        ignore_labels: list - labels to ignore during training
        log_func: function - logging function (if None, will create one)
    """
    
    # Setup logging if not provided
    if log_func is None:
        logger = setup_logger(name="PointNet")
        log_func = get_log_string_function(logger)
    
    log_func("🏗️ Setting up Ingolstadt SEMANTIC SEGMENTATION training...")
    log_func(f"📁 Data path: {data_path}")
    log_func(f"🎯 Points per sample: {n_points} (using FPS)")
    log_func(f"📦 Batch size: {batch_size}")
    log_func(f"🔄 Epochs: {num_epochs}")
    log_func(f"🎨 Task: Semantic Segmentation (per-point classification)")
    if ignore_labels:
        log_func(f"🚫 Ignoring labels: {ignore_labels}")
    
    # Check GPU availability and optimize settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_func(f"💻 Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log_func(f"🎮 GPU: {gpu_name}")
        log_func(f"💾 GPU Memory: {gpu_memory:.1f}GB")
        
        # Optimize for large GPU memory (48GB)
        if gpu_memory > 40:
            log_func("🚀 Large GPU detected - optimizing performance settings...")
            # Enable optimizations for large memory
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    # 1. Create dataloaders 
    log_func("\n📂 Loading dataset...")
    
    # Auto-detect data type
    is_precomputed = False
    is_grid_data = False
    
    # Check for precomputed FPS data (many small files with _fps pattern)
    if "_fps" in data_path.lower() or "octree_fps" in data_path.lower():
        is_precomputed = True
        log_func("🚀 Detected precomputed FPS dataset - no sampling needed!")
    
    # Check for grid data (many small chunks, but not precomputed)
    elif "grid" in data_path.lower() or len(glob.glob(os.path.join(data_path, "train*chunk*.npy"))) > 5:
        is_grid_data = True
        log_func("🔲 Detected grid-based dataset - fast FPS sampling on small chunks!")
        # For grid data, we can use the full sample_multiplier since files are small
        log_func(f"   🎯 Using sample_multiplier={sample_multiplier} for grid data")
    
    # Check for large single files
    else:
        large_files = glob.glob(os.path.join(data_path, "train*.npy"))
        if large_files:
            log_func("📄 Detected large single files - will need FPS sampling during training")
        else:
            log_func("❓ Data format auto-detection uncertain - using default settings")
    
    # Optimize data loading - use system recommended workers
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    optimal_workers = 4  # Use system recommended number for stability
    log_func(f"🔧 Using {optimal_workers} data workers for optimal performance")
    
    # Use fixed 10-class mapping to match original training
    # Original training used: [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    ALLOWED_CLASSES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    log_func(f"🎯 Using fixed class mapping: {ALLOWED_CLASSES} (12 classes)")
    
    data_setup = create_ingolstadt_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        n_points=n_points,
        num_workers=optimal_workers,  # Optimized for large GPU
        is_precomputed=is_precomputed,
        sample_multiplier=sample_multiplier if not is_precomputed else 1.0,
        log_func=log_func,
        allowed_classes=ALLOWED_CLASSES
    )
    
    dataloaders = data_setup['dataloaders']
    info = data_setup['info']
    
    # 2. Create model and training setup
    log_func("\n🤖 Creating semantic segmentation model...")
    
    # Validate class consistency across splits
    train_classes = info['num_classes']
    log_func(f"🔍 Validating class consistency...")
    log_func(f"   Training classes: {train_classes}")
    
    # Check validation classes if available
    if dataloaders['val'] is not None and hasattr(data_setup['datasets']['val'], 'num_classes'):
        val_classes = data_setup['datasets']['val'].num_classes
        log_func(f"   Validation classes: {val_classes}")
        if val_classes != train_classes:
            log_func(f"⚠️ WARNING: Class mismatch between train ({train_classes}) and val ({val_classes})")
    
    # Check test classes if available
    if dataloaders['test'] is not None and hasattr(data_setup['datasets']['test'], 'num_classes'):
        test_classes = data_setup['datasets']['test'].num_classes
        log_func(f"   Test classes: {test_classes}")
        if test_classes != train_classes:
            log_func(f"⚠️ WARNING: Class mismatch between train ({train_classes}) and test ({test_classes})")
    
    model, criterion = create_pointnet_segmentation(
        num_classes=info['num_classes'],
        feature_transform=True,
        channel=3
    )
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.7
    )
    
    log_func(f"📊 Model created for {info['num_classes']} classes")
    log_func(f"🏷️ Classes: {list(info['class_to_idx'].keys())}")
    
    # Check for existing model to resume from
    start_epoch = 0
    best_val_acc = 0.0
    best_val_miou = 0.0
    train_losses = []
    val_accuracies = []
    val_mious = []
    
    if os.path.exists(save_path):
        log_func(f"\n🔄 Found existing model, resuming training from: {save_path}")
        try:
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            
            # Restore model and optimizer state
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_accuracy', checkpoint.get('val_accuracy', 0.0))
            best_val_miou = checkpoint.get('best_val_miou', 0.0)
            
            # Restore training history if available
            train_losses = checkpoint.get('train_losses', [])
            val_accuracies = checkpoint.get('val_accuracies', [])
            val_mious = checkpoint.get('val_mious', [])
            
            # Update scheduler to the correct step
            for _ in range(start_epoch):
                scheduler.step()
            
            log_func(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%, best mIoU: {best_val_miou:.2f}%")
            log_func(f"📈 Training history: {len(train_losses)} losses, {len(val_accuracies)} val accuracies, {len(val_mious)} mIoUs")
            
        except Exception as e:
            log_func(f"⚠️ Error loading checkpoint: {e}")
            log_func("🆕 Starting fresh training...")
            start_epoch = 0
            best_val_acc = 0.0
            best_val_miou = 0.0
            train_losses = []
            val_accuracies = []
            val_mious = []
    else:
        log_func("🆕 No existing model found, starting fresh training...")
    
    # 3. Training loop
    log_func(f"\n🚀 {'Resuming' if start_epoch > 0 else 'Starting'} training...")
    if start_epoch > 0:
        log_func(f"📍 Starting from epoch {start_epoch+1}/{num_epochs}")
    else:
        log_func(f"📍 Training for {num_epochs} epochs")
    
    # Safety check for empty dataloader
    if len(dataloaders['train']) == 0:
        log_func("❌ Error: Training dataloader is empty!")
        log_func("💡 This might be because:")
        log_func("   • Dataset has no samples")
        log_func("   • Batch size is larger than dataset size")
        log_func("   • Data loading failed")
        return None
    
    log_func(f"📊 Training batches per epoch: {len(dataloaders['train'])}")
    
    # Initialize training metrics
    train_loss = 0.0
    val_acc = 0.0
    val_miou = 0.0
    
    # Don't train if we've already completed all epochs
    if start_epoch >= num_epochs:
        log_func(f"✅ Training already completed ({start_epoch}/{num_epochs} epochs)")
        # Run final test evaluation and exit
        if dataloaders['test'] is not None:
            log_func("\n🧪 Running comprehensive test evaluation...")
            final_test_acc = evaluate_model(
                model_path=save_path,
                data_path=data_path,
                n_points=n_points,
                batch_size=batch_size,
                log_func=log_func
            )
            log_func(f'🏆 Test accuracy: {final_test_acc:.2f}%')
        return {
            'model': None,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'final_test_acc': final_test_acc if 'final_test_acc' in locals() else 0.0,
            'info': info
        }
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        log_func(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        batch_count = 0
        total_batches = len(dataloaders['train'])
        mid_batch = total_batches // 2
        
        # Create progress bar for training batches
        pbar = tqdm(enumerate(dataloaders['train']), 
                   total=len(dataloaders['train']),
                   desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in pbar:
            # Get data
            points = batch['points'].to(device)  # [B, N, 3]
            target = batch['point_labels'].to(device)  # [B, N] - per-point labels
            
            # Transpose for PointNet: [B, N, 3] -> [B, 3, N]
            points = points.transpose(2, 1)
            
            # Forward pass
            optimizer.zero_grad()
            pred, trans_feat = model(points)  # [B, N, num_classes]
            
            # Reshape for loss computation
            pred = pred.contiguous().view(-1, info['num_classes'])  # [B*N, num_classes]
            target = target.view(-1)  # [B*N]
            
            loss = criterion(pred, target, trans_feat)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics (per-point accuracy)
            epoch_train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            batch_count += 1
            
            # Update progress bar with current metrics
            current_acc = 100. * train_correct / train_total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate training metrics (with safety check)
        train_loss = epoch_train_loss / batch_count
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)

        
        # Validation phase (if available)
        epoch_val_acc = 0.0
        epoch_val_miou = 0.0
        if dataloaders['val'] is not None:
            model.eval()
            
            # Initialize incremental metric calculation
            total_correct = 0
            total_points = 0
            class_intersections = torch.zeros(info['num_classes'])
            class_unions = torch.zeros(info['num_classes'])
            
            with torch.no_grad():
                val_pbar = tqdm(dataloaders['val'], desc="Validation", leave=False)
                for batch in val_pbar:
                    points = batch['points'].to(device)
                    target = batch['point_labels'].to(device)  # [B, N]
                    points = points.transpose(2, 1)
                    
                    pred, _ = model(points)  # [B, N, num_classes]
                    pred = pred.contiguous().view(-1, info['num_classes'])  # [B*N, num_classes]
                    target = target.view(-1)  # [B*N]
                    
                    _, predicted = torch.max(pred, 1)
                    
                    # Calculate accuracy incrementally
                    batch_correct = (predicted == target).sum().item()
                    batch_total = target.size(0)
                    total_correct += batch_correct
                    total_points += batch_total
                    
                    # Calculate IoU incrementally for each class
                    predicted_cpu = predicted.cpu()
                    target_cpu = target.cpu()
                    
                    for class_idx in range(info['num_classes']):
                        pred_mask = (predicted_cpu == class_idx)
                        target_mask = (target_cpu == class_idx)
                        intersection = (pred_mask & target_mask).sum().item()
                        union = (pred_mask | target_mask).sum().item()
                        
                        class_intersections[class_idx] += intersection
                        class_unions[class_idx] += union
                    
                    # Update validation progress bar with current accuracy
                    current_acc = 100. * batch_correct / batch_total if batch_total > 0 else 0
                    val_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
            
            # Calculate final metrics
            epoch_val_acc = 100. * total_correct / total_points if total_points > 0 else 0
            
            # Calculate mIoU
            valid_classes = 0
            total_iou = 0
            per_class_ious = {}
            
            for class_idx in range(info['num_classes']):
                if class_unions[class_idx] > 0:
                    iou = class_intersections[class_idx] / class_unions[class_idx]
                    per_class_ious[class_idx] = iou
                    total_iou += iou
                    valid_classes += 1
            
            epoch_val_miou = (total_iou / valid_classes * 100) if valid_classes > 0 else 0
            
            val_accuracies.append(epoch_val_acc)
            val_mious.append(epoch_val_miou)
            
            log_func(f"📊 Validation Results - Acc: {epoch_val_acc:.2f}%, mIoU: {epoch_val_miou:.2f}%")
            
            # Save best model based on mIoU (primary metric)
            if epoch_val_miou > best_val_miou:
                best_val_miou = epoch_val_miou
                best_val_acc = epoch_val_acc  # Update accuracy too
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_accuracy': epoch_val_acc,
                    'val_miou': epoch_val_miou,
                    'best_val_accuracy': best_val_acc,
                    'best_val_miou': best_val_miou,
                    'train_losses': train_losses,
                    'val_accuracies': val_accuracies,
                    'val_mious': val_mious,
                    'class_to_idx': info['class_to_idx'],
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate
                }, save_path)
                log_func(f'💾 New best model saved! mIoU: {epoch_val_miou:.2f}%, Acc: {epoch_val_acc:.2f}%')
        else:
            # No validation data available
            epoch_val_miou = 0.0
        
        # Update learning rate
        scheduler.step()
        
        # Update final validation metrics for this epoch
        val_acc = epoch_val_acc
        val_miou = epoch_val_miou
        
        log_func(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val mIoU: {val_miou:.2f}%')
        
        # Save checkpoint every 10 epochs (for resuming)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_path.replace('.pth', f'_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_accuracy': val_acc,
                'val_miou': val_miou,
                'best_val_accuracy': best_val_acc,
                'best_val_miou': best_val_miou,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'val_mious': val_mious,
                'class_to_idx': info['class_to_idx'],
                'num_epochs': num_epochs,
                'learning_rate': learning_rate
            }, checkpoint_path)
            log_func(f'📁 Checkpoint saved: {checkpoint_path}')
    
    # Final comprehensive test evaluation
    final_test_acc = 0.0
    if dataloaders['test'] is not None:
        log_func("\n🧪 Running comprehensive test evaluation...")
        
        # Save final model state
        torch.save({
            'epoch': num_epochs-1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_accuracy': val_acc,
            'val_miou': val_miou,
            'best_val_accuracy': best_val_acc,
            'best_val_miou': best_val_miou,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'val_mious': val_mious,
            'class_to_idx': info['class_to_idx'],
            'num_epochs': num_epochs,
            'learning_rate': learning_rate
        }, save_path)
        
        # Run test evaluation on the BEST model (not final model)
        # The best model is already saved in save_path from the training loop
        log_func("🏆 Using BEST model for test evaluation (highest validation accuracy)")
        final_test_acc = evaluate_model(
            model_path=save_path,
            data_path=data_path,
            n_points=n_points,
            batch_size=batch_size,
            log_func=log_func
        )
    
    log_func(f'🏆 Training completed! Best validation mIoU: {best_val_miou:.2f}%, Best accuracy: {best_val_acc:.2f}%')
    if final_test_acc > 0:
        log_func(f'🏆 Final test accuracy: {final_test_acc:.2f}%')
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'final_test_acc': final_test_acc,
        'info': info
    }


def evaluate_model(model_path, data_path='/home/stud/nguyenti/storage/user/EARLy/data', 
                  n_points=1024, batch_size=32, log_func=None):
    """
    Evaluate a trained model on test data
    
    Args:
        model_path: str - path to saved model
        data_path: str - path to data folder
        n_points: int - number of points per sample
        batch_size: int - batch size for evaluation
        log_func: function - logging function (if None, will create one)
    """
    # Setup logging if not provided
    if log_func is None:
        logger = setup_logger(name="PointNet")
        log_func = get_log_string_function(logger)
    
    log_func(f"🔍 Evaluating model: {model_path}")
    
    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_func(f"💻 Using device: {device}")
    
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)
    
    log_func(f"📊 Model classes: {list(class_to_idx.keys())}")
    
    # Create model
    model, criterion = create_pointnet_segmentation(
        num_classes=num_classes,
        feature_transform=True,
        channel=3
    )
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data with auto-detection
    is_precomputed = False
    
    # Auto-detect data type (same logic as training)
    if "_fps" in data_path.lower() or "octree_fps" in data_path.lower():
        is_precomputed = True
        log_func("🚀 Detected precomputed FPS dataset for evaluation")
    elif "grid" in data_path.lower() or len(glob.glob(os.path.join(data_path, "test*chunk*.npy"))) > 5:
        log_func("🔲 Detected grid-based dataset for evaluation")
    
    # Optimize for evaluation performance
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    optimal_workers = 4  # Use system recommended number for stability
    
    # Use fixed 10-class mapping to match original training
    ALLOWED_CLASSES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    
    data_setup = create_ingolstadt_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        n_points=n_points,
        num_workers=optimal_workers,  # Optimized for large GPU
        is_precomputed=is_precomputed,
        log_func=log_func,
        allowed_classes=ALLOWED_CLASSES
    )
    
    test_loader = data_setup['dataloaders']['test']
    
    # Evaluation
    log_func("🧪 Running evaluation...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc="Evaluating")
        for batch in eval_pbar:
            points = batch['points'].to(device)
            target = batch['point_labels'].to(device)  # [B, N] - per-point labels
            points = points.transpose(2, 1)
            
            pred, _ = model(points)  # [B, N, num_classes]
            pred = pred.contiguous().view(-1, num_classes)  # [B*N, num_classes]
            target = target.view(-1)  # [B*N]
            
            _, predicted = torch.max(pred, 1)
            
            # Collect predictions and targets for comprehensive metrics
            all_predictions.append(predicted.cpu())
            all_targets.append(target.cpu())
            
            # Update evaluation progress bar
            current_correct = (predicted == target).sum().item()
            current_total = target.size(0)
            current_acc = 100. * current_correct / current_total if current_total > 0 else 0
            eval_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    # Calculate comprehensive metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    test_metrics = calculate_metrics(
        all_predictions, all_targets, 
        num_classes, class_to_idx
    )
    
    # Print comprehensive results
    log_func(f"\n🏆 COMPREHENSIVE TEST RESULTS:")
    log_func(f"   Overall Accuracy: {test_metrics['overall_accuracy']:.2f}%")
    log_func(f"   Mean IoU (mIoU): {test_metrics['mean_iou']:.2f}%")
    log_func(f"   Mean Class Accuracy: {test_metrics['mean_class_accuracy']:.2f}%")
    
    log_func(f"\n📊 Per-Class Results:")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    for class_idx in range(num_classes):
        if test_metrics['class_counts'][class_idx] > 0:
            class_name = idx_to_class[class_idx]
            class_acc = test_metrics['class_accuracy'][class_idx]
            class_iou = test_metrics['iou_per_class'][class_idx]
            class_count = int(test_metrics['class_counts'][class_idx])
            log_func(f"   {class_name}: Acc={class_acc:.2f}%, IoU={class_iou:.2f}% (n={class_count:,})")
    
    return test_metrics['overall_accuracy']

