"""
PointNet2++ training and evaluation functions
Core training logic for PointNet2++ semantic segmentation
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from ..data.dataloader import create_ingolstadt_dataloaders
from ..models.pointnet2 import PointNet2SemanticSegmentation, create_pointnet2_segmentation
from .metrics import calculate_metrics
from ..utils.logging import setup_logger, get_log_string_function


class PointNet2Loss(nn.Module):
    """Loss function for PointNet2++ semantic segmentation"""
    def __init__(self):
        super(PointNet2Loss, self).__init__()
    
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss


def inplace_relu(m):
    """Apply inplace ReLU to model"""
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def weights_init(m):
    """Initialize model weights"""
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    """Adjust BatchNorm momentum"""
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def train_pointnet2_segmentation(data_path='/home/stud/nguyenti/storage/user/EARLy/data', 
                                 n_points=2048, batch_size=32, num_epochs=50, 
                                 learning_rate=0.0001, save_path='pointnet2_segmentation.pth',
                                 sample_multiplier=1.0, ignore_labels=None, log_func=None):
    """
    Train PointNet2++ for semantic segmentation
    
    Args:
        data_path: path to data directory
        n_points: number of points per sample
        batch_size: batch size for training
        num_epochs: number of training epochs
        learning_rate: learning rate
        save_path: path to save the model
        sample_multiplier: multiplier for samples
        ignore_labels: list of labels to ignore
        log_func: logging function (if None, will create one)
    """
    # Setup logging if not provided
    if log_func is None:
        logger = setup_logger(name="PointNet2++")
        log_func = get_log_string_function(logger)
    
    log_func("🚀 Starting PointNet2++ semantic segmentation training...")
    
    # Setup device
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
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    # Create output directory with dataset name
    timestr = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    
    # Extract dataset name from data_path
    dataset_name = Path(data_path).name
    if not dataset_name:
        dataset_name = Path(data_path).parent.name
    
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('pointnet2_sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(f'{timestr}_{dataset_name}')
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    log_func(f"📁 Experiment directory: {experiment_dir}")
    log_func(f"📊 Dataset: {dataset_name}")
    
    log_func("🏗️ Setting up PointNet2++ SEMANTIC SEGMENTATION training...")
    log_func(f"📁 Data path: {data_path}")
    log_func(f"🎯 Points per sample: {n_points}")
    log_func(f"📦 Batch size: {batch_size}")
    log_func(f"🔄 Epochs: {num_epochs}")
    log_func(f"🎨 Task: Semantic Segmentation (per-point classification)")
    if ignore_labels:
        log_func(f"🚫 Ignoring labels: {ignore_labels}")
    
    # Load dataset
    log_func("📂 Loading dataset...")
    
    # Auto-detect precomputed data
    is_precomputed = True  # PointNet2++ typically uses precomputed FPS data
    if "_fps" in data_path.lower() or "octree_fps" in data_path.lower():
        is_precomputed = True
        log_func("🚀 Detected precomputed FPS dataset - no sampling needed!")
    elif "grid" in data_path.lower() or len(glob.glob(os.path.join(data_path, "train*chunk*.npy"))) > 5:
        log_func("🔲 Detected grid-based dataset")
    
    # Use fixed 10-class mapping to match original training
    # Original training used: [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    ALLOWED_CLASSES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    log_func(f"🎯 Using fixed class mapping: {ALLOWED_CLASSES} (12 classes)")
    
    data_setup = create_ingolstadt_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        n_points=n_points,
        num_workers=4,
        sample_multiplier=sample_multiplier,
        is_precomputed=is_precomputed,
        log_func=log_func,
        allowed_classes=ALLOWED_CLASSES
    )
    
    dataloaders = data_setup['dataloaders']
    info = data_setup['info']
    
    # Validate class consistency across splits
    train_classes = info['num_classes']
    log_func(f"🔍 Validating class consistency...")
    log_func(f"   Training classes: {train_classes}")
    
    # Collect all class mappings from all splits
    all_class_mappings = [info['class_to_idx']]
    max_classes = train_classes
    unified_class_to_idx = info['class_to_idx'].copy()
    
    # Check validation classes if available
    if dataloaders['val'] is not None and hasattr(data_setup['datasets']['val'], 'num_classes'):
        val_classes = data_setup['datasets']['val'].num_classes
        log_func(f"   Validation classes: {val_classes}")
        if val_classes != train_classes:
            log_func(f"⚠️ WARNING: Class mismatch between train ({train_classes}) and val ({val_classes})")
            max_classes = max(max_classes, val_classes)
            val_mapping = data_setup['datasets']['val'].class_to_idx
            for cls, idx in val_mapping.items():
                if cls not in unified_class_to_idx:
                    unified_class_to_idx[cls] = len(unified_class_to_idx)
    
    # Check test classes if available
    if dataloaders['test'] is not None and hasattr(data_setup['datasets']['test'], 'num_classes'):
        test_classes = data_setup['datasets']['test'].num_classes
        log_func(f"   Test classes: {test_classes}")
        if test_classes != train_classes:
            log_func(f"⚠️ WARNING: Class mismatch between train ({train_classes}) and test ({test_classes})")
            max_classes = max(max_classes, test_classes)
            test_mapping = data_setup['datasets']['test'].class_to_idx
            for cls, idx in test_mapping.items():
                if cls not in unified_class_to_idx:
                    unified_class_to_idx[cls] = len(unified_class_to_idx)
    
    # Update info with unified mapping if there was a mismatch
    if max_classes != train_classes:
        log_func(f"🔧 Using {max_classes} classes for model to handle all splits")
        info['num_classes'] = max_classes
        info['class_to_idx'] = unified_class_to_idx
        info['idx_to_class'] = {v: k for k, v in unified_class_to_idx.items()}
        log_func(f"🎯 Unified class mapping: {unified_class_to_idx}")
    
    num_classes = info['num_classes']
    
    log_func(f"🤖 Creating PointNet2++ semantic segmentation model...")
    log_func(f"📊 Model created for {num_classes} classes")
    log_func(f"🏷️ Classes: {list(info['class_to_idx'].keys())}")
    
    # Create model
    model = PointNet2SemanticSegmentation(num_classes).to(device)
    criterion = PointNet2Loss().to(device)
    model.apply(inplace_relu)

    # Setup optimizer BEFORE loading checkpoint
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )

    # Try to load existing model AND optimizer state
    start_epoch = 0
    try:
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            start_epoch = checkpoint.get('epoch', 0)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log_func('🔄 Loaded model and optimizer state from checkpoint')
            else:
                log_func('🔄 Loaded model weights only (no optimizer state found)')
                
            log_func(f'📍 Resuming from epoch {start_epoch}')
        else:
            log_func('🆕 No existing model found, starting fresh training...')
            model = model.apply(weights_init)
    except Exception as e:
        log_func(f'⚠️ Error loading checkpoint: {e}')
        log_func('🆕 Starting fresh training...')
        start_epoch = 0
        model = model.apply(weights_init)
    
    # Training parameters
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = 10
    step_size = 10
    lr_decay = 0.7
    
    # Class weights (equal weights for now)
    weights = torch.ones(num_classes).to(device)
    
    log_func("🚀 Starting training...")
    log_func(f"📍 Training for {num_epochs} epochs")
    log_func(f"📊 Training batches per epoch: {len(dataloaders['train'])}")
    
    best_iou = 0
    train_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        log_func(f'**** Epoch {epoch + 1} ({epoch + 1}/{num_epochs}) ****')
        
        # Learning rate scheduling
        lr = max(learning_rate * (lr_decay ** (epoch // step_size)), LEARNING_RATE_CLIP)
        log_func(f'Learning rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # BatchNorm momentum adjustment
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_func(f'BN momentum updated to: {momentum}')
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        
        # Training phase
        model.train()
        num_batches = len(dataloaders['train'])
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(num_classes)]
        total_correct_class = [0 for _ in range(num_classes)]
        total_iou_deno_class = [0 for _ in range(num_classes)]
        
        train_pbar = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            
            points = batch['points'].to(device)  # [B, N, 3]
            target = batch['point_labels'].to(device)  # [B, N]
            
            # Filter out ignored labels
            if ignore_labels:
                valid_mask = torch.ones_like(target, dtype=torch.bool)
                for ignore_label in ignore_labels:
                    valid_mask &= (target != ignore_label)
                
                if torch.any(valid_mask):
                    points = points[valid_mask.view(points.shape[0], -1)]
                    target = target[valid_mask]
                    
                    # Reshape points back to [B, N, 3] format
                    if len(points.shape) == 2:
                        points = points.view(-1, 3)
                        target = target.view(-1)
                        
                        # Ensure we have enough points for the model
                        if len(points) < n_points:
                            pad_size = n_points - len(points)
                            pad_indices = torch.randint(0, len(points), (pad_size,))
                            pad_points = points[pad_indices]
                            pad_target = target[pad_indices]
                            points = torch.cat([points, pad_points], dim=0)
                            target = torch.cat([target, pad_target], dim=0)
                        elif len(points) > n_points:
                            indices = torch.randperm(len(points))[:n_points]
                            points = points[indices]
                            target = target[indices]
                        
                        points = points.view(1, -1, 3)
                        target = target.view(1, -1)
                else:
                    continue
            
            # Transpose for PointNet2++: [B, N, 3] -> [B, 3, N]
            points = points.transpose(2, 1)
            
            # Forward pass
            seg_pred, trans_feat = model(points)  # [B, N, num_classes]
            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            
            loss = criterion(seg_pred, target, trans_feat, weights)

            # Check for NaN loss and stop training if detected
            if torch.isnan(loss) or torch.isinf(loss):
                log_func(f"❌ NaN or Inf loss detected! Loss: {loss.item()}")
                log_func("Stopping training to prevent corruption...")
                break

            loss.backward()

            # Add gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            # Statistics
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (batch_size * n_points)
            loss_sum += loss.item()
            
            for l in range(num_classes):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))
            
            # Update progress bar
            current_acc = 100. * total_correct / total_seen
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate training metrics
        train_mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
        train_loss = loss_sum / num_batches
        train_acc = total_correct / float(total_seen)
        train_class_acc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))
        train_losses.append(train_loss)
        
        log_func(f"\n📈 TRAINING RESULTS - Epoch {epoch+1}:")
        log_func(f"   Training Loss: {train_loss:.6f}")
        log_func(f"   Training Accuracy: {train_acc*100:.2f}%")
        log_func(f"   Training mIoU: {train_mIoU*100:.2f}%")
        log_func(f"   Training Class Accuracy: {train_class_acc*100:.2f}%")
        
        # Save model every 5 epochs
        if epoch % 5 == 0:
            log_func('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_func(f'Saving at {savepath}')
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': info['class_to_idx'],
                'num_classes': num_classes,
            }
            torch.save(state, savepath)
            # Also save to main save_path
            torch.save(state, save_path)
            log_func('Saving model....')
        
        # Validation phase
        if dataloaders['val'] is not None:
            model.eval()
            with torch.no_grad():
                num_batches = len(dataloaders['val'])
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                total_seen_class = [0 for _ in range(num_classes)]
                total_correct_class = [0 for _ in range(num_classes)]
                total_iou_deno_class = [0 for _ in range(num_classes)]
                
                log_func('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
                val_pbar = tqdm(dataloaders['val'], desc="Validation")
                
                # Collect all predictions and targets for comprehensive metrics
                all_predictions = []
                all_targets = []
                
                for i, batch in enumerate(val_pbar):
                    points = batch['points'].to(device)
                    target = batch['point_labels'].to(device)
                    
                    # Filter out ignored labels for validation too
                    if ignore_labels:
                        valid_mask = torch.ones_like(target, dtype=torch.bool)
                        for ignore_label in ignore_labels:
                            valid_mask &= (target != ignore_label)
                        
                        if torch.any(valid_mask):
                            points = points[valid_mask.view(points.shape[0], -1)]
                            target = target[valid_mask]
                            
                            if len(points.shape) == 2:
                                points = points.view(-1, 3)
                                target = target.view(-1)
                                
                                if len(points) < n_points:
                                    pad_size = n_points - len(points)
                                    pad_indices = torch.randint(0, len(points), (pad_size,))
                                    pad_points = points[pad_indices]
                                    pad_target = target[pad_indices]
                                    points = torch.cat([points, pad_points], dim=0)
                                    target = torch.cat([target, pad_target], dim=0)
                                elif len(points) > n_points:
                                    indices = torch.randperm(len(points))[:n_points]
                                    points = points[indices]
                                    target = target[indices]
                                
                                points = points.view(1, -1, 3)
                                target = target.view(1, -1)
                        else:
                            continue
                    
                    points = points.transpose(2, 1)
                    
                    seg_pred, trans_feat = model(points)
                    pred_val = seg_pred.contiguous().cpu().data.numpy()
                    seg_pred = seg_pred.contiguous().view(-1, num_classes)
                    
                    batch_label = target.cpu().data.numpy()
                    target_loss = target.view(-1, 1)[:, 0]
                    
                    pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                    correct = np.sum(pred_choice == batch_label.flatten())
                    total_correct += correct
                    total_seen += (batch_label.size)
                    loss_sum += criterion(seg_pred, target_loss, trans_feat, weights).item()
                    
                    tmp, _ = np.histogram(batch_label.flatten(), range(num_classes + 1))
                    total_seen_class += tmp
                    
                    tmp, _ = np.histogram(pred_choice, range(num_classes + 1))
                    total_correct_class += np.minimum(tmp, total_seen_class)
                    
                    for l in range(num_classes):
                        total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label.flatten() == l)))
                    
                    # Collect for comprehensive metrics
                    all_predictions.append(pred_choice)
                    all_targets.append(batch_label.flatten())
                    
                    # Update validation progress bar
                    current_acc = 100. * correct / batch_label.size if batch_label.size > 0 else 0
                    val_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
                
                # Calculate validation metrics
                val_mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
                val_loss = loss_sum / num_batches
                val_acc = total_correct / float(total_seen)
                
                log_func(f"\n📊 VALIDATION RESULTS - Epoch {epoch+1}:")
                log_func(f"   Validation Loss: {val_loss:.6f}")
                log_func(f"   Validation Accuracy: {val_acc*100:.2f}%")
                log_func(f"   Validation mIoU: {val_mIoU*100:.2f}%")
                
                # Save best model
                if val_mIoU > best_iou:
                    best_iou = val_mIoU
                    log_func(f'💾 New best model! mIoU: {val_mIoU*100:.2f}%')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_miou': best_iou,
                        'val_accuracy': val_acc,
                        'class_to_idx': info['class_to_idx'],
                        'num_classes': num_classes,
                    }
                    torch.save(state, savepath)
                    torch.save(state, save_path)
    
    log_func(f'🏆 Training completed! Best validation mIoU: {best_iou*100:.2f}%')
    
    return {
        'model': model,
        'train_losses': train_losses,
        'best_val_miou': best_iou,
        'info': info
    }


def evaluate_model(model_path, data_path='/home/stud/nguyenti/storage/user/EARLy/data', 
                  n_points=1024, batch_size=32, ignore_labels=None, log_func=None):
    """
    Evaluate a trained PointNet2++ model on test data
    
    Args:
        model_path: str - path to saved model
        data_path: str - path to data folder
        n_points: int - number of points per sample
        batch_size: int - batch size for evaluation
        ignore_labels: list - labels to ignore during evaluation
        log_func: function - logging function (if None, will create one)
    """
    # Setup logging if not provided
    if log_func is None:
        logger = setup_logger(name="PointNet2++")
        log_func = get_log_string_function(logger)
    
    log_func(f"🔍 Evaluating model: {model_path}")
    
    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_func(f"💻 Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load test data
    is_precomputed = True
    # Use fixed 10-class mapping to match original training
    ALLOWED_CLASSES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    
    data_setup = create_ingolstadt_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        n_points=n_points,
        num_workers=4,
        is_precomputed=is_precomputed,
        log_func=log_func,
        allowed_classes=ALLOWED_CLASSES
    )
    
    # Determine number of classes from model architecture
    final_layer_key = 'conv2.weight'  # Final classification layer
    if final_layer_key in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict'][final_layer_key].shape[0]
        log_func(f"🔍 Detected {num_classes} classes from model architecture")
    else:
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
            log_func("🔍 Using num_classes from checkpoint")
        else:
            num_classes = len(data_setup['datasets']['test'].class_to_idx)
            log_func("⚠️ Using num_classes from dataset (may cause issues)")
    
    # Create model with correct number of classes
    model = PointNet2SemanticSegmentation(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get class mapping for evaluation metrics
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        log_func("🔍 Using class mapping from model checkpoint")
    else:
        log_func("⚠️ No class mapping in checkpoint, creating dummy mapping...")
        class_to_idx = {f"class_{i}": i for i in range(num_classes)}
        log_func("✅ Created dummy class mapping for evaluation")
    
    log_func(f"📊 Model classes ({len(class_to_idx)}): {list(class_to_idx.keys())}")
    
    test_loader = data_setup['dataloaders']['test']
    
    if test_loader is None:
        log_func("❌ No test data available!")
        return 0.0
    
    # Evaluation
    log_func("🧪 Running evaluation...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc="Evaluating")
        for batch in eval_pbar:
            points = batch['points'].to(device)
            target = batch['point_labels'].to(device)  # [B, N] - per-point labels
            
            # Filter out ignored labels for test evaluation
            if ignore_labels:
                valid_mask = torch.ones_like(target, dtype=torch.bool)
                for ignore_label in ignore_labels:
                    valid_mask &= (target != ignore_label)
                
                if torch.any(valid_mask):
                    points = points[valid_mask.view(points.shape[0], -1)]
                    target = target[valid_mask]
                    
                    if len(points.shape) == 2:
                        points = points.view(-1, 3)
                        target = target.view(-1)
                        
                        if len(points) < n_points:
                            pad_size = n_points - len(points)
                            pad_indices = torch.randint(0, len(points), (pad_size,))
                            pad_points = points[pad_indices]
                            pad_target = target[pad_indices]
                            points = torch.cat([points, pad_points], dim=0)
                            target = torch.cat([target, pad_target], dim=0)
                        elif len(points) > n_points:
                            indices = torch.randperm(len(points))[:n_points]
                            points = points[indices]
                            target = target[indices]
                        
                        points = points.view(1, -1, 3)
                        target = target.view(1, -1)
                else:
                    continue
            
            points = points.transpose(2, 1)
            
            seg_pred, _ = model(points)  # [B, N, num_classes]
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            pred_val = np.argmax(pred_val, 2)
            batch_label = target.cpu().data.numpy()
            
            # Collect predictions and targets for comprehensive metrics
            all_predictions.append(pred_val.flatten())
            all_targets.append(batch_label.flatten())
            
            # Update evaluation progress bar
            current_correct = np.sum(pred_val == batch_label)
            current_total = batch_label.size
            current_acc = 100. * current_correct / current_total if current_total > 0 else 0
            eval_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    # Calculate comprehensive metrics
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
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
            class_name = str(idx_to_class.get(class_idx, f"class_{class_idx}"))
            class_acc = test_metrics['class_accuracy'][class_idx]
            class_iou = test_metrics['iou_per_class'][class_idx]
            class_prec = test_metrics.get('precision_per_class', [0]*num_classes)[class_idx]
            class_rec = test_metrics.get('recall_per_class', [0]*num_classes)[class_idx]
            class_f1 = test_metrics.get('f1_per_class', [0]*num_classes)[class_idx]
            class_count = int(test_metrics['class_counts'][class_idx])
            log_func(f"   {class_name}: Acc={class_acc:.2f}%, IoU={class_iou:.2f}%, Prec={class_prec:.2f}%, Rec={class_rec:.2f}%, F1={class_f1:.2f}% (n={class_count:,})")
    
    return test_metrics['overall_accuracy']

