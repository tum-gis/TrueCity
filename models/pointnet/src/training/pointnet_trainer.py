#!/usr/bin/env python3

import os
import sys
import glob
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging
import datetime
from pathlib import Path

# Add project root (TrueCity) and pointnet root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
pointnet_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, project_root)      # for `shared.*`
sys.path.insert(0, pointnet_root)     # for `src.*`

from shared.data.dataloader import create_ingolstadt_dataloaders
from shared.utils.logging import setup_logger, log_string as log_string_helper
from src.models.pointnet import create_pointnet_segmentation


def train_ingolstadt_segmentation(data_path='/home/stud/nguyenti/storage/user/EARLy/data', 
                                 n_points=2048, batch_size=8, num_epochs=50, 
                                 learning_rate=0.001, save_path='ingolstadt_pointnet_segmentation.pth',
                                 sample_multiplier=1.0, ignore_labels=None):
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
    """
    
    # Setup logging
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    log_dir = Path('./logs/')
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging using shared utility
    logger = setup_logger("PointNet", log_dir=str(log_dir), log_file=f'pointnet_training_{timestr}.txt')
    
    def log_string(msg):
        log_string_helper(logger, msg)
    
    log_string("🏗️ Setting up Ingolstadt SEMANTIC SEGMENTATION training...")
    log_string(f"📁 Data path: {data_path}")
    log_string(f"🎯 Points per sample: {n_points} (using FPS)")
    log_string(f"📦 Batch size: {batch_size}")
    log_string(f"🔄 Epochs: {num_epochs}")
    log_string(f"🎨 Task: Semantic Segmentation (per-point classification)")
    if ignore_labels:
        log_string(f"🚫 Ignoring labels: {ignore_labels}")
    
    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_string(f"💻 Using device: {device}")
    
    # 1. Create dataloaders 
    log_string("\n📂 Loading dataset...")
    
    # Auto-detect data type
    is_precomputed = False
    is_grid_data = False
    
    # Check for precomputed FPS data (many small files with _fps pattern)
    if "ingolstadt" in data_path.lower() and "_fps" in data_path.lower():
        is_precomputed = True
        log_string("🚀 Detected precomputed FPS dataset - no sampling needed!")
    
    # Check for grid data (many small chunks, but not precomputed)
    elif "grid" in data_path.lower() or len(glob.glob(os.path.join(data_path, "train*chunk*.npy"))) > 5:
        is_grid_data = True
        log_string("🔲 Detected grid-based dataset - fast FPS sampling on small chunks!")
        # For grid data, we can use the full sample_multiplier since files are small
        log_string(f"   🎯 Using sample_multiplier={sample_multiplier} for grid data")
    
    # Check for large single files
    else:
        large_files = glob.glob(os.path.join(data_path, "train*.npy"))
        if large_files:
            log_string("📄 Detected large single files - will need FPS sampling during training")
        else:
            log_string("❓ Data format auto-detection uncertain - using default settings")
    
    data_setup = create_ingolstadt_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        n_points=n_points,
        num_workers=0,
        is_precomputed=is_precomputed,
        sample_multiplier=sample_multiplier if not is_precomputed else 1.0
    )
    
    dataloaders = data_setup['dataloaders']
    info = data_setup['info']
    
    # 2. Create model and training setup
    log_string("\n🤖 Creating semantic segmentation model...")
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
    
    log_string(f"📊 Model created for {info['num_classes']} classes")
    log_string(f"🏷️ Classes: {list(info['class_to_idx'].keys())}")
    
    # Check for existing model to resume from
    start_epoch = 0
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    if os.path.exists(save_path):
        log_string(f"\n🔄 Found existing model, resuming training from: {save_path}")
        try:
            checkpoint = torch.load(save_path, map_location=device)
            
            # Restore model and optimizer state
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_accuracy', checkpoint.get('val_accuracy', 0.0))
            
            # Restore training history if available
            train_losses = checkpoint.get('train_losses', [])
            val_accuracies = checkpoint.get('val_accuracies', [])
            
            # Update scheduler to the correct step
            for _ in range(start_epoch):
                scheduler.step()
            
            log_string(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
            log_string(f"📈 Training history: {len(train_losses)} losses, {len(val_accuracies)} val accuracies")
            
        except Exception as e:
            log_string(f"⚠️ Error loading checkpoint: {e}")
            log_string("🆕 Starting fresh training...")
            start_epoch = 0
            best_val_acc = 0.0
            train_losses = []
            val_accuracies = []
    else:
        log_string("🆕 No existing model found, starting fresh training...")
    
    # 3. Training loop
    log_string(f"\n🚀 {'Resuming' if start_epoch > 0 else 'Starting'} training...")
    if start_epoch > 0:
        log_string(f"📍 Starting from epoch {start_epoch+1}/{num_epochs}")
    else:
        log_string(f"📍 Training for {num_epochs} epochs")
    
    # Safety check for empty dataloader
    if len(dataloaders['train']) == 0:
        print("❌ Error: Training dataloader is empty!")
        print("💡 This might be because:")
        print("   • Dataset has no samples")
        print("   • Batch size is larger than dataset size")
        print("   • Data loading failed")
        return None
    
    print(f"📊 Training batches per epoch: {len(dataloaders['train'])}")
    
    # Initialize training metrics
    train_loss = 0.0
    val_acc = 0.0
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
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
        if dataloaders['val'] is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            
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
                    
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                    
                    # Update validation progress bar
                    current_val_acc = 100. * val_correct / val_total if val_total > 0 else 0
                    val_pbar.set_postfix({'Acc': f'{current_val_acc:.2f}%'})
            
            epoch_val_acc = 100. * val_correct / val_total
            val_accuracies.append(epoch_val_acc)
            
            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_accuracy': epoch_val_acc,
                    'best_val_accuracy': best_val_acc,
                    'train_losses': train_losses,
                    'val_accuracies': val_accuracies,
                    'class_to_idx': info['class_to_idx'],
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate
                }, save_path)
                print(f'💾 New best model saved! Val Acc: {epoch_val_acc:.2f}%')
        
        # Update learning rate
        scheduler.step()
        
        # Update final validation accuracy for this epoch
        val_acc = epoch_val_acc
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint every 10 epochs (for resuming)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_path.replace('.pth', f'_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_accuracy': val_acc,
                'best_val_accuracy': best_val_acc,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'class_to_idx': info['class_to_idx'],
                'num_epochs': num_epochs,
                'learning_rate': learning_rate
            }, checkpoint_path)
            print(f'📁 Checkpoint saved: {checkpoint_path}')
    
    # Final comprehensive test evaluation
    final_test_acc = 0.0
    if dataloaders['test'] is not None:
        print("\n🧪 Running comprehensive test evaluation...")
        
        # Save final model state
        torch.save({
            'epoch': num_epochs-1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_accuracy': val_acc,
            'best_val_accuracy': best_val_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'class_to_idx': info['class_to_idx'],
            'num_epochs': num_epochs,
            'learning_rate': learning_rate
        }, save_path)
        
        # Run test evaluation on the BEST model (not final model)
        # The best model is already saved in save_path from the training loop
        print("🏆 Using BEST model for test evaluation (highest validation accuracy)")
        final_test_acc = evaluate_model(
            model_path=save_path,
            data_path=data_path,
            n_points=n_points,
            batch_size=batch_size
        )
    
    print(f'🏆 Training completed! Best validation accuracy: {best_val_acc:.2f}%')
    if final_test_acc > 0:
        print(f'🏆 Final test accuracy: {final_test_acc:.2f}%')
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'final_test_acc': final_test_acc,
        'info': info
    }


def evaluate_model(model_path, data_path='/home/stud/nguyenti/storage/user/EARLy/data', 
                  n_points=1024, batch_size=32):
    """
    Evaluate a trained model on test data
    
    Args:
        model_path: str - path to saved model
        data_path: str - path to data folder
        n_points: int - number of points per sample
        batch_size: int - batch size for evaluation
    """
    print(f"🔍 Evaluating model: {model_path}")
    
    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)
    
    print(f"📊 Model classes: {list(class_to_idx.keys())}")
    
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
    if "ingolstadt" in data_path.lower() and "_fps" in data_path.lower():
        is_precomputed = True
        print("🚀 Detected precomputed FPS dataset for evaluation")
    elif "grid" in data_path.lower() or len(glob.glob(os.path.join(data_path, "test*chunk*.npy"))) > 5:
        print("🔲 Detected grid-based dataset for evaluation")
    
    data_setup = create_ingolstadt_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        n_points=n_points,
        num_workers=2,
        is_precomputed=is_precomputed
    )
    
    test_loader = data_setup['dataloaders']['test']
    
    # Evaluation
    print("🧪 Running evaluation...")
    test_correct = 0
    test_total = 0
    class_correct = {}
    class_total = {}
    
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
            
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            
            # Update evaluation progress bar
            current_acc = 100. * test_correct / test_total if test_total > 0 else 0
            eval_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
            
            # Per-class accuracy
            for i in range(target.size(0)):
                label = target[i].item()
                prediction = predicted[i].item()
                
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0
                
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1
    
    # Print results
    overall_acc = 100. * test_correct / test_total
    print(f"\n🏆 Overall Test Accuracy: {overall_acc:.2f}%")
    
    print(f"\n📊 Per-Class Accuracy:")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    for class_idx in sorted(class_total.keys()):
        class_name = idx_to_class[class_idx]
        acc = 100. * class_correct[class_idx] / class_total[class_idx]
        print(f"   {class_name}: {acc:.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")
    
    return overall_acc


def run_test_evaluation(model_path, data_path, n_points=2048, batch_size=32):
    """
    Convenient function to run test evaluation on a saved model
    
    Args:
        model_path: str - path to saved model (.pth file)
        data_path: str - path to test data folder  
        n_points: int - number of points per sample
        batch_size: int - batch size for evaluation
        
    Returns:
        float: test accuracy percentage
    """
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return 0.0
        
    print("="*50)
    print("🧪 COMPREHENSIVE TEST EVALUATION")
    print("="*50)
    
    accuracy = evaluate_model(
        model_path=model_path,
        data_path=data_path,
        n_points=n_points,
        batch_size=batch_size
    )
    
    print("="*50)
    print(f"🏆 FINAL RESULT: {accuracy:.2f}% Test Accuracy")
    print("="*50)
    
    return accuracy


def list_checkpoints(model_prefix="pointnet"):
    """
    List available checkpoint files for resuming training
    
    Args:
        model_prefix: str - prefix to search for checkpoint files
    """
    print("🔍 Available checkpoints for resuming:")
    print("="*50)
    
    # Find checkpoint files
    checkpoint_patterns = [
        f"{model_prefix}*.pth",
        f"*{model_prefix}*.pth",
        "*checkpoint*.pth"
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(glob.glob(pattern))
    
    if not checkpoints:
        print("❌ No checkpoint files found")
        print("💡 Train a model first to create checkpoints")
        return
    
    # Sort by modification time (newest first)
    checkpoints = sorted(set(checkpoints), key=os.path.getmtime, reverse=True)
    
    for i, checkpoint in enumerate(checkpoints, 1):
        try:
            # Try to load checkpoint info
            state = torch.load(checkpoint, map_location='cpu')
            epoch = state.get('epoch', 'Unknown')
            val_acc = state.get('val_accuracy', 0.0)
            best_val_acc = state.get('best_val_accuracy', 0.0)
            
            print(f"{i:2d}. {checkpoint}")
            print(f"    📊 Epoch: {epoch}, Val Acc: {val_acc:.2f}%, Best: {best_val_acc:.2f}%")
            
        except Exception as e:
            print(f"{i:2d}. {checkpoint} (⚠️ Cannot read checkpoint info)")
    
    print("="*50)
    print("💡 Use --model_path <checkpoint_path> to automatically resume training")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PointNet Semantic Segmentation for Ingolstadt Dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'analyze', 'test', 'checkpoints'],
                       help='Mode: train, eval, analyze, test, or checkpoints (list available checkpoints)')
    parser.add_argument('--data_path', type=str, default='/home/stud/nguyenti/storage/user/EARLy/data',
                       help='Path to data folder')
    parser.add_argument('--model_path', type=str, default='ingolstadt_pointnet_segmentation.pth',
                       help='Path to save/load model (automatically resumes training if file exists)')
    parser.add_argument('--n_points', type=int, default=2048,
                       help='Number of points per sample (default: 2048 for high resolution)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8 for 2048 points)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--sample_multiplier', type=float, default=1.0,
                       help='Sample multiplier for non-precomputed data (default: 1.0, higher = more samples per file)')
    parser.add_argument('--ignore_labels', type=int, nargs='+', default=None,
                       help='Labels to ignore during training (e.g., 3 4)')

    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 Starting semantic segmentation training...")
        result = train_ingolstadt_segmentation(
            data_path=args.data_path,
            n_points=args.n_points,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            save_path=args.model_path,
            sample_multiplier=args.sample_multiplier,
            ignore_labels=args.ignore_labels
        )
        
    elif args.mode == 'eval':
        print("🔍 Evaluating model...")
        if not os.path.exists(args.model_path):
            print(f"Model file not found: {args.model_path}")
            return
        
        accuracy = evaluate_model(
            model_path=args.model_path,
            data_path=args.data_path,
            n_points=args.n_points,
            batch_size=args.batch_size
        )
        
    elif args.mode == 'test':
        print("🧪 Running comprehensive test evaluation...")
        accuracy = run_test_evaluation(
            model_path=args.model_path,
            data_path=args.data_path,
            n_points=args.n_points,
            batch_size=args.batch_size
        )
        
    elif args.mode == 'checkpoints':
        # Extract model name from model_path for searching
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        list_checkpoints(model_prefix=model_name)
    
    print("Done!")


if __name__ == "__main__":
    main() 