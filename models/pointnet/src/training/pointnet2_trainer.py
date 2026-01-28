#!/usr/bin/env python3
"""
PointNet2++ Training Script
Adapted from the PointNet2.ipynb notebook
"""

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
from tqdm import tqdm
import argparse
import datetime
from pathlib import Path
import logging

# Add project root (TrueCity) and pointnet root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
pointnet_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, project_root)      # for `shared.*`
sys.path.insert(0, pointnet_root)     # for `src.*`

from shared.data.dataloader import create_ingolstadt_dataloaders
from shared.utils.logging import setup_logger, log_string as log_string_helper
from src.models.pointnet2 import PointNet2SemanticSegmentation
from src.models.pointnet import SemanticSegmentationLoss


class PointNet2Loss(nn.Module):
    def __init__(self):
        super(PointNet2Loss, self).__init__()
    
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def train_pointnet2_segmentation(data_path='/home/stud/nguyenti/storage/user/EARLy/data', 
                                 n_points=2048, batch_size=32, num_epochs=50, 
                                 learning_rate=0.0001, save_path='pointnet2_segmentation.pth',
                                 sample_multiplier=1.0, ignore_labels=None):
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
    """
    print("🚀 Starting PointNet2++ semantic segmentation training...")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    # Create output directory
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('pointnet2_sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging using shared utility
    logger = setup_logger("PointNet2++", log_dir=str(experiment_dir / 'logs'), log_file=f'pointnet2_training_{timestr}.txt')
    
    def log_string(msg):
        log_string_helper(logger, msg)
    
    log_string("🏗️ Setting up PointNet2++ SEMANTIC SEGMENTATION training...")
    log_string(f"📁 Data path: {data_path}")
    log_string(f"🎯 Points per sample: {n_points}")
    log_string(f"📦 Batch size: {batch_size}")
    log_string(f"🔄 Epochs: {num_epochs}")
    log_string(f"🎨 Task: Semantic Segmentation (per-point classification)")
    if ignore_labels:
        log_string(f"🚫 Ignoring labels: {ignore_labels}")
    
    # Load dataset
    log_string("📂 Loading dataset...")
    
    data_setup = create_ingolstadt_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        n_points=n_points,
        num_workers=4,
        sample_multiplier=sample_multiplier,
        is_precomputed=True
    )
    
    dataloaders = data_setup['dataloaders']
    # Get dataset info
    info = data_setup['info']
    num_classes = info['num_classes']
    
    log_string(f"🤖 Creating PointNet2++ semantic segmentation model...")
    log_string(f"📊 Model created for {num_classes} classes")
    log_string(f"🏷️ Classes: {list(info['class_to_idx'].keys())}")
    
    # Create model
    model = PointNet2SemanticSegmentation(num_classes).to(device)
    criterion = SemanticSegmentationLoss().to(device)
    model.apply(inplace_relu)
    
    # Try to load existing model
    try:
        checkpoint = torch.load(save_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('🔄 Use pretrain model')
    except:
        log_string('🆕 No existing model found, starting fresh training...')
        start_epoch = 0
        model = model.apply(weights_init)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    
    # Training parameters
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = 10
    step_size = 10
    lr_decay = 0.7
    
    # Class weights (equal weights for now)
    weights = torch.ones(num_classes).to(device)
    
    log_string("🚀 Starting training...")
    log_string(f"📍 Training for {num_epochs} epochs")
    log_string(f"📊 Training batches per epoch: {len(dataloaders['train'])}")
    
    best_iou = 0
    train_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        log_string(f'**** Epoch {epoch + 1} ({epoch + 1}/{num_epochs}) ****')
        
        # Learning rate scheduling
        lr = max(learning_rate * (lr_decay ** (epoch // step_size)), LEARNING_RATE_CLIP)
        log_string(f'Learning rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # BatchNorm momentum adjustment
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_string(f'BN momentum updated to: {momentum}')
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
            loss.backward()
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
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
        train_loss = loss_sum / num_batches
        train_acc = total_correct / float(total_seen)
        train_losses.append(train_loss)
        
        log_string(f'Training avg class IoU: {mIoU:.6f}')
        log_string(f'Training mean loss: {train_loss:.6f}')
        log_string(f'Training accuracy: {train_acc:.6f}')
        log_string(f'Training point avg class acc: {np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6)):.6f}')
        
        # Save model every 5 epochs
        if epoch % 5 == 0:
            log_string('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string(f'Saving at {savepath}')
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        
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
                
                log_string('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
                val_pbar = tqdm(dataloaders['val'], desc="Validation")
                
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
                    target = target.view(-1, 1)[:, 0]
                    loss = criterion(seg_pred, target, trans_feat, weights)
                    loss_sum += loss.item()
                    
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum((pred_val == batch_label))
                    total_correct += correct
                    total_seen += (batch_size * n_points)
                    
                    for l in range(num_classes):
                        total_seen_class[l] += np.sum((batch_label == l))
                        total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                        total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                
                mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
                val_loss = loss_sum / num_batches
                val_acc = total_correct / float(total_seen)
                
                log_string(f'eval mean loss: {val_loss:.6f}')
                log_string(f'eval point avg class IoU: {mIoU:.6f}')
                log_string(f'eval point accuracy: {val_acc:.6f}')
                log_string(f'eval point avg class acc: {np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6)):.6f}')
                
                # Save best model
                if mIoU >= best_iou:
                    best_iou = mIoU
                    log_string('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string(f'Saving at {savepath}')
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                    log_string('Saving model....')
                log_string(f'Best mIoU: {best_iou:.6f}')
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
    }, save_path)
    
    log_string(f"✅ Training complete! Model saved to {save_path}")
    return {
        'model': model,
        'best_iou': best_iou,
        'train_losses': train_losses
    }


def main():
    parser = argparse.ArgumentParser(description='PointNet2++ Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Training or testing mode')
    parser.add_argument('--data_path', type=str, 
                       default='/home/stud/nguyenti/storage/user/EARLy/data',
                       help='Path to data directory')
    parser.add_argument('--n_points', type=int, default=2048,
                       help='Number of points per sample')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--save_path', type=str, default='pointnet2_segmentation.pth',
                       help='Path to save the model')
    parser.add_argument('--sample_multiplier', type=float, default=1.0,
                       help='Multiplier for samples')
    parser.add_argument('--ignore_labels', type=int, nargs='+', default=None,
                       help='Labels to ignore during training')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_pointnet2_segmentation(
            data_path=args.data_path,
            n_points=args.n_points,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            save_path=args.save_path,
            sample_multiplier=args.sample_multiplier,
            ignore_labels=args.ignore_labels
        )
    else:
        print("Test mode not implemented yet")


if __name__ == "__main__":
    main() 