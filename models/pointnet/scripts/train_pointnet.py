#!/usr/bin/env python3
"""
PointNet training script entry point
Command-line interface for training and evaluating PointNet models
"""

import os
import sys
import argparse
import glob
import torch

# Add shared and local src to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_root, 'shared'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.pointnet_trainer import train_ingolstadt_segmentation, evaluate_model
from shared.data.dataloader import analyze_ingolstadt_dataset
from shared.utils.logging import setup_logger, log_string


def run_test_evaluation(model_path, data_path, n_points=2048, batch_size=32, log_func=None):
    """
    Convenient function to run test evaluation on a saved model
    
    Args:
        model_path: str - path to saved model (.pth file)
        data_path: str - path to test data folder  
        n_points: int - number of points per sample
        batch_size: int - batch size for evaluation
        log_func: function - logging function
        
    Returns:
        float: test accuracy percentage
    """
    if log_func is None:
        logger = setup_logger(name="PointNet")
        log_func = lambda msg: log_string(logger, msg)
    
    if not os.path.exists(model_path):
        log_func(f"❌ Model file not found: {model_path}")
        return 0.0
        
    log_func("="*50)
    log_func("🧪 COMPREHENSIVE TEST EVALUATION")
    log_func("="*50)
    
    accuracy = evaluate_model(
        model_path=model_path,
        data_path=data_path,
        n_points=n_points,
        batch_size=batch_size
    )
    
    log_func("="*50)
    log_func(f"🏆 FINAL RESULT: {accuracy:.2f}% Test Accuracy")
    log_func("="*50)
    
    return accuracy


def list_checkpoints(model_prefix="pointnet", log_func=None):
    """
    List available checkpoint files for resuming training
    
    Args:
        model_prefix: str - prefix to search for checkpoint files
        log_func: function - logging function
    """
    if log_func is None:
        logger = setup_logger(name="PointNet")
        log_func = lambda msg: log_string(logger, msg)
    
    log_func("🔍 Available checkpoints for resuming:")
    log_func("="*50)
    
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
        log_func("❌ No checkpoint files found")
        log_func("💡 Train a model first to create checkpoints")
        return
    
    # Sort by modification time (newest first)
    checkpoints = sorted(set(checkpoints), key=os.path.getmtime, reverse=True)
    
    for i, checkpoint in enumerate(checkpoints, 1):
        try:
            # Try to load checkpoint info
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            epoch = state.get('epoch', 'Unknown')
            val_acc = state.get('val_accuracy', 0.0)
            best_val_acc = state.get('best_val_accuracy', 0.0)
            
            log_func(f"{i:2d}. {checkpoint}")
            log_func(f"    📊 Epoch: {epoch}, Val Acc: {val_acc:.2f}%, Best: {best_val_acc:.2f}%")
            
        except Exception as e:
            log_func(f"{i:2d}. {checkpoint} (⚠️ Cannot read checkpoint info)")
    
    log_func("="*50)
    log_func("💡 Use --model_path <checkpoint_path> to automatically resume training")


def main():
    """Main execution function"""
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
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to analyze (for large datasets)')

    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(name="PointNet")
    log_func = lambda msg: log_string(logger, msg)
    
    if args.mode == 'analyze':
        log_func("📊 Analyzing dataset...")
        stats = analyze_ingolstadt_dataset(data_path=args.data_path, max_files=args.max_files)
        

    elif args.mode == 'train':
        log_func("🚀 Starting semantic segmentation training...")
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
        log_func("🔍 Evaluating model...")
        if not os.path.exists(args.model_path):
            log_func(f"Model file not found: {args.model_path}")
            return
        
        accuracy = evaluate_model(
            model_path=args.model_path,
            data_path=args.data_path,
            n_points=args.n_points,
            batch_size=args.batch_size
        )
        
    elif args.mode == 'test':
        log_func("🧪 Running comprehensive test evaluation...")
        accuracy = run_test_evaluation(
            model_path=args.model_path,
            data_path=args.data_path,
            n_points=args.n_points,
            batch_size=args.batch_size,
            log_func=log_func
        )
        
    elif args.mode == 'checkpoints':
        # Extract model name from model_path for searching
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        list_checkpoints(model_prefix=model_name, log_func=log_func)
    
    log_func("Done!")


if __name__ == "__main__":
    main()

