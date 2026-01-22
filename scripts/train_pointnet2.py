#!/usr/bin/env python3
"""
PointNet2++ training script entry point
Command-line interface for training and evaluating PointNet2++ models
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.training.pointnet2_trainer import train_pointnet2_segmentation, evaluate_model
from src.utils.logging import setup_logger, get_log_string_function


def run_test_evaluation(model_path, data_path, n_points=2048, batch_size=32, ignore_labels=None, log_func=None):
    """
    Convenient function to run test evaluation on a saved PointNet2++ model
    
    Args:
        model_path: str - path to saved model (.pth file)
        data_path: str - path to test data folder  
        n_points: int - number of points per sample
        batch_size: int - batch size for evaluation
        ignore_labels: list - labels to ignore during evaluation
        log_func: function - logging function
        
    Returns:
        float: test accuracy percentage
    """
    if log_func is None:
        logger = setup_logger(name="PointNet2++")
        log_func = get_log_string_function(logger)
    
    if not os.path.exists(model_path):
        log_string(f"❌ Model file not found: {model_path}")
        return 0.0
        
    log_string("="*50)
    log_string("🧪 COMPREHENSIVE PointNet2++ TEST EVALUATION")
    log_string("="*50)
    
    accuracy = evaluate_model(
        model_path=model_path,
        data_path=data_path,
        n_points=n_points,
        batch_size=batch_size,
        ignore_labels=ignore_labels,
        log_func=log_func
    )
    
    log_string("="*50)
    log_string(f"🏆 FINAL RESULT: {accuracy:.2f}% Test Accuracy")
    log_string("="*50)
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='PointNet2++ Training and Evaluation')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'test'],
                       help='Mode: train, eval, or test')
    parser.add_argument('--data_path', type=str, 
                       default='/home/stud/nguyenti/storage/user/EARLy/data',
                       help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='pointnet2_segmentation.pth',
                       help='Path to save/load model')
    parser.add_argument('--n_points', type=int, default=2048,
                       help='Number of points per sample')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training/evaluation')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--sample_multiplier', type=float, default=1.0,
                       help='Multiplier for samples')
    parser.add_argument('--ignore_labels', type=int, nargs='+', default=None,
                       help='Labels to ignore during training')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(name="PointNet2++")
    log_string = get_log_string_function(logger)
    
    if args.mode == 'train':
        log_string("🚀 Starting PointNet2++ semantic segmentation training...")
        result = train_pointnet2_segmentation(
            data_path=args.data_path,
            n_points=args.n_points,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            save_path=args.model_path,
            sample_multiplier=args.sample_multiplier,
            ignore_labels=args.ignore_labels,
            log_func=log_string
        )
        
    elif args.mode == 'eval':
        log_string("🔍 Evaluating PointNet2++ model...")
        if not os.path.exists(args.model_path):
            log_string(f"Model file not found: {args.model_path}")
            return
        
        accuracy = evaluate_model(
            model_path=args.model_path,
            data_path=args.data_path,
            n_points=args.n_points,
            batch_size=args.batch_size,
            ignore_labels=args.ignore_labels,
            log_func=log_string
        )
        
    elif args.mode == 'test':
        log_string("🧪 Running comprehensive PointNet2++ test evaluation...")
        accuracy = run_test_evaluation(
            model_path=args.model_path,
            data_path=args.data_path,
            n_points=args.n_points,
            batch_size=args.batch_size,
            ignore_labels=args.ignore_labels,
            log_func=log_string
        )
    
    log_string("Done!")


if __name__ == "__main__":
    main()




