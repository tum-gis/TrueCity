"""
Logging utilities for training and evaluation
Provides unified logging setup for both PointNet and PointNet2++ training
"""

import logging
import datetime
from pathlib import Path


def setup_logger(name="PointNet", log_dir="./logs/"):
    """
    Setup global logger for the entire script
    
    Args:
        name: str - logger name (e.g., "PointNet" or "PointNet2++")
        log_dir: str - directory to save log files
    
    Returns:
        logger: configured logger instance
    """
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_dir / f'{name.lower().replace("++", "2")}_training_{timestr}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_log_string_function(logger):
    """
    Get a logging function that logs to both file and console
    
    Args:
        logger: logging.Logger instance
    
    Returns:
        log_string: function that takes a message string and logs it
    """
    def log_string(msg):
        """Global logging function that logs to both file and console"""
        logger.info(msg)
    
    return log_string




