"""
Logging utilities for training scripts
"""
import logging
import datetime
from pathlib import Path


def setup_logger(name, log_dir='./logs/', log_file=None):
    """
    Setup a logger with file and console handlers
    
    Args:
        name: logger name
        log_dir: directory for log files
        log_file: optional log file name (auto-generated if None)
    
    Returns:
        logger: configured logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    if log_file is None:
        timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = f'{name}_training_{timestr}.txt'
    
    file_handler = logging.FileHandler(log_path / log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_string(logger, message):
    """
    Log a message using the logger and print it
    
    Args:
        logger: logger instance
        message: message to log
    """
    logger.info(message)
    print(message)

