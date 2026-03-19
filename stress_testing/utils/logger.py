"""Logging configuration"""

import logging
import sys
from pathlib import Path
from typing import Optional
def setup_logger(name: str, log_file: Optional[str] = None, 
    """setup_logger implementation"""
                 log_level: int = logging.INFO) -> logging.Logger:
"""Setup a logger with both console and optional file handlers. Args: name: Logger name log_file: Optional file path to write logs to log_level: Logging level (DEBUG, INFO, WARNING, ERROR) Returns: Configured logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    return logger
def log_section(logger: logging.Logger, title: str):
    """Log a formatted section header."""
    logger.info("\n" + "="*60)
    logger.info(f"  {title}")
    logger.info("="*60)
def log_subsection(logger: logging.Logger, title: str):
    """Log a formatted subsection header."""
    logger.info(f"\n  {title}")
    logger.info("  " + "-"*56)

