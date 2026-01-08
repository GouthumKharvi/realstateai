"""
Logger Module - Centralized Logging Configuration
=================================================

Provides production-ready logging with:
- Console and file handlers
- Rotating file logs (auto-cleanup)
- Different formats for console vs file
- Environment-based log levels
- Automatic log directory creation

Usage:
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("Failed to load data", exc_info=True)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

LOG_DIR = "logs"
LOG_FILE = f"{LOG_DIR}/procurement_ai_{datetime.now().strftime('%Y%m%d')}.log"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
BACKUP_COUNT = 5  # Keep 5 backup files

# Log format
DETAILED_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(funcName)s:%(lineno)d | %(message)s"
)
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# LOGGER SETUP
# ============================================================

def setup_logger(
    name: str = "procurement_ai",
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Enable console logging
        file_output: Enable file logging
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__, log_level="DEBUG")
        >>> logger.info("System initialized")
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create log directory if needed
    if file_output:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # ========== CONSOLE HANDLER ==========
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            CONSOLE_FORMAT,
            datefmt=DATE_FORMAT
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # ========== FILE HANDLER (ROTATING) ==========
    if file_output:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(
            DETAILED_FORMAT,
            datefmt=DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get or create a logger instance (simplified interface).
    
    Args:
        name: Logger name (use __name__ from calling module)
        
    Returns:
        Logger instance
        
    Example:
        >>> from utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing stage 4")
    """
    if name is None:
        name = "procurement_ai"
    
    # Check if logger already configured
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


# ============================================================
# LOG LEVEL HELPERS
# ============================================================

def set_log_level(logger: logging.Logger, level: str):
    """
    Change log level dynamically.
    
    Args:
        logger: Logger instance
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Example:
        >>> logger = get_logger(__name__)
        >>> set_log_level(logger, "DEBUG")
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)


def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exception: Exception object
        context: Additional context message
        
    Example:
        >>> try:
        >>>     risky_operation()
        >>> except Exception as e:
        >>>     log_exception(logger, e, "Failed to process vendor data")
    """
    if context:
        logger.error(f"{context}: {str(exception)}", exc_info=True)
    else:
        logger.error(f"{type(exception).__name__}: {str(exception)}", exc_info=True)


# ============================================================
# STRUCTURED LOGGING HELPERS
# ============================================================

def log_stage_start(logger: logging.Logger, stage_name: str, input_count: int = None):
    """Log the start of a processing stage."""
    msg = f"{'='*60}\n🚀 STARTING: {stage_name}"
    if input_count:
        msg += f" | Records: {input_count}"
    logger.info(msg)


def log_stage_end(logger: logging.Logger, stage_name: str, output_count: int = None, 
                  duration: float = None):
    """Log the completion of a processing stage."""
    msg = f"✅ COMPLETED: {stage_name}"
    if output_count is not None:
        msg += f" | Output: {output_count} records"
    if duration:
        msg += f" | Duration: {duration:.2f}s"
    msg += f"\n{'='*60}"
    logger.info(msg)


def log_metric(logger: logging.Logger, metric_name: str, value: float, unit: str = ""):
    """Log a metric value."""
    logger.info(f"📊 METRIC: {metric_name} = {value:.2f}{unit}")


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test the logger
    test_logger = get_logger("test_module")
    
    log_stage_start(test_logger, "Stage 1: Vendor Development", input_count=150)
    
    test_logger.debug("This is a debug message (only in file)")
    test_logger.info("Processing vendor records...")
    test_logger.warning("Vendor XYZ has incomplete data")
    
    log_metric(test_logger, "Approval Rate", 87.5, "%")
    
    try:
        raise ValueError("Sample error for testing")
    except Exception as e:
        log_exception(test_logger, e, "Error during vendor validation")
    
    log_stage_end(test_logger, "Stage 1: Vendor Development", 
                  output_count=142, duration=2.5)
    
    print(f"\n✅ Logger test complete! Check: {LOG_FILE}")
