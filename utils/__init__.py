"""
Utils Package - Helper Functions and Utilities
==============================================

This package provides essential utilities for the AI-enabled
real estate procurement system:

- logger: Centralized logging functionality
- validators: Input validation and data checks
- constants: System-wide configuration constants
- formatters: Output formatting utilities

Usage:
    from utils import get_logger, validate_dataframe
    from utils.constants import CONFIDENCE_THRESHOLDS
"""

from .logger import get_logger, setup_logger
from .validators import (
    validate_dataframe,
    validate_numeric,
    validate_date,
    validate_email,
    validate_file_path
)
from .constants import (
    CONFIDENCE_THRESHOLDS,
    ANOMALY_THRESHOLDS,
    DATE_FORMATS,
    CURRENCY_FORMATS,
    STAGE_NAMES
)
from .formatters import (
    format_currency,
    format_percentage,
    format_date,
    format_confidence,
    format_output_dict
)

__version__ = '1.0.0'
__author__ = 'Gouthum Kharvi'

__all__ = [
    # Logger
    'get_logger',
    'setup_logger',
    
    # Validators
    'validate_dataframe',
    'validate_numeric',
    'validate_date',
    'validate_email',
    'validate_file_path',
    
    # Constants
    'CONFIDENCE_THRESHOLDS',
    'ANOMALY_THRESHOLDS',
    'DATE_FORMATS',
    'CURRENCY_FORMATS',
    'STAGE_NAMES',
    
    # Formatters
    'format_currency',
    'format_percentage',
    'format_date',
    'format_confidence',
    'format_output_dict',
]
