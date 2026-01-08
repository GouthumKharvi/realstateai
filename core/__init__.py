"""
Core processing modules for Contract & Purchase AI
Includes input layer, OCR, data processing, and AI engines
"""

# Input Layer - File loading and validation
from .input_layer import (
    load_csv,
    load_excel,
    load_pdf,
    validate_input,
    validate_file_upload,
    VALIDATION_PATTERNS
)

# OCR Layer - Text extraction from PDFs and images
from .ocr_layer import (
    extract_text_from_pdf,
    extract_text_from_image,
    preprocess_image
)

# Data Processing - NLP and feature extraction
from .data_processing import (
    clean_text,
    normalize_data,
    normalize_currency,
    normalize_date,
    segment_clauses,
    identify_clause_type,
    extract_entities,
    extract_features,
    structure_data,
    structure_contract_data
)

__all__ = [
    # Input Layer
    'load_csv',
    'load_excel',
    'load_pdf',
    'validate_input',
    'validate_file_upload',
    'VALIDATION_PATTERNS',
    
    # OCR Layer
    'extract_text_from_pdf',
    'extract_text_from_image',
    'preprocess_image',
    
    # Data Processing
    'clean_text',
    'normalize_data',
    'normalize_currency',
    'normalize_date',
    'segment_clauses',
    'identify_clause_type',
    'extract_entities',
    'extract_features',
    'structure_data',
    'structure_contract_data'
]

__version__ = '1.0.0'
