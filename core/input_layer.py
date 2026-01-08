"""
Input Layer - File loading and validation
Handles CSV, PDF, Excel file uploads with validation
"""

import pandas as pd
from pypdf import PdfReader
import os
from typing import Tuple, List, Dict, Any
import re

from pypdf import PdfReader

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# Loaded the CSV file and make sure it actually exists and has data
def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with CSV data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    # itwill Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # it will Check if file is empty
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"CSV file is empty: {file_path}")
    
    try:
        # Try UTF-8 encoding first
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:

        # Fallback to latin1 encoding
        df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # it will Check if DataFrame is empty
    if df.empty:
        raise ValueError(f"CSV file contains no data: {file_path}")
    
    return df


# it Read a PDF file and pull out all the text from every page
def load_pdf(file_path: str) -> str:
    """
    Extract text from PDF file
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If PDF cannot be read
    """
    #it  Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    # Check if file is empty
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"PDF file is empty: {file_path}")
    
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        
        # Check if text was extracted
        if not text.strip():
            raise ValueError(f"No text could be extracted from PDF: {file_path}")
        
        return text.strip()
    
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")


# Load an Excel sheet (first sheet by default) and make sure it has rows
def load_excel(file_path: str, sheet_name: str = 0) -> pd.DataFrame:
    """
    Load Excel file into pandas DataFrame
    
    Args:
        file_path: Path to Excel file (.xlsx or .xls)
        sheet_name: Sheet name or index (default: first sheet)
        
    Returns:
        DataFrame with Excel data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    # Check if file is empty
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Excel file is empty: {file_path}")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError(f"Excel file contains no data: {file_path}")
    
    return df


# Run all basic checks on a DataFrame (columns, types, nulls, regex, duplicates)
def validate_input(
    data: pd.DataFrame,
    expected_columns: List[str] = None,
    data_types: Dict[str, type] = None,
    required_non_null: List[str] = None,
    validation_rules: Dict[str, Any] = None
) -> Tuple[bool, List[str]]:
    """
    Validate input DataFrame for required columns, data types, and business rules
    
    Args:
        data: DataFrame to validate
        expected_columns: List of required column names
        data_types: Dict mapping column names to expected data types
        required_non_null: List of columns that cannot have null values
        validation_rules: Dict of custom validation rules
        
    Returns:
        Tuple of (is_valid: bool, errors: List[str])
        
    Example:
        is_valid, errors = validate_input(
            data=df,
            expected_columns=['vendor_id', 'vendor_name', 'gstin'],
            data_types={'vendor_id': str, 'gstin': str},
            required_non_null=['vendor_name', 'gstin'],
            validation_rules={
                'gstin': r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$',
                'email': r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
            }
        )
    """
    errors = []
    
    # Check if DataFrame is empty
    if data.empty:
        errors.append("DataFrame is empty - no data to validate")
        return False, errors
    
    # Validate expected columns
    if expected_columns:
        missing_columns = set(expected_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Validate data types
    if data_types:
        for col, expected_type in data_types.items():
            if col in data.columns:
                # Check if column can be converted to expected type
                try:
                    if expected_type == str:
                        data[col] = data[col].astype(str)
                    elif expected_type in [int, float]:
                        pd.to_numeric(data[col], errors='raise')
                except:
                    errors.append(f"Column '{col}' has invalid data type (expected {expected_type.__name__})")
    
    # Validate non-null requirements
    if required_non_null:
        for col in required_non_null:
            if col in data.columns:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}' has {null_count} null/empty values (required field)")
    
    # Validate custom rules (regex patterns)
    if validation_rules:
        for col, pattern in validation_rules.items():
            if col in data.columns:
                # Skip null values
                non_null_data = data[col].dropna()
                
                if isinstance(pattern, str):  # Regex pattern
                    invalid_rows = []
                    for idx, value in non_null_data.items():
                        if not re.match(pattern, str(value)):
                            invalid_rows.append(idx + 2)  # +2 for Excel row (header + 0-index)
                    
                    if invalid_rows:
                        errors.append(
                            f"Column '{col}' has invalid format in rows: {invalid_rows[:5]}"
                            + (f" and {len(invalid_rows) - 5} more" if len(invalid_rows) > 5 else "")
                        )
    
    # Check for duplicate primary keys (if vendor_id or similar exists)
    primary_key_candidates = ['vendor_id', 'contract_id', 'invoice_id', 'bid_id', 'rfq_id']
    for pk in primary_key_candidates:
        if pk in data.columns:
            duplicates = data[pk].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate values in '{pk}' (should be unique)")
    
    # Return validation result
    is_valid = len(errors) == 0
    return is_valid, errors


# Quick checks on uploaded file (type + size) before trying to read it
def validate_file_upload(file) -> Tuple[bool, str]:
    """
    Validate uploaded file (for Streamlit file uploader)
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if file is None:
        return False, "No file uploaded"
    
    # Check file extension
    allowed_extensions = ['.csv', '.xlsx', '.xls', '.pdf']
    file_ext = os.path.splitext(file.name)[1].lower()
    
    if file_ext not in allowed_extensions:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (max 50MB)
    max_size_mb = 50
    file_size_mb = file.size / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        return False, f"File too large ({file_size_mb:.2f}MB). Maximum: {max_size_mb}MB"
    
    return True, ""


# Common regex patterns you can reuse when calling validate_input()
VALIDATION_PATTERNS = {
    'gstin': r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$',
    'pan': r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$',
    'email': r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$',
    'phone': r'^[6-9]\d{9}$',  # Indian phone number
    'ifsc': r'^[A-Z]{4}0[A-Z0-9]{6}$',
    'pincode': r'^\d{6}$'
}


# Simple local test runner so we  can quickly check if this file works standalone
if __name__ == "__main__":
    # Test the functions
    print("Testing input_layer.py...")
    
    # Test CSV loading
    try:
        df = load_csv("mockdata/vendors.csv")
        print(f"✅ CSV loaded: {len(df)} rows")
    except Exception as e:
        print(f"❌ CSV load failed: {e}")
    
    # Test validation
    try:
        is_valid, errors = validate_input(
            data=df,
            expected_columns=['vendor_id', 'vendor_name'],
            required_non_null=['vendor_name']
        )
        if is_valid:
            print("✅ Validation passed")
        else:
            print(f"❌ Validation failed: {errors}")
    except Exception as e:
        print(f"❌ Validation error: {e}")

