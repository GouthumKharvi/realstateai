"""
Validators Module - Input Validation and Data Quality Checks
============================================================

Provides validation functions for:
- DataFrames (structure, required columns, data types)
- Numeric values (range checks, positive values)
- Dates (format validation, range checks)
- Email addresses (format validation)
- File paths (existence, permissions)
- Currency values (format, range)

Usage:
    from utils.validators import validate_dataframe, validate_numeric
    
    validate_dataframe(df, required_columns=['vendor_id', 'amount'])
    validate_numeric(price, min_value=0, field_name="price")
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Any
from utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# DATAFRAME VALIDATION
# ============================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    min_rows: int = 1,
    allow_empty: bool = False,
    check_nulls: bool = True,
    df_name: str = "DataFrame"
) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must exist
        min_rows: Minimum number of rows required
        allow_empty: Allow empty DataFrame
        check_nulls: Warn about null values
        df_name: Name for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> validate_dataframe(
        ...     vendor_df,
        ...     required_columns=['vendor_id', 'name', 'amount'],
        ...     min_rows=1
        ... )
    """
    # Check if DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{df_name} must be a pandas DataFrame, got {type(df)}")
    
    # Check if empty
    if df.empty and not allow_empty:
        raise ValueError(f"{df_name} is empty (0 rows)")
    
    # Check minimum rows
    if len(df) < min_rows:
        raise ValueError(
            f"{df_name} has {len(df)} rows, minimum required: {min_rows}"
        )
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"{df_name} missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )
    
    # Check for null values (warning only)
    if check_nulls and df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        logger.warning(
            f"{df_name} contains null values:\n{null_cols.to_dict()}"
        )
    
    logger.debug(f"✅ {df_name} validated: {len(df)} rows, {len(df.columns)} columns")
    return True


def validate_column_type(
    df: pd.DataFrame,
    column: str,
    expected_type: type,
    df_name: str = "DataFrame"
) -> bool:
    """
    Validate column data type.
    
    Args:
        df: DataFrame
        column: Column name
        expected_type: Expected Python type (int, float, str, datetime)
        df_name: Name for error messages
        
    Returns:
        True if valid
        
    Example:
        >>> validate_column_type(df, 'amount', float)
    """
    if column not in df.columns:
        raise ValueError(f"{df_name} missing column: {column}")
    
    # Map Python types to pandas dtypes
    type_mapping = {
        int: ['int64', 'int32', 'Int64'],
        float: ['float64', 'float32'],
        str: ['object', 'string'],
        datetime: ['datetime64[ns]']
    }
    
    expected_dtypes = type_mapping.get(expected_type, [])
    actual_dtype = str(df[column].dtype)
    
    if not any(dt in actual_dtype for dt in expected_dtypes):
        raise ValueError(
            f"{df_name}['{column}'] has type {actual_dtype}, "
            f"expected {expected_type.__name__}"
        )
    
    return True


# ============================================================
# NUMERIC VALIDATION
# ============================================================

def validate_numeric(
    value: Union[int, float],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_negative: bool = True,
    allow_zero: bool = True,
    field_name: str = "value"
) -> bool:
    """
    Validate numeric value.
    
    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_negative: Allow negative values
        allow_zero: Allow zero
        field_name: Field name for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> validate_numeric(price, min_value=0, field_name="price")
    """
    # Check if numeric
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field_name} must be numeric, got {type(value)}")
    
    # Check for NaN
    if pd.isna(value):
        raise ValueError(f"{field_name} is NaN")
    
    # Check negative
    if not allow_negative and value < 0:
        raise ValueError(f"{field_name} cannot be negative, got {value}")
    
    # Check zero
    if not allow_zero and value == 0:
        raise ValueError(f"{field_name} cannot be zero")
    
    # Check range
    if min_value is not None and value < min_value:
        raise ValueError(
            f"{field_name} = {value} is below minimum {min_value}"
        )
    
    if max_value is not None and value > max_value:
        raise ValueError(
            f"{field_name} = {value} exceeds maximum {max_value}"
        )
    
    return True


def validate_percentage(
    value: float,
    field_name: str = "percentage"
) -> bool:
    """
    Validate percentage value (0-100).
    
    Args:
        value: Percentage value
        field_name: Field name for error messages
        
    Returns:
        True if valid
        
    Example:
        >>> validate_percentage(85.5, field_name="discount")
    """
    return validate_numeric(
        value,
        min_value=0,
        max_value=100,
        allow_negative=False,
        field_name=field_name
    )


# ============================================================
# DATE VALIDATION
# ============================================================

def validate_date(
    date_value: Union[str, datetime, pd.Timestamp],
    date_format: str = "%Y-%m-%d",
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
    field_name: str = "date"
) -> datetime:
    """
    Validate and parse date.
    
    Args:
        date_value: Date as string, datetime, or Timestamp
        date_format: Expected date format for strings
        min_date: Minimum allowed date
        max_date: Maximum allowed date
        field_name: Field name for error messages
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> validate_date("2026-01-07", min_date=datetime(2025, 1, 1))
    """
    # Convert to datetime
    if isinstance(date_value, str):
        try:
            dt = datetime.strptime(date_value, date_format)
        except ValueError as e:
            raise ValueError(
                f"{field_name} '{date_value}' does not match format '{date_format}'"
            ) from e
    elif isinstance(date_value, pd.Timestamp):
        dt = date_value.to_pydatetime()
    elif isinstance(date_value, datetime):
        dt = date_value
    else:
        raise ValueError(
            f"{field_name} must be string, datetime, or Timestamp, got {type(date_value)}"
        )
    
    # Check range
    if min_date and dt < min_date:
        raise ValueError(
            f"{field_name} {dt.date()} is before minimum {min_date.date()}"
        )
    
    if max_date and dt > max_date:
        raise ValueError(
            f"{field_name} {dt.date()} is after maximum {max_date.date()}"
        )
    
    return dt


def validate_date_range(
    start_date: datetime,
    end_date: datetime,
    allow_same_day: bool = True
) -> bool:
    """
    Validate date range (start before end).
    
    Args:
        start_date: Start date
        end_date: End date
        allow_same_day: Allow start and end on same day
        
    Returns:
        True if valid
        
    Example:
        >>> validate_date_range(project_start, project_end)
    """
    if allow_same_day:
        if end_date < start_date:
            raise ValueError(
                f"End date {end_date.date()} is before start date {start_date.date()}"
            )
    else:
        if end_date <= start_date:
            raise ValueError(
                f"End date must be after start date"
            )
    
    return True


# ============================================================
# STRING VALIDATION
# ============================================================

def validate_email(email: str, field_name: str = "email") -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address
        field_name: Field name for error messages
        
    Returns:
        True if valid
        
    Example:
        >>> validate_email("vendor@example.com")
    """
    if not isinstance(email, str):
        raise ValueError(f"{field_name} must be a string")
    
    # Simple email regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        raise ValueError(f"{field_name} '{email}' is not a valid email address")
    
    return True


def validate_phone(
    phone: str,
    country_code: str = "IN",
    field_name: str = "phone"
) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number
        country_code: Country code (IN, US, etc.)
        field_name: Field name for error messages
        
    Returns:
        True if valid
        
    Example:
        >>> validate_phone("+91 98765 43210", country_code="IN")
    """
    if not isinstance(phone, str):
        raise ValueError(f"{field_name} must be a string")
    
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)\+]', '', phone)
    
    # Country-specific patterns
    patterns = {
        'IN': r'^(91)?[6-9]\d{9}$',  # India: 10 digits starting with 6-9
        'US': r'^(1)?[2-9]\d{9}$',    # US: 10 digits
    }
    
    pattern = patterns.get(country_code, r'^\d{10,15}$')
    
    if not re.match(pattern, cleaned):
        raise ValueError(
            f"{field_name} '{phone}' is not a valid {country_code} phone number"
        )
    
    return True


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    allowed_extensions: List[str] = None,
    check_readable: bool = True,
    field_name: str = "file"
) -> Path:
    """
    Validate file path.
    
    Args:
        file_path: Path to file
        must_exist: File must exist
        allowed_extensions: List of allowed extensions (e.g., ['.csv', '.xlsx'])
        check_readable: Check if file is readable
        field_name: Field name for error messages
        
    Returns:
        Path object
        
    Example:
        >>> validate_file_path("data/vendors.csv", allowed_extensions=['.csv'])
    """
    path = Path(file_path)
    
    # Check existence
    if must_exist and not path.exists():
        raise ValueError(f"{field_name} does not exist: {file_path}")
    
    # Check extension
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValueError(
                f"{field_name} must have extension {allowed_extensions}, "
                f"got {path.suffix}"
            )
    
    # Check readable
    if check_readable and must_exist:
        if not os.access(path, os.R_OK):
            raise ValueError(f"{field_name} is not readable: {file_path}")
    
    return path


# ============================================================
# CURRENCY VALIDATION
# ============================================================

def validate_currency(
    amount: float,
    currency: str = "INR",
    min_amount: float = 0,
    max_amount: Optional[float] = None,
    field_name: str = "amount"
) -> bool:
    """
    Validate currency amount.
    
    Args:
        amount: Amount value
        currency: Currency code (INR, USD, etc.)
        min_amount: Minimum allowed amount
        max_amount: Maximum allowed amount
        field_name: Field name for error messages
        
    Returns:
        True if valid
        
    Example:
        >>> validate_currency(150000, currency="INR", min_amount=0)
    """
    validate_numeric(
        amount,
        min_value=min_amount,
        max_value=max_amount,
        allow_negative=False,
        field_name=f"{field_name} ({currency})"
    )
    
    # Validate currency code
    valid_currencies = ['INR', 'USD', 'EUR', 'GBP', 'AED']
    if currency not in valid_currencies:
        logger.warning(
            f"Currency '{currency}' not in standard list: {valid_currencies}"
        )
    
    return True


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING VALIDATORS")
    print("="*60)
    
    # Test DataFrame validation
    print("\n✅ Testing DataFrame validation...")
    test_df = pd.DataFrame({
        'vendor_id': [1, 2, 3],
        'name': ['ABC Corp', 'XYZ Ltd', 'PQR Inc'],
        'amount': [100000, 150000, 200000]
    })
    
    try:
        validate_dataframe(test_df, required_columns=['vendor_id', 'name', 'amount'])
        print("   ✅ Valid DataFrame passed")
    except ValueError as e:
        print(f"   ❌ Failed: {e}")
    
    # Test numeric validation
    print("\n✅ Testing numeric validation...")
    try:
        validate_numeric(150000, min_value=0, field_name="price")
        print("   ✅ Valid number passed")
    except ValueError as e:
        print(f"   ❌ Failed: {e}")
    
    try:
        validate_numeric(-100, min_value=0, allow_negative=False, field_name="price")
        print("   ❌ Should have failed (negative)")
    except ValueError:
        print("   ✅ Correctly rejected negative value")
    
    # Test date validation
    print("\n✅ Testing date validation...")
    try:
        dt = validate_date("2026-01-07")
        print(f"   ✅ Valid date parsed: {dt.date()}")
    except ValueError as e:
        print(f"   ❌ Failed: {e}")
    
    # Test email validation
    print("\n✅ Testing email validation...")
    try:
        validate_email("vendor@example.com")
        print("   ✅ Valid email passed")
    except ValueError as e:
        print(f"   ❌ Failed: {e}")
    
    try:
        validate_email("invalid-email")
        print("   ❌ Should have failed (invalid email)")
    except ValueError:
        print("   ✅ Correctly rejected invalid email")
    
    # Test currency validation
    print("\n✅ Testing currency validation...")
    try:
        validate_currency(150000, currency="INR", min_amount=0)
        print("   ✅ Valid currency amount passed")
    except ValueError as e:
        print(f"   ❌ Failed: {e}")
    
    print("\n" + "="*60)
    print("🎉 VALIDATOR TESTS COMPLETE")
    print("="*60)
