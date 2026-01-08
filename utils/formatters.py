"""
Formatters Module - Data Formatting and Output Utilities
========================================================

Provides formatting functions for:
- Currency values (INR, USD with proper symbols)
- Percentages (with precision control)
- Dates (multiple format styles)
- Confidence scores (with labels)
- Output dictionaries (clean JSON-ready format)
- Numbers (with thousand separators)

Usage:
    from utils.formatters import format_currency, format_percentage
    
    print(format_currency(150000))  # ₹1,50,000
    print(format_percentage(87.5))  # 87.50%
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Union, Dict, Any, List
import json


# ============================================================
# CURRENCY FORMATTING
# ============================================================

def format_currency(
    amount: Union[int, float],
    currency: str = "INR",
    show_symbol: bool = True,
    decimals: int = 2
) -> str:
    """
    Format currency with proper symbol and thousand separators.
    
    Args:
        amount: Amount to format
        currency: Currency code (INR, USD, EUR, GBP)
        show_symbol: Include currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
        
    Example:
        >>> format_currency(150000)
        '₹1,50,000.00'
        >>> format_currency(1500, currency="USD")
        '$1,500.00'
    """
    # Currency symbols
    symbols = {
        'INR': '₹',
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'AED': 'AED '
    }
    
    symbol = symbols.get(currency, currency + ' ') if show_symbol else ''
    
    # Indian numbering system for INR (lakhs/crores)
    if currency == 'INR':
        # Format with Indian comma placement
        s = f"{amount:,.{decimals}f}"
        # Convert Western (1,000,000) to Indian (10,00,000)
        parts = s.split('.')
        integer_part = parts[0].replace(',', '')
        
        if len(integer_part) > 3:
            last_3 = integer_part[-3:]
            remaining = integer_part[:-3]
            # Add commas every 2 digits for remaining
            formatted = ''
            for i, digit in enumerate(reversed(remaining)):
                if i > 0 and i % 2 == 0:
                    formatted = ',' + formatted
                formatted = digit + formatted
            integer_part = formatted + ',' + last_3
        
        if decimals > 0:
            return f"{symbol}{integer_part}.{parts[1]}"
        else:
            return f"{symbol}{integer_part}"
    
    # Standard formatting for other currencies
    else:
        formatted = f"{amount:,.{decimals}f}"
        return f"{symbol}{formatted}"


def format_currency_compact(amount: Union[int, float], currency: str = "INR") -> str:
    """
    Format currency in compact notation (K, L, Cr).
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Compact formatted string
        
    Example:
        >>> format_currency_compact(150000)
        '₹1.5L'
        >>> format_currency_compact(5000000)
        '₹50L' or '₹5Cr'
    """
    symbols = {'INR': '₹', 'USD': '$', 'EUR': '€', 'GBP': '£'}
    symbol = symbols.get(currency, currency + ' ')
    
    if currency == 'INR':
        # Indian notation: Thousand, Lakh, Crore
        if amount >= 10000000:  # 1 Crore
            return f"{symbol}{amount/10000000:.2f}Cr"
        elif amount >= 100000:  # 1 Lakh
            return f"{symbol}{amount/100000:.2f}L"
        elif amount >= 1000:  # 1 Thousand
            return f"{symbol}{amount/1000:.2f}K"
        else:
            return f"{symbol}{amount:.2f}"
    else:
        # Western notation: K, M, B
        if amount >= 1000000000:  # Billion
            return f"{symbol}{amount/1000000000:.2f}B"
        elif amount >= 1000000:  # Million
            return f"{symbol}{amount/1000000:.2f}M"
        elif amount >= 1000:  # Thousand
            return f"{symbol}{amount/1000:.2f}K"
        else:
            return f"{symbol}{amount:.2f}"


# ============================================================
# PERCENTAGE FORMATTING
# ============================================================

def format_percentage(
    value: Union[int, float],
    decimals: int = 2,
    show_symbol: bool = True
) -> str:
    """
    Format percentage value.
    
    Args:
        value: Percentage value (0-100)
        decimals: Number of decimal places
        show_symbol: Include % symbol
        
    Returns:
        Formatted percentage string
        
    Example:
        >>> format_percentage(87.5)
        '87.50%'
        >>> format_percentage(100, decimals=0)
        '100%'
    """
    formatted = f"{value:.{decimals}f}"
    return f"{formatted}%" if show_symbol else formatted


def format_confidence(score: float) -> str:
    """
    Format confidence score with label.
    
    Args:
        score: Confidence score (0-1)
        
    Returns:
        Formatted string with label
        
    Example:
        >>> format_confidence(0.95)
        'High (95%)'
        >>> format_confidence(0.65)
        'Medium (65%)'
    """
    percentage = score * 100
    
    if score >= 0.8:
        label = "High"
    elif score >= 0.6:
        label = "Medium"
    else:
        label = "Low"
    
    return f"{label} ({percentage:.0f}%)"


# ============================================================
# DATE FORMATTING
# ============================================================

def format_date(
    date_value: Union[str, datetime, pd.Timestamp, date],
    format_style: str = "standard"
) -> str:
    """
    Format date in various styles.
    
    Args:
        date_value: Date to format
        format_style: Style - 'standard', 'short', 'long', 'iso'
        
    Returns:
        Formatted date string
        
    Example:
        >>> format_date(datetime(2026, 1, 7), 'standard')
        '07-Jan-2026'
        >>> format_date(datetime(2026, 1, 7), 'long')
        '7 January 2026'
    """
    # Convert to datetime
    if isinstance(date_value, str):
        dt = pd.to_datetime(date_value)
    elif isinstance(date_value, pd.Timestamp):
        dt = date_value.to_pydatetime()
    elif isinstance(date_value, date) and not isinstance(date_value, datetime):
        dt = datetime.combine(date_value, datetime.min.time())
    else:
        dt = date_value
    
    formats = {
        'standard': '%d-%b-%Y',      # 07-Jan-2026
        'short': '%d/%m/%Y',         # 07/01/2026
        'long': '%-d %B %Y',         # 7 January 2026
        'iso': '%Y-%m-%d',           # 2026-01-07
        'datetime': '%d-%b-%Y %H:%M' # 07-Jan-2026 22:30
    }
    
    # Handle Windows vs Unix for %-d (no leading zero)
    fmt = formats.get(format_style, formats['standard'])
    try:
        return dt.strftime(fmt)
    except ValueError:
        # Windows doesn't support %-d, use %#d instead
        fmt = fmt.replace('%-d', '%#d')
        return dt.strftime(fmt)


def format_date_range(
    start_date: Union[datetime, str],
    end_date: Union[datetime, str],
    format_style: str = "standard"
) -> str:
    """
    Format date range.
    
    Args:
        start_date: Start date
        end_date: End date
        format_style: Date format style
        
    Returns:
        Formatted date range string
        
    Example:
        >>> format_date_range('2026-01-01', '2026-01-31')
        '01-Jan-2026 to 31-Jan-2026'
    """
    start_str = format_date(start_date, format_style)
    end_str = format_date(end_date, format_style)
    return f"{start_str} to {end_str}"


# ============================================================
# NUMBER FORMATTING
# ============================================================

def format_number(
    value: Union[int, float],
    decimals: int = 0,
    use_separators: bool = True
) -> str:
    """
    Format number with thousand separators.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        use_separators: Use comma separators
        
    Returns:
        Formatted number string
        
    Example:
        >>> format_number(1500000)
        '1,500,000'
        >>> format_number(1234.567, decimals=2)
        '1,234.57'
    """
    if use_separators:
        return f"{value:,.{decimals}f}"
    else:
        return f"{value:.{decimals}f}"


def format_large_number(value: Union[int, float]) -> str:
    """
    Format large numbers with suffixes.
    
    Args:
        value: Number to format
        
    Returns:
        Formatted string with suffix
        
    Example:
        >>> format_large_number(1500000)
        '1.5M'
        >>> format_large_number(2500)
        '2.5K'
    """
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.0f}"


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def format_output_dict(
    data: Dict[str, Any],
    indent: int = 2,
    sort_keys: bool = False
) -> str:
    """
    Format dictionary for clean output/logging.
    
    Args:
        data: Dictionary to format
        indent: Indentation level
        sort_keys: Sort dictionary keys
        
    Returns:
        Formatted JSON string
        
    Example:
        >>> result = {'vendor': 'ABC', 'score': 0.95, 'amount': 150000}
        >>> print(format_output_dict(result))
    """
    # Convert non-serializable types
    def convert_types(obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj
    
    # Deep convert
    def deep_convert(d):
        if isinstance(d, dict):
            return {k: deep_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [deep_convert(item) for item in d]
        else:
            return convert_types(d)
    
    converted = deep_convert(data)
    return json.dumps(converted, indent=indent, sort_keys=sort_keys)


def format_list_output(
    items: List[Any],
    max_items: int = 10,
    delimiter: str = ", "
) -> str:
    """
    Format list for output with truncation.
    
    Args:
        items: List to format
        max_items: Maximum items to show
        delimiter: Separator between items
        
    Returns:
        Formatted string
        
    Example:
        >>> vendors = ['ABC', 'XYZ', 'PQR', ...]
        >>> format_list_output(vendors, max_items=3)
        'ABC, XYZ, PQR... (10 total)'
    """
    if len(items) <= max_items:
        return delimiter.join(str(item) for item in items)
    else:
        shown = delimiter.join(str(item) for item in items[:max_items])
        return f"{shown}... ({len(items)} total)"


# ============================================================
# TABLE FORMATTING
# ============================================================

def format_dataframe_summary(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Format DataFrame summary for logging.
    
    Args:
        df: DataFrame to summarize
        max_rows: Maximum rows to show
        
    Returns:
        Formatted summary string
        
    Example:
        >>> print(format_dataframe_summary(vendor_df))
    """
    summary = []
    summary.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    summary.append(f"Columns: {', '.join(df.columns[:5])}")
    if len(df.columns) > 5:
        summary.append(f"... (+{len(df.columns) - 5} more)")
    
    return "\n".join(summary)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING FORMATTERS")
    print("="*60)
    
    # Test currency formatting
    print("\n✅ Testing Currency Formatting...")
    print(f"   INR Standard: {format_currency(150000)}")
    print(f"   INR Compact:  {format_currency_compact(5000000)}")
    print(f"   USD Standard: {format_currency(1500, currency='USD')}")
    print(f"   USD Compact:  {format_currency_compact(1500000, currency='USD')}")
    
    # Test percentage formatting
    print("\n✅ Testing Percentage Formatting...")
    print(f"   Percentage:   {format_percentage(87.5)}")
    print(f"   Confidence:   {format_confidence(0.95)}")
    
    # Test date formatting
    print("\n✅ Testing Date Formatting...")
    test_date = datetime(2026, 1, 7, 22, 30)
    print(f"   Standard:     {format_date(test_date, 'standard')}")
    print(f"   Short:        {format_date(test_date, 'short')}")
    print(f"   ISO:          {format_date(test_date, 'iso')}")
    print(f"   Date Range:   {format_date_range('2026-01-01', '2026-01-31')}")
    
    # Test number formatting
    print("\n✅ Testing Number Formatting...")
    print(f"   Large Number: {format_number(1500000)}")
    print(f"   Compact:      {format_large_number(2500000)}")
    
    # Test output formatting
    print("\n✅ Testing Output Formatting...")
    test_dict = {
        'vendor_name': 'ABC Corp',
        'score': 0.95,
        'amount': 150000,
        'date': datetime(2026, 1, 7)
    }
    print("   Dictionary Output:")
    print(format_output_dict(test_dict, indent=4))
    
    # Test list formatting
    print("\n✅ Testing List Formatting...")
    vendors = ['ABC', 'XYZ', 'PQR', 'LMN', 'DEF', 'GHI']
    print(f"   List Output:  {format_list_output(vendors, max_items=3)}")
    
    print("\n" + "="*60)
    print("🎉 FORMATTER TESTS COMPLETE")
    print("="*60)
