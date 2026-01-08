"""
Quick tests for input_layer.py
"""

import sys
import os

# Add parent directory to Python path so it can find 'core' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now imports will work
from core.input_layer import (
    load_csv, 
    load_pdf, 
    load_excel, 
    validate_input,
    validate_file_upload,
    VALIDATION_PATTERNS
)
import pandas as pd


def test_load_csv():
    """Test CSV loading"""
    print("\n🧪 Testing CSV loading...")
    try:
        # Correct path: tests/ -> parent -> mockdata/
        df = load_csv("../mockdata/vendors.csv")
        print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_load_pdf():
    """Test PDF loading"""
    print("\n🧪 Testing PDF loading...")
    try:
        # Correct path: tests/ -> parent -> mockdata/samplepdf/
        text = load_pdf("../mockdata/samplepdf/sample_contract.pdf")
        print(f"   ✅ Loaded PDF: {len(text)} characters")
        print(f"   Preview: {text[:100]}...")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False



def test_validation_pass():
    """Test validation with good data"""
    print("\n🧪 Testing validation (should pass)...")
    df = load_csv("../mockdata/vendors.csv")
    
    is_valid, errors = validate_input(
        data=df,
        expected_columns=['vendor_id', 'vendor_name'],
        required_non_null=['vendor_name']
    )
    
    if is_valid:
        print("   ✅ Validation passed")
        return True
    else:
        print(f"   ❌ Unexpected errors: {errors}")
        return False


def test_validation_fail():
    """Test validation with missing columns"""
    print("\n🧪 Testing validation (should fail - missing column)...")
    df = load_csv("../mockdata/vendors.csv")
    
    is_valid, errors = validate_input(
        data=df,
        expected_columns=['vendor_id', 'missing_column'],  # This column doesn't exist
    )
    
    if not is_valid and len(errors) > 0:
        print(f"   ✅ Correctly caught error: {errors[0]}")
        return True
    else:
        print("   ❌ Should have failed but didn't")
        return False


def test_regex_validation():
    """Test GSTIN regex validation"""
    print("\n🧪 Testing GSTIN regex validation...")
    
    # Create test data
    test_data = pd.DataFrame({
        'vendor_id': ['V001', 'V002', 'V003'],
        'gstin': ['22AAAAA0000A1Z5', 'INVALID', '27BBBBB1111B1Z6']
    })
    
    is_valid, errors = validate_input(
        data=test_data,
        validation_rules={'gstin': VALIDATION_PATTERNS['gstin']}
    )
    
    if not is_valid and 'invalid format' in errors[0].lower():
        print(f"   ✅ Correctly caught invalid GSTIN: {errors[0]}")
        return True
    else:
        print(f"   ❌ Regex validation didn't work: {errors}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🚀 RUNNING INPUT LAYER TESTS")
    print("=" * 60)
    
    results = []
    results.append(("CSV Loading", test_load_csv()))
    results.append(("PDF Loading", test_load_pdf()))
    results.append(("Validation Pass", test_validation_pass()))
    results.append(("Validation Fail", test_validation_fail()))
    results.append(("Regex Validation", test_regex_validation()))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
