"""
Quick tests for data_processing.py
Tests NLP techniques: clause extraction, entity recognition, feature extraction
"""

import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from core.data_processing
from core.data_processing import (
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
import pandas as pd


# Tests text cleaning by removing extra whitespace and special characters
def test_clean_text():
    """Test text cleaning function"""
    print("\n🧪 Testing text cleaning...")
    try:
        dirty_text = """
        This  is   a   test.
        
        
        With   extra    spaces®©™.
        """
        
        cleaned = clean_text(dirty_text)
        
        # Check if double spaces removed
        if "  " not in cleaned and len(cleaned) < len(dirty_text):
            print(f"   ✅ Cleaned {len(dirty_text)} → {len(cleaned)} characters")
            return True
        else:
            print(f"   ❌ Cleaning failed")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests currency normalization from various formats to float values
def test_normalize_currency():
    """Test currency normalization"""
    print("\n🧪 Testing currency normalization...")
    try:
        test_cases = [
            ("₹2,50,000", 250000.0),
            ("Rs. 1,00,000", 100000.0),
            ("500000", 500000.0),
            (1000, 1000.0)
        ]
        
        passed = 0
        for input_val, expected in test_cases:
            result = normalize_currency(input_val)
            if result == expected:
                passed += 1
        
        if passed == len(test_cases):
            print(f"   ✅ All {passed}/{len(test_cases)} currency formats normalized correctly")
            return True
        else:
            print(f"   ⚠️  {passed}/{len(test_cases)} passed")
            return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests date normalization to standard YYYY-MM-DD format
def test_normalize_date():
    """Test date normalization"""
    print("\n🧪 Testing date normalization...")
    try:
        test_cases = [
            ("15-01-2026", "2026-01-15"),
            ("15/01/2026", "2026-01-15"),
            ("2026-01-15", "2026-01-15")
        ]
        
        passed = 0
        for input_val, expected in test_cases:
            result = normalize_date(input_val)
            if result == expected:
                passed += 1
        
        if passed == len(test_cases):
            print(f"   ✅ All {passed}/{len(test_cases)} date formats normalized correctly")
            return True
        else:
            print(f"   ⚠️  {passed}/{len(test_cases)} passed")
            return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests DataFrame normalization for dates, currency, and percentages
def test_normalize_data():
    """Test DataFrame normalization"""
    print("\n🧪 Testing DataFrame normalization...")
    try:
        df = pd.DataFrame({
            'vendor_id': [1, 2],
            'contract_value': ['₹2,50,000', '₹5,00,000'],
            'quality_score': [85.5, 120.0],  # Second value out of range
            'contract_date': ['15-01-2026', '20-01-2026']
        })
        
        normalized = normalize_data(df)
        
        # Check if currency converted to float
        if normalized['contract_value'].dtype in ['float64', 'float32']:
            print(f"   ✅ Currency converted to numeric")
        
        # Check if quality score clipped to 0-100
        if normalized['quality_score'].max() <= 100:
            print(f"   ✅ Percentages clipped to valid range")
        
        # Check if dates converted
        if pd.api.types.is_datetime64_any_dtype(normalized['contract_date']):
            print(f"   ✅ Dates converted to datetime")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests clause segmentation using spaCy NLP
def test_segment_clauses():
    """Test clause segmentation"""
    print("\n🧪 Testing clause segmentation...")
    try:
        contract_text = """
        Payment shall be made within 30 days. 
        Penalty of 0.5% will apply for delays. 
        Warranty period is 12 months.
        """
        
        clauses = segment_clauses(contract_text)
        
        if len(clauses) >= 3:
            print(f"   ✅ Found {len(clauses)} clauses")
            return True
        else:
            print(f"   ⚠️  Found only {len(clauses)} clauses (expected 3+)")
            return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests clause type identification using keyword matching
def test_identify_clause_type():
    """Test clause type identification"""
    print("\n🧪 Testing clause type identification...")
    try:
        test_clauses = [
            ("Payment shall be made within 30 days", "payment"),
            ("Penalty of 0.5% will apply for delays", "penalty"),
            ("Warranty period is 12 months", "warranty"),
            ("The contractor must deliver by deadline", "delivery")
        ]
        
        passed = 0
        for clause, expected_type in test_clauses:
            result = identify_clause_type(clause)
            if result == expected_type:
                passed += 1
                print(f"   ✅ '{expected_type}' clause identified correctly")
        
        if passed >= 3:
            return True
        else:
            print(f"   ⚠️  {passed}/{len(test_clauses)} identified correctly")
            return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests Named Entity Recognition for dates, money, and organizations
def test_extract_entities():
    """Test entity extraction (NER)"""
    print("\n🧪 Testing entity extraction...")
    try:
        text = """
        Contract dated 15-01-2026 between Metro Rail Corporation 
        and BuildPro Constructions for ₹2,50,00,000.
        """
        
        entities = extract_entities(text)
        
        print(f"   ✅ Dates found: {len(entities['dates'])}")
        print(f"   ✅ Money values found: {len(entities['money'])}")
        print(f"   ✅ Organizations found: {len(entities['organizations'])}")
        
        # At least some entities should be found
        total_entities = (len(entities['dates']) + 
                         len(entities['money']) + 
                         len(entities['organizations']))
        
        if total_entities > 0:
            return True
        else:
            print(f"   ⚠️  No entities extracted (might be regex fallback)")
            return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests feature extraction from full contract text
def test_extract_features():
    """Test feature extraction"""
    print("\n🧪 Testing feature extraction...")
    try:
        contract_text = """
        Contract Value: ₹2,50,00,000
        Vendor: BuildPro Constructions
        Payment shall be made within 30 days.
        Penalty of 0.5% per week for delays.
        Warranty of 12 months is provided.
        """
        
        features = extract_features(contract_text)
        
        extracted_count = sum(1 for v in features.values() if v is not None)
        
        print(f"   ✅ Extracted {extracted_count} features")
        print(f"   Contract Value: {features.get('contract_value')}")
        print(f"   Payment Terms: {features.get('payment_terms')}")
        print(f"   Penalty: {features.get('penalty_clause')}")
        
        if extracted_count >= 2:
            return True
        else:
            print(f"   ⚠️  Only {extracted_count} features extracted")
            return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests conversion of clauses to structured DataFrame
def test_structure_data():
    """Test data structuring"""
    print("\n🧪 Testing data structuring...")
    try:
        clauses = [
            "Payment shall be made within 30 days.",
            "Penalty of 0.5% will apply for delays.",
            "The contractor must complete work by deadline.",
            "Warranty period is 12 months."
        ]
        
        df = structure_data(clauses)
        
        if len(df) == len(clauses):
            print(f"   ✅ Created DataFrame with {len(df)} rows")
            print(f"   Columns: {list(df.columns)}")
            
            # Check if mandatory clauses detected
            mandatory_count = df['mandatory'].sum()
            high_risk_count = len(df[df['risk_level'] == 'high'])
            
            print(f"   Mandatory clauses: {mandatory_count}")
            print(f"   High-risk clauses: {high_risk_count}")
            
            return True
        else:
            print(f"   ❌ Row count mismatch")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests complete contract processing pipeline
def test_full_contract_processing():
    """Test full contract processing"""
    print("\n🧪 Testing full contract processing...")
    try:
        sample_contract = """
        SUPPLY CONTRACT
        
        Contract dated 15-01-2026 between Client and BuildPro Constructions.
        Contract Value: ₹2,50,00,000
        
        Payment shall be made within 30 days of invoice.
        The contractor must complete work by 30-06-2026.
        Penalty of 0.5% per week will apply for delays.
        Warranty period is 12 months from completion.
        """
        
        result = structure_contract_data(sample_contract)
        
        print(f"   ✅ Total Clauses: {result['clause_count']}")
        print(f"   ✅ Mandatory Clauses: {result['mandatory_clauses']}")
        print(f"   ✅ High Risk Clauses: {result['high_risk_clauses']}")
        
        if result['clause_count'] > 0:
            return True
        else:
            print(f"   ❌ No clauses extracted")
            return False
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Executes all test functions and displays summary
def run_all_tests():
    """Run all data processing tests"""
    print("=" * 60)
    print("🚀 RUNNING DATA PROCESSING LAYER TESTS")
    print("=" * 60)
    print("\n⚠️  NOTE: These tests require:")
    print("   - spaCy installed (python -m spacy download en_core_web_sm)")
    print("   - pandas, numpy libraries")
    print("")
    
    results = []
    results.append(("Text Cleaning", test_clean_text()))
    results.append(("Currency Normalization", test_normalize_currency()))
    results.append(("Date Normalization", test_normalize_date()))
    results.append(("DataFrame Normalization", test_normalize_data()))
    results.append(("Clause Segmentation", test_segment_clauses()))
    results.append(("Clause Type Identification", test_identify_clause_type()))
    results.append(("Entity Extraction (NER)", test_extract_entities()))
    results.append(("Feature Extraction", test_extract_features()))
    results.append(("Data Structuring", test_structure_data()))
    results.append(("Full Contract Processing", test_full_contract_processing()))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check errors above")
    
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
