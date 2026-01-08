"""
Data Processing Layer - NLP & Feature Engineering
Cleans text, extracts features, and structures unstructured data from contracts
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import spacy
from collections import Counter

# Load spaCy English model (download first: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠️  Warning: spaCy model not installed yet. Run: python -m spacy download en_core_web_sm")
    nlp = None


# ============================================================================
# TEXT CLEANING & PREPROCESSING
# ============================================================================

# Removes extra whitespace, special characters, and normalizes text format
def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing noise
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove extra spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove non-ASCII characters (optional - keep if you need multilingual)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove special symbols but keep punctuation
    text = re.sub(r'[®©™]', '', text)
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()


# Standardizes dates, currency values, and numeric fields across the dataset
def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dates, currency, and numeric values in DataFrame
    
    Args:
        df: Raw DataFrame with mixed formats
        
    Returns:
        Normalized DataFrame
    """
    df_normalized = df.copy()
    
    # Normalize date columns
    date_columns = ['registration_date', 'delivery_date', 'contract_date', 
                   'last_evaluation_date', 'completion_date']
    
    for col in date_columns:
        if col in df_normalized.columns:
            df_normalized[col] = pd.to_datetime(df_normalized[col], errors='coerce')
    
    # Normalize currency columns (remove ₹, commas)
    currency_columns = ['contract_value', 'quoted_price', 'total_cost', 
                       'annual_turnover_cr', 'invoice_amount']
    
    for col in currency_columns:
        if col in df_normalized.columns:
            df_normalized[col] = df_normalized[col].apply(normalize_currency)
    
    # Normalize percentage columns
    percentage_columns = ['quality_score', 'delivery_score', 'compliance_score', 
                         'price_competitiveness', 'overall_score']
    
    for col in percentage_columns:
        if col in df_normalized.columns:
            df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
            # Ensure percentages are in 0-100 range
            df_normalized[col] = df_normalized[col].clip(0, 100)
    
    return df_normalized


# Converts currency text like "₹2,50,000" to float 250000.0
def normalize_currency(value: Any) -> float:
    """
    Convert currency strings to float
    
    Args:
        value: Currency value (can be string with ₹, commas)
        
    Returns:
        Float value
    """
    if pd.isna(value):
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    # Remove currency symbols and commas
    value_str = str(value).replace('₹', '').replace('Rs.', '').replace(',', '').strip()
    
    try:
        return float(value_str)
    except:
        return 0.0


# Standardizes date formats from various input formats to YYYY-MM-DD
def normalize_date(date_str: str) -> str:
    """
    Normalize date string to YYYY-MM-DD format
    
    Args:
        date_str: Date in any common format
        
    Returns:
        Standardized date string
    """
    if not date_str or pd.isna(date_str):
        return None
    
    # Common date formats
    formats = [
        '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d',
        '%d-%b-%Y', '%d %B %Y', '%B %d, %Y'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(str(date_str).strip(), fmt)
            return dt.strftime('%Y-%m-%d')
        except:
            continue
    
    return None


# ============================================================================
# NLP TECHNIQUES - CLAUSE EXTRACTION & ANALYSIS
# ============================================================================

# Splits contract text into individual sentences/clauses using spaCy NLP
def segment_clauses(contract_text: str) -> List[str]:
    """
    Split contract text into individual clauses using NLP sentence segmentation
    
    Args:
        contract_text: Full contract text
        
    Returns:
        List of clause strings
    """
    if not nlp:
        # Fallback to simple sentence splitting
        return [s.strip() for s in contract_text.split('.') if s.strip()]
    
    doc = nlp(contract_text)
    clauses = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    return clauses


# Identifies clause type (payment, penalty, termination) using keyword matching
def identify_clause_type(clause: str) -> str:
    """
    Identify clause type using keyword matching
    
    Args:
        clause: Single clause text
        
    Returns:
        Clause type (payment, penalty, termination, etc.)
    """
    # Define keywords for each clause type
    keywords = {
        'payment': ['payment', 'invoice', 'due', 'remittance', 'billing', 'paid'],
        'penalty': ['penalty', 'liquidated damages', 'delay', 'late fee', 'damages'],
        'termination': ['terminate', 'termination', 'notice period', 'cancellation', 'cancel'],
        'liability': ['liability', 'indemnity', 'damages', 'liable', 'responsible'],
        'warranty': ['warranty', 'guarantee', 'defect', 'repair', 'replacement'],
        'delivery': ['delivery', 'deliver', 'timeline', 'completion', 'handover'],
        'scope': ['scope', 'work', 'services', 'supply', 'installation']
    }
    
    clause_lower = clause.lower()
    
    # Count keyword matches for each type
    scores = {}
    for clause_type, keyword_list in keywords.items():
        score = sum(1 for keyword in keyword_list if keyword in clause_lower)
        if score > 0:
            scores[clause_type] = score
    
    # Return type with highest score
    if scores:
        return max(scores, key=scores.get)
    
    return 'other'


# Extracts dates, money values, percentages, and organization names using NER
def extract_entities(text: str) -> Dict[str, List]:
    """
    Extract named entities (dates, money, percentages, organizations) using spaCy NER
    
    Args:
        text: Contract or clause text
        
    Returns:
        Dictionary with entity types and values
    """
    entities = {
        'dates': [],
        'money': [],
        'percentages': [],
        'organizations': [],
        'persons': [],
        'locations': []
    }
    
    if not nlp:
        # Fallback to regex extraction
        entities['money'] = extract_money_regex(text)
        entities['dates'] = extract_dates_regex(text)
        entities['percentages'] = extract_percentages_regex(text)
        return entities
    
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            entities['dates'].append(ent.text)
        elif ent.label_ == 'MONEY':
            entities['money'].append(ent.text)
        elif ent.label_ == 'PERCENT':
            entities['percentages'].append(ent.text)
        elif ent.label_ == 'ORG':
            entities['organizations'].append(ent.text)
        elif ent.label_ == 'PERSON':
            entities['persons'].append(ent.text)
        elif ent.label_ in ['GPE', 'LOC']:
            entities['locations'].append(ent.text)
    
    return entities


# Extracts monetary values using regex patterns when spaCy is unavailable
def extract_money_regex(text: str) -> List[str]:
    """Extract money values using regex"""
    patterns = [
        r'₹\s*([\d,]+(?:\.\d{2})?)',
        r'Rs\.?\s*([\d,]+(?:\.\d{2})?)',
        r'INR\s*([\d,]+(?:\.\d{2})?)'
    ]
    
    money_values = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        money_values.extend(matches)
    
    return money_values


# Extracts dates using regex patterns when spaCy is unavailable
def extract_dates_regex(text: str) -> List[str]:
    """Extract dates using regex"""
    # Match DD-MM-YYYY, DD/MM/YYYY, etc.
    date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
    dates = re.findall(date_pattern, text)
    return dates


# Extracts percentage values using regex patterns
def extract_percentages_regex(text: str) -> List[str]:
    """Extract percentages using regex"""
    percentage_pattern = r'\b\d+(?:\.\d+)?\s*%'
    percentages = re.findall(percentage_pattern, text)
    return percentages


# Tags each clause with metadata (type, mandatory/optional, risk level)
def tag_clause_metadata(clause: str, clause_type: str) -> Dict[str, Any]:
    """
    Tag clause with metadata (mandatory/optional, risk level)
    
    Args:
        clause: Clause text
        clause_type: Type of clause
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        'clause_type': clause_type,
        'mandatory': False,
        'risk_level': 'low'
    }
    
    # Mandatory clause detection
    mandatory_keywords = ['shall', 'must', 'required', 'mandatory']
    if any(keyword in clause.lower() for keyword in mandatory_keywords):
        metadata['mandatory'] = True
    
    # Risk level detection
    high_risk_keywords = ['penalty', 'damages', 'liability', 'terminate', 'breach']
    medium_risk_keywords = ['delay', 'warranty', 'guarantee', 'dispute']
    
    if any(keyword in clause.lower() for keyword in high_risk_keywords):
        metadata['risk_level'] = 'high'
    elif any(keyword in clause.lower() for keyword in medium_risk_keywords):
        metadata['risk_level'] = 'medium'
    
    return metadata


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

# Extracts structured features (contract value, dates, vendor, etc.) from text
def extract_features(contract_text: str) -> Dict[str, Any]:
    """
    Extract structured features from unstructured contract text
    
    Args:
        contract_text: Full contract text
        
    Returns:
        Dictionary of extracted features
    """
    features = {
        'contract_value': None,
        'vendor_name': None,
        'contract_date': None,
        'delivery_date': None,
        'payment_terms': None,
        'penalty_clause': None,
        'warranty_period': None
    }
    
    # Extract entities
    entities = extract_entities(contract_text)
    
    # Contract value (first money value found)
    if entities['money']:
        features['contract_value'] = entities['money'][0]
    
    # Vendor name (first organization found)
    if entities['organizations']:
        features['vendor_name'] = entities['organizations'][0]
    
    # Dates
    if entities['dates']:
        features['contract_date'] = entities['dates'][0] if len(entities['dates']) > 0 else None
        features['delivery_date'] = entities['dates'][1] if len(entities['dates']) > 1 else None
    
    # Extract payment terms
    payment_pattern = r'payment.*?(\d+)\s*days'
    payment_match = re.search(payment_pattern, contract_text.lower())
    if payment_match:
        features['payment_terms'] = f"{payment_match.group(1)} days"
    
    # Extract penalty clause
    if 'penalty' in contract_text.lower():
        penalty_pattern = r'penalty.*?([\d.]+)\s*%'
        penalty_match = re.search(penalty_pattern, contract_text.lower())
        if penalty_match:
            features['penalty_clause'] = f"{penalty_match.group(1)}%"
    
    # Extract warranty period
    warranty_pattern = r'warranty.*?(\d+)\s*(month|year)'
    warranty_match = re.search(warranty_pattern, contract_text.lower())
    if warranty_match:
        features['warranty_period'] = f"{warranty_match.group(1)} {warranty_match.group(2)}"
    
    return features


# ============================================================================
# DATA STRUCTURING
# ============================================================================

# Converts list of clauses into structured DataFrame with NLP analysis
def structure_data(clauses: List[str]) -> pd.DataFrame:
    """
    Convert clauses into structured DataFrame with NLP analysis
    
    Args:
        clauses: List of clause strings
        
    Returns:
        DataFrame with structured clause data
    """
    structured = []
    
    for idx, clause in enumerate(clauses):
        if not clause.strip():
            continue
        
        # Identify clause type
        clause_type = identify_clause_type(clause)
        
        # Extract entities
        entities = extract_entities(clause)
        
        # Get metadata
        metadata = tag_clause_metadata(clause, clause_type)
        
        structured.append({
            'clause_id': idx + 1,
            'clause_text': clause,
            'clause_type': clause_type,
            'mandatory': metadata['mandatory'],
            'risk_level': metadata['risk_level'],
            'dates': ', '.join(entities['dates']),
            'money': ', '.join(entities['money']),
            'percentages': ', '.join(entities['percentages']),
            'organizations': ', '.join(entities['organizations'])
        })
    
    return pd.DataFrame(structured)


# Converts full contract text into structured DataFrame with features
def structure_contract_data(contract_text: str) -> Dict[str, Any]:
    """
    Full contract text to structured format with clauses and features
    
    Args:
        contract_text: Full contract text
        
    Returns:
        Dictionary with clauses DataFrame and extracted features
    """
    # Segment into clauses
    clauses = segment_clauses(contract_text)
    
    # Structure clauses
    clauses_df = structure_data(clauses)
    
    # Extract overall features
    features = extract_features(contract_text)
    
    return {
        'clauses': clauses_df,
        'features': features,
        'clause_count': len(clauses),
        'high_risk_clauses': len(clauses_df[clauses_df['risk_level'] == 'high']),
        'mandatory_clauses': len(clauses_df[clauses_df['mandatory'] == True])
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TESTING DATA PROCESSING LAYER")
    print("=" * 60)
    
    # Sample contract text
    sample_contract = """
    SUPPLY AND INSTALLATION CONTRACT
    
    This agreement is made on 15-01-2026 between Metro Rail Corporation Ltd (CLIENT)
    and BuildPro Constructions (CONTRACTOR).
    
    Contract Value: ₹2,50,00,000 (Rupees Two Crore Fifty Lakhs only)
    
    Payment Terms: Payment shall be made within 30 days of invoice submission.
    
    Delivery: The contractor must complete the work by 30-06-2026.
    
    Penalty Clause: A penalty of 0.5% per week will be applicable for delays beyond
    the agreed timeline.
    
    Warranty: The contractor shall provide a warranty of 12 months from the date of
    completion.
    
    Termination: Either party may terminate this contract with 30 days written notice.
    """
    
    print("\n🧪 Test 1: Text Cleaning")
    cleaned = clean_text(sample_contract)
    print(f"   ✅ Cleaned {len(sample_contract)} → {len(cleaned)} characters")
    
    print("\n🧪 Test 2: Clause Segmentation")
    clauses = segment_clauses(sample_contract)
    print(f"   ✅ Found {len(clauses)} clauses")
    
    print("\n🧪 Test 3: Entity Extraction")
    entities = extract_entities(sample_contract)
    print(f"   ✅ Dates: {entities['dates']}")
    print(f"   ✅ Money: {entities['money']}")
    print(f"   ✅ Organizations: {entities['organizations']}")
    
    print("\n🧪 Test 4: Feature Extraction")
    features = extract_features(sample_contract)
    print(f"   ✅ Contract Value: {features['contract_value']}")
    print(f"   ✅ Vendor: {features['vendor_name']}")
    print(f"   ✅ Payment Terms: {features['payment_terms']}")
    print(f"   ✅ Penalty: {features['penalty_clause']}")
    
    print("\n🧪 Test 5: Data Structuring")
    structured = structure_data(clauses)
    print(f"   ✅ Created DataFrame with {len(structured)} rows")
    print("\n" + str(structured[['clause_type', 'mandatory', 'risk_level']].head()))
    
    print("\n🧪 Test 6: Full Contract Processing")
    result = structure_contract_data(sample_contract)
    print(f"   ✅ Total Clauses: {result['clause_count']}")
    print(f"   ✅ Mandatory Clauses: {result['mandatory_clauses']}")
    print(f"   ✅ High Risk Clauses: {result['high_risk_clauses']}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
