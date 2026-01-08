"""
Constants Module - System-Wide Configuration Values
===================================================

Centralized configuration for:
- Confidence thresholds for ML predictions
- Anomaly detection thresholds
- Date and currency formats
- Stage names and descriptions
- File paths and directories
- Validation rules
- API configurations

Usage:
    from utils.constants import CONFIDENCE_THRESHOLDS, STAGE_NAMES
    
    if score >= CONFIDENCE_THRESHOLDS['high']:
        status = 'approved'
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

CONFIDENCE_THRESHOLDS = {
    'high': 0.80,      # 80%+ = High confidence
    'medium': 0.60,    # 60-79% = Medium confidence
    'low': 0.40,       # 40-59% = Low confidence
    'reject': 0.40     # Below 40% = Reject/Flag
}

# ML Model Performance Thresholds
MODEL_PERFORMANCE = {
    'min_accuracy': 0.75,        # Minimum acceptable accuracy
    'min_precision': 0.70,       # Minimum precision
    'min_recall': 0.70,          # Minimum recall
    'min_f1_score': 0.70,        # Minimum F1 score
    'min_r2_score': 0.65         # Minimum R² for regression
}


# ============================================================
# ANOMALY DETECTION THRESHOLDS
# ============================================================

ANOMALY_THRESHOLDS = {
    'contamination': 0.05,           # Expected % of anomalies (5%)
    'z_score_threshold': 3.0,        # Standard deviations for outliers
    'iqr_multiplier': 1.5,           # IQR multiplier for outliers
    
    # Severity levels based on anomaly score
    'severity': {
        'critical': 0.90,   # Score > 0.90
        'high': 0.75,       # Score 0.75-0.90
        'medium': 0.50,     # Score 0.50-0.75
        'low': 0.25         # Score 0.25-0.50
    }
}


# ============================================================
# VENDOR SCORING WEIGHTS
# ============================================================

VENDOR_WEIGHTS = {
    'delivery_score': 0.30,           # 30% weight
    'quality_score': 0.25,            # 25% weight
    'cost_competitiveness': 0.25,     # 25% weight
    'payment_compliance': 0.10,       # 10% weight
    'dispute_history': 0.10           # 10% weight
}

# Vendor classification thresholds
VENDOR_CLASSIFICATION = {
    'excellent': 85,    # Score >= 85
    'good': 70,         # Score 70-84
    'average': 50,      # Score 50-69
    'poor': 50          # Score < 50
}


# ============================================================
# FINANCIAL THRESHOLDS
# ============================================================

FINANCIAL_THRESHOLDS = {
    # Invoice amounts (INR)
    'small_invoice': 50000,          # < 50K
    'medium_invoice': 500000,        # 50K - 5L
    'large_invoice': 5000000,        # 5L - 50L
    'critical_invoice': 5000000,     # > 50L (requires extra approval)
    
    # Price variation acceptable limits
    'max_price_increase': 0.15,      # 15% increase allowed
    'max_price_decrease': 0.20,      # 20% decrease allowed
    'critical_variance': 0.25,       # 25%+ variance = critical flag
    
    # Payment terms
    'standard_payment_days': 30,
    'max_payment_days': 90,
    'early_payment_discount': 0.02   # 2% discount for early payment
}


# ============================================================
# RISK ASSESSMENT THRESHOLDS
# ============================================================

RISK_LEVELS = {
    'critical': 80,     # Risk score >= 80
    'high': 60,         # Risk score 60-79
    'medium': 40,       # Risk score 40-59
    'low': 20,          # Risk score 20-39
    'minimal': 20       # Risk score < 20
}

RISK_WEIGHTS = {
    'financial_risk': 0.35,
    'operational_risk': 0.25,
    'compliance_risk': 0.20,
    'vendor_risk': 0.20
}


# ============================================================
# DATE FORMATS
# ============================================================

DATE_FORMATS = {
    'standard': '%Y-%m-%d',              # 2026-01-07
    'display': '%d-%b-%Y',               # 07-Jan-2026
    'short': '%d/%m/%Y',                 # 07/01/2026
    'long': '%d %B %Y',                  # 7 January 2026
    'datetime': '%Y-%m-%d %H:%M:%S',     # 2026-01-07 22:30:00
    'filename': '%Y%m%d_%H%M%S',         # 20260107_223000
    'indian': '%d-%m-%Y'                 # 07-01-2026
}


# ============================================================
# CURRENCY FORMATS
# ============================================================

CURRENCY_FORMATS = {
    'INR': {
        'symbol': '₹',
        'decimals': 2,
        'thousand_sep': ',',
        'use_indian_notation': True
    },
    'USD': {
        'symbol': '$',
        'decimals': 2,
        'thousand_sep': ',',
        'use_indian_notation': False
    },
    'EUR': {
        'symbol': '€',
        'decimals': 2,
        'thousand_sep': ',',
        'use_indian_notation': False
    },
    'GBP': {
        'symbol': '£',
        'decimals': 2,
        'thousand_sep': ',',
        'use_indian_notation': False
    }
}

DEFAULT_CURRENCY = 'INR'


# ============================================================
# STAGE NAMES AND DESCRIPTIONS
# ============================================================

STAGE_NAMES = {
    1: "Vendor Development & Approval Workflow",
    2: "Automated Document Creation (Contracts, POs)",
    3: "RFQ Price & Specification Anomaly Detection",
    4: "Vendor Shortlisting for Tenders",
    5: "AI-Driven Negotiation Insights",
    6: "Contract Review & Compliance Checks",
    7: "Change Order Analysis",
    8: "Risk Assessment for Procurement Decisions",
    9: "Predictive Insights for Vendor Shortlisting",
    10: "Spend Analysis & Optimization Suggestions",
    11: "Invoice Fraud Detection"
}

STAGE_DESCRIPTIONS = {
    1: "Automate vendor approval based on financial health, compliance, and past performance",
    2: "Generate contracts and purchase orders using templates and historical data",
    3: "Detect price anomalies and specification mismatches in RFQ responses",
    4: "Score and rank vendors based on multiple criteria for tender selection",
    5: "Provide data-driven insights for contract negotiations",
    6: "Analyze contracts for compliance with company policies and regulations",
    7: "Detect unusual patterns in change orders and cost variations",
    8: "Assess procurement risks across financial, operational, and vendor dimensions",
    9: "Predict vendor reliability and performance for future contracts",
    10: "Analyze spending patterns and suggest optimization opportunities",
    11: "Detect fraudulent invoices using anomaly detection and pattern matching"
}

# Stage dependencies (which stages need to run before others)
STAGE_DEPENDENCIES = {
    1: [],           # No dependencies
    2: [1],          # Needs vendor approval
    3: [],           # Independent
    4: [1, 3],       # Needs vendor data and RFQ analysis
    5: [4],          # Needs shortlisted vendors
    6: [2],          # Needs contract documents
    7: [2, 6],       # Needs contracts
    8: [1, 4],       # Needs vendor and procurement data
    9: [1, 4, 8],    # Needs historical vendor data
    10: [],          # Independent analysis
    11: [2]          # Needs invoice data
}


# ============================================================
# DATA VALIDATION RULES
# ============================================================

REQUIRED_COLUMNS = {
    'vendors': ['vendor_id', 'vendor_name', 'contact_email'],
    'invoices': ['invoice_id', 'vendor_id', 'amount', 'date'],
    'rfqs': ['rfq_id', 'item_description', 'quantity', 'expected_price'],
    'contracts': ['contract_id', 'vendor_id', 'start_date', 'end_date', 'value'],
    'purchase_orders': ['po_id', 'vendor_id', 'amount', 'date']
}

DATA_TYPES = {
    'vendor_id': 'string',
    'amount': 'float',
    'date': 'datetime',
    'quantity': 'int',
    'score': 'float'
}


# ============================================================
# ML MODEL CONFIGURATIONS
# ============================================================

ML_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    },
    'isolation_forest': {
        'contamination': 0.05,
        'random_state': 42
    }
}

# Train/test split ratio
TRAIN_TEST_SPLIT = 0.8

# Cross-validation folds
CV_FOLDS = 5


# ============================================================
# STATISTICAL ANALYSIS SETTINGS
# ============================================================

STATISTICAL_SETTINGS = {
    'correlation_threshold': 0.7,        # Strong correlation
    'trend_window': 12,                  # Months for trend analysis
    'outlier_method': 'iqr',             # 'iqr' or 'z_score'
    'confidence_interval': 0.95,         # 95% confidence
    'min_sample_size': 30                # Minimum samples for analysis
}


# ============================================================
# OUTPUT SETTINGS
# ============================================================

OUTPUT_FORMATS = ['csv', 'json', 'excel', 'pdf']
DEFAULT_OUTPUT_FORMAT = 'csv'

REPORT_SETTINGS = {
    'max_rows_display': 100,
    'include_charts': True,
    'include_summary': True,
    'timestamp_reports': True
}


# ============================================================
# LOGGING SETTINGS
# ============================================================

LOG_LEVELS = {
    'development': 'DEBUG',
    'testing': 'INFO',
    'production': 'WARNING'
}

DEFAULT_LOG_LEVEL = 'INFO'


# ============================================================
# EMAIL NOTIFICATION SETTINGS
# ============================================================

EMAIL_TEMPLATES = {
    'vendor_approval': 'Vendor {vendor_name} approved for procurement',
    'anomaly_detected': 'Anomaly detected in {entity_type}: {description}',
    'high_risk_alert': 'High risk detected: {risk_description}',
    'fraud_alert': 'Potential fraud detected in invoice {invoice_id}'
}


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("📋 SYSTEM CONSTANTS")
    print("="*60)
    
    print("\n📁 Project Paths:")
    print(f"   Root:    {PROJECT_ROOT}")
    print(f"   Data:    {DATA_DIR}")
    print(f"   Models:  {MODELS_DIR}")
    print(f"   Logs:    {LOGS_DIR}")
    
    print("\n🎯 Confidence Thresholds:")
    for level, threshold in CONFIDENCE_THRESHOLDS.items():
        print(f"   {level.capitalize()}: {threshold:.0%}")
    
    print("\n⚠️ Anomaly Detection:")
    print(f"   Contamination: {ANOMALY_THRESHOLDS['contamination']:.0%}")
    print(f"   Z-Score: {ANOMALY_THRESHOLDS['z_score_threshold']}")
    
    print("\n💰 Financial Thresholds (INR):")
    print(f"   Small Invoice:    ₹{FINANCIAL_THRESHOLDS['small_invoice']:,}")
    print(f"   Critical Invoice: ₹{FINANCIAL_THRESHOLDS['critical_invoice']:,}")
    
    print("\n📊 Vendor Weights:")
    for factor, weight in VENDOR_WEIGHTS.items():
        print(f"   {factor}: {weight:.0%}")
    
    print("\n🏗️ Stages:")
    for stage_id, name in list(STAGE_NAMES.items())[:5]:
        print(f"   Stage {stage_id}: {name}")
    print(f"   ... ({len(STAGE_NAMES)} total stages)")
    
    print("\n📅 Date Formats:")
    from datetime import datetime
    now = datetime.now()
    for name, fmt in list(DATE_FORMATS.items())[:4]:
        print(f"   {name}: {now.strftime(fmt)}")
    
    print("\n💱 Currencies:")
    for curr, config in CURRENCY_FORMATS.items():
        print(f"   {curr}: {config['symbol']}")
    
    print("\n" + "="*60)
    print("✅ All constants loaded successfully!")
    print("="*60)
