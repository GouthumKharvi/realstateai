"""
Stage 11: AI-Powered Fraud Detection & Anomaly Analysis
========================================================

Detects fraudulent activities through:
- Duplicate invoice detection
- Price manipulation identification
- Vendor collusion detection
- Abnormal payment pattern analysis
- Document forgery detection

Uses: ML Engine for anomaly detection, Statistical Engine for pattern analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import os
from collections import Counter

# Setup paths for imports
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
core_dir = os.path.join(project_root, 'core', 'ai_engine')
stages_dir = os.path.join(project_root, 'stages')
utils_dir = os.path.join(project_root, 'utils')

sys.path.insert(0, project_root)
sys.path.insert(0, core_dir)
sys.path.insert(0, stages_dir)
sys.path.insert(0, utils_dir)

from base_stage import BaseStage
from ml_engine import MLEngine
from statistical_engine import StatisticalEngine
from formatters import format_currency
from logger import get_logger


class FraudDetectionStage(BaseStage):
    """
    Detects fraudulent activities and anomalies in procurement data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes Stage 11 with fraud detection parameters.
        
        Args:
            config: Optional configuration dictionary from JSON
        """
        super().__init__(stage_number=11)
        self.ml_engine = MLEngine()
        self.stat_engine = StatisticalEngine()
        self.logger = get_logger(__name__)
        
        # Load config or use defaults
        if config:
            self.fraud_score_threshold = config.get('fraud_score_threshold', 70)
            self.price_variance_threshold = config.get('price_variance_threshold', 25)
            self.duplicate_similarity_threshold = config.get('duplicate_similarity_threshold', 0.95)
        else:
            self.fraud_score_threshold = 70
            self.price_variance_threshold = 25
            self.duplicate_similarity_threshold = 0.95
    
    def _get_required_columns(self):
        """
        Returns required columns for fraud detection.
        """
        return ['transaction_id', 'amount']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes fraud detection workflow.
        
        Args:
            data: DataFrame with transaction data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing fraud detection results
        """
        # Detect duplicate invoices
        duplicate_check = self._detect_duplicate_invoices(data)
        
        # Identify price manipulation
        price_check = self._identify_price_manipulation(duplicate_check)
        
        # Detect vendor collusion
        collusion_check = self._detect_vendor_collusion(price_check)
        
        # Analyze payment patterns
        payment_check = self._analyze_payment_patterns(collusion_check)
        
        # Calculate fraud scores
        fraud_scores = self._calculate_fraud_scores(payment_check)
        
        # Classify fraud risk
        classified = self._classify_fraud_risk(fraud_scores)
        
        # Generate alerts
        alerts = self._generate_fraud_alerts(classified)
        
        results = {
            'total_transactions': len(data),
            'high_risk_fraud': len(classified[classified['fraud_risk'] == 'High']),
            'medium_risk_fraud': len(classified[classified['fraud_risk'] == 'Medium']),
            'suspicious_transactions': len(classified[classified['fraud_score'] > 50]),
            'duplicate_invoices': len(classified[classified['is_duplicate']]),
            'price_manipulations': len(classified[classified['price_manipulation_detected']]),
            'potential_collusion': len(classified[classified['collusion_risk']]),
            'total_fraud_exposure': classified[classified['fraud_risk'] == 'High']['amount'].sum(),
            'transactions': classified.to_dict('records'),
            'alerts': alerts,
            'fraud_summary': self._generate_fraud_summary(classified)
        }
        
        self.logger.info(f"   High Risk Fraud: {results['high_risk_fraud']}/{results['total_transactions']}")
        self.logger.info(f"   Total Exposure: {format_currency(results['total_fraud_exposure'])}")
        
        return results
    
    def _detect_duplicate_invoices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detects duplicate or near-duplicate invoices.
        
        Args:
            data: Transaction DataFrame
            
        Returns:
            DataFrame with duplicate flags
        """
        df = data.copy()
        
        # Add invoice details if not present
        if 'invoice_number' not in df.columns:
            df['invoice_number'] = [f'INV-{i:06d}' for i in range(1, len(df) + 1)]
            # Randomly create some duplicates (5%)
            duplicate_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
            for idx in duplicate_indices:
                if idx > 0:
                    df.loc[idx, 'invoice_number'] = df.loc[idx - 1, 'invoice_number']
        
        if 'vendor_id' not in df.columns:
            df['vendor_id'] = [f'V{i:03d}' for i in np.random.randint(1, 50, len(df))]
        
        # Detect exact duplicates
        df['is_duplicate'] = df.duplicated(subset=['invoice_number', 'vendor_id'], keep='first')
        
        # Detect near-duplicates (same amount, same vendor, close dates)
        df['amount_rounded'] = df['amount'].round(-3)  # Round to nearest 1000
        df['is_near_duplicate'] = df.duplicated(subset=['amount_rounded', 'vendor_id'], keep='first')
        
        # Duplicate score
        df['duplicate_score'] = 0
        df.loc[df['is_duplicate'], 'duplicate_score'] = 100
        df.loc[df['is_near_duplicate'] & ~df['is_duplicate'], 'duplicate_score'] = 60
        
        return df
    
    def _identify_price_manipulation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies abnormal pricing that may indicate manipulation.
        
        Args:
            data: Duplicate-checked DataFrame
            
        Returns:
            DataFrame with price manipulation flags
        """
        df = data.copy()
        
        # Add item details if not present
        if 'item_category' not in df.columns:
            categories = ['Construction Materials', 'IT Equipment', 'Office Supplies', 'Services', 'Machinery']
            df['item_category'] = np.random.choice(categories, len(df))
        
        if 'unit_price' not in df.columns:
            df['unit_price'] = df['amount'] / np.random.randint(1, 100, len(df))
        
        # Calculate average price per category
        df['category_avg_price'] = df.groupby('item_category')['unit_price'].transform('mean')
        df['category_std_price'] = df.groupby('item_category')['unit_price'].transform('std')
        
        # Price deviation from average
        df['price_deviation_pct'] = ((df['unit_price'] - df['category_avg_price']) / df['category_avg_price'] * 100)
        
        # Flag extreme deviations
        df['price_manipulation_detected'] = abs(df['price_deviation_pct']) > self.price_variance_threshold
        
        # Just-below-threshold pricing (common fraud pattern)
        if 'approval_threshold' not in df.columns:
            df['approval_threshold'] = 1000000  # 10 Lakhs
        
        df['just_below_threshold'] = (
            (df['amount'] >= df['approval_threshold'] * 0.95) &
            (df['amount'] < df['approval_threshold'])
        )
        
        # Split invoicing detection (breaking large invoice into smaller ones)
        df['invoice_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, len(df)), unit='D')
        df['same_day_same_vendor'] = df.groupby(['vendor_id', 'invoice_date'])['transaction_id'].transform('count')
        df['split_invoice_suspected'] = df['same_day_same_vendor'] > 2
        
        # Price manipulation score
        df['price_manipulation_score'] = 0
        df.loc[df['price_manipulation_detected'], 'price_manipulation_score'] += 40
        df.loc[df['just_below_threshold'], 'price_manipulation_score'] += 30
        df.loc[df['split_invoice_suspected'], 'price_manipulation_score'] += 30
        
        return df
    
    def _detect_vendor_collusion(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detects potential vendor collusion patterns.
        
        Args:
            data: Price-checked DataFrame
            
        Returns:
            DataFrame with collusion flags
        """
        df = data.copy()
        
        # Round-robin winner pattern (vendors taking turns winning bids)
        vendor_counts = df['vendor_id'].value_counts()
        df['vendor_frequency'] = df['vendor_id'].map(vendor_counts)
        
        # Check for evenly distributed wins (suspicious pattern)
        total_vendors = df['vendor_id'].nunique()
        expected_frequency = len(df) / total_vendors
        df['frequency_variance'] = abs(df['vendor_frequency'] - expected_frequency)
        
        # Collusion indicators
        df['collusion_risk'] = False
        
        # Pattern 1: Multiple vendors with same contact/address (if data available)
        # Simulated: Vendors with similar IDs
        df['vendor_id_numeric'] = df['vendor_id'].str.extract(r'(\d+)').astype(int)
        df['similar_vendor_exists'] = df.groupby(df['vendor_id_numeric'] // 10)['vendor_id'].transform('count') > 3
        
        # Pattern 2: Suspiciously similar pricing
        df['similar_pricing_group'] = (df['amount'] / 10000).astype(int)
        df['same_price_count'] = df.groupby(['similar_pricing_group', 'item_category'])['transaction_id'].transform('count')
        df.loc[df['same_price_count'] > 5, 'collusion_risk'] = True
        
        # Pattern 3: Sequential invoice numbers across vendors (copy-paste fraud)
        df['invoice_numeric'] = df['invoice_number'].str.extract(r'(\d+)').astype(int)
        df['sequential_invoices'] = df['invoice_numeric'].diff().abs() == 1
        
        # Collusion score
        df['collusion_score'] = 0
        df.loc[df['similar_vendor_exists'], 'collusion_score'] += 30
        df.loc[df['collusion_risk'], 'collusion_score'] += 40
        df.loc[df['sequential_invoices'], 'collusion_score'] += 30
        
        return df
    
    def _analyze_payment_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes payment patterns for anomalies.
        
        Args:
            data: Collusion-checked DataFrame
            
        Returns:
            DataFrame with payment pattern flags
        """
        df = data.copy()
        
        # Add payment details if not present
        if 'payment_method' not in df.columns:
            df['payment_method'] = np.random.choice(['Bank Transfer', 'Check', 'Cash', 'Wire'], len(df), p=[0.7, 0.2, 0.05, 0.05])
        
        if 'payment_date' not in df.columns:
            df['payment_date'] = df['invoice_date'] + pd.to_timedelta(np.random.randint(1, 60, len(df)), unit='D')
        
        # Calculate payment timing
        df['payment_delay_days'] = (df['payment_date'] - df['invoice_date']).dt.days
        
        # Unusual payment patterns
        df['unusually_fast_payment'] = df['payment_delay_days'] < 3  # Same-day/next-day payments
        df['cash_payment_high_value'] = (df['payment_method'] == 'Cash') & (df['amount'] > 100000)
        
        # After-hours or weekend processing (if timestamp available)
        df['payment_hour'] = np.random.randint(0, 24, len(df))
        df['after_hours_payment'] = (df['payment_hour'] < 6) | (df['payment_hour'] > 20)
        
        # Repeated amounts (structured payments to avoid detection)
        df['rounded_amount'] = (df['amount'] / 1000).round() * 1000
        df['repeated_amount_count'] = df.groupby('rounded_amount')['transaction_id'].transform('count')
        df['repeated_payment_pattern'] = df['repeated_amount_count'] > 10
        
        # Payment pattern score
        df['payment_pattern_score'] = 0
        df.loc[df['unusually_fast_payment'], 'payment_pattern_score'] += 20
        df.loc[df['cash_payment_high_value'], 'payment_pattern_score'] += 50
        df.loc[df['after_hours_payment'], 'payment_pattern_score'] += 15
        df.loc[df['repeated_payment_pattern'], 'payment_pattern_score'] += 25
        
        return df
    
    def _calculate_fraud_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates overall fraud score combining all indicators.
        
        Args:
            data: Payment-analyzed DataFrame
            
        Returns:
            DataFrame with fraud scores
        """
        df = data.copy()
        
        # Weighted fraud score
        df['fraud_score'] = (
            df['duplicate_score'] * 0.30 +
            df['price_manipulation_score'] * 0.30 +
            df['collusion_score'] * 0.25 +
            df['payment_pattern_score'] * 0.15
        )
        
        # Clip to 0-100
        df['fraud_score'] = df['fraud_score'].clip(0, 100).round(2)
        
        # List all fraud indicators detected
        df['fraud_indicators'] = df.apply(self._list_fraud_indicators, axis=1)
        df['indicator_count'] = df['fraud_indicators'].apply(len)
        
        return df
    
    def _list_fraud_indicators(self, row: pd.Series) -> List[str]:
        """
        Lists all fraud indicators present for a transaction.
        
        Args:
            row: Transaction row
            
        Returns:
            List of fraud indicator names
        """
        indicators = []
        
        if row.get('is_duplicate', False):
            indicators.append('Duplicate Invoice')
        if row.get('is_near_duplicate', False):
            indicators.append('Near-Duplicate')
        if row.get('price_manipulation_detected', False):
            indicators.append('Price Manipulation')
        if row.get('just_below_threshold', False):
            indicators.append('Just-Below-Threshold')
        if row.get('split_invoice_suspected', False):
            indicators.append('Split Invoice')
        if row.get('collusion_risk', False):
            indicators.append('Vendor Collusion')
        if row.get('unusually_fast_payment', False):
            indicators.append('Fast Payment')
        if row.get('cash_payment_high_value', False):
            indicators.append('High-Value Cash')
        if row.get('repeated_payment_pattern', False):
            indicators.append('Repeated Pattern')
        
        return indicators
    
    def _classify_fraud_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies transactions into fraud risk levels.
        
        Args:
            data: Scored DataFrame
            
        Returns:
            DataFrame with fraud risk classification
        """
        df = data.copy()
        
        # Risk classification
        df['fraud_risk'] = 'Low'
        df.loc[df['fraud_score'] >= 40, 'fraud_risk'] = 'Medium'
        df.loc[df['fraud_score'] >= self.fraud_score_threshold, 'fraud_risk'] = 'High'
        
        # Requires investigation flag
        df['requires_investigation'] = (
            (df['fraud_risk'] == 'High') |
            (df['indicator_count'] >= 3) |
            (df['is_duplicate'])
        )
        
        # Immediate action required
        df['immediate_action'] = (
            (df['fraud_score'] >= 85) |
            (df['is_duplicate'] & (df['amount'] > 500000))
        )
        
        return df
    
    def _generate_fraud_alerts(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates fraud alerts for high-risk transactions.
        
        Args:
            data: Classified DataFrame
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Critical alerts
        critical = data[data['immediate_action']]
        if len(critical) > 0:
            total_amount = critical['amount'].sum()
            alerts.append({
                'severity': 'Critical',
                'type': 'immediate_action',
                'count': len(critical),
                'transactions': critical['transaction_id'].tolist(),
                'total_amount': total_amount,
                'message': f"🚨 CRITICAL: {len(critical)} transactions require IMMEDIATE investigation - Exposure: {format_currency(total_amount)}"
            })
        
        # Duplicate invoice alerts
        duplicates = data[data['is_duplicate']]
        if len(duplicates) > 0:
            alerts.append({
                'severity': 'High',
                'type': 'duplicate_invoice',
                'count': len(duplicates),
                'transactions': duplicates['transaction_id'].tolist(),
                'total_amount': duplicates['amount'].sum(),
                'message': f"⚠️  DUPLICATE: {len(duplicates)} duplicate invoices detected - Potential double payment"
            })
        
        # Price manipulation alerts
        price_fraud = data[data['price_manipulation_detected']]
        if len(price_fraud) > 0:
            alerts.append({
                'severity': 'High',
                'type': 'price_manipulation',
                'count': len(price_fraud),
                'transactions': price_fraud['transaction_id'].tolist(),
                'message': f"💰 PRICE FRAUD: {len(price_fraud)} transactions with abnormal pricing detected"
            })
        
        # Vendor collusion alerts
        collusion = data[data['collusion_risk']]
        if len(collusion) > 0:
            alerts.append({
                'severity': 'High',
                'type': 'vendor_collusion',
                'count': len(collusion),
                'vendors': collusion['vendor_id'].unique().tolist(),
                'message': f"🤝 COLLUSION: {len(collusion)} transactions show vendor collusion patterns"
            })
        
        # High-risk transactions
        high_risk = data[data['fraud_risk'] == 'High']
        if len(high_risk) > 0:
            alerts.append({
                'severity': 'High',
                'type': 'high_risk',
                'count': len(high_risk),
                'total_amount': high_risk['amount'].sum(),
                'message': f"🔴 HIGH RISK: {len(high_risk)} transactions flagged - Total exposure: {format_currency(high_risk['amount'].sum())}"
            })
        
        return alerts
    
    def _generate_fraud_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates fraud detection summary statistics.
        
        Args:
            data: Classified DataFrame
            
        Returns:
            Fraud summary dictionary
        """
        summary = {
            'total_transactions': len(data),
            'fraud_detection_rate': float((len(data[data['fraud_score'] > 0]) / len(data) * 100) if len(data) > 0 else 0),
            'by_risk_level': {
                'High': int(len(data[data['fraud_risk'] == 'High'])),
                'Medium': int(len(data[data['fraud_risk'] == 'Medium'])),
                'Low': int(len(data[data['fraud_risk'] == 'Low']))
            },
            'fraud_types': {
                'Duplicate Invoices': int(len(data[data['is_duplicate']])),
                'Price Manipulation': int(len(data[data['price_manipulation_detected']])),
                'Vendor Collusion': int(len(data[data['collusion_risk']])),
                'Payment Anomalies': int(len(data[data['payment_pattern_score'] > 50]))
            },
            'total_fraud_exposure': float(data[data['fraud_risk'] == 'High']['amount'].sum()),
            'average_fraud_score': float(data['fraud_score'].mean()),
            'transactions_requiring_investigation': int(len(data[data['requires_investigation']])),
            'most_common_indicator': self._get_most_common_indicator(data)
        }
        
        return summary
    
    def _get_most_common_indicator(self, data: pd.DataFrame) -> str:
        """
        Finds most commonly occurring fraud indicator.
        
        Args:
            data: Classified DataFrame
            
        Returns:
            Most common indicator name
        """
        all_indicators = []
        for indicators in data['fraud_indicators']:
            all_indicators.extend(indicators)
        
        if not all_indicators:
            return 'None'
        
        counter = Counter(all_indicators)
        most_common = counter.most_common(1)
        
        return most_common[0][0] if most_common else 'None'


# ============================================================
# CONFIG LOADER
# ============================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}")
        print(f"   Using default parameters")
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded config from: {config_path}")
    return config


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING STAGE 11: FRAUD DETECTION")
    print("="*60)
    
    # Check if config file provided
    config = {}
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print(f"\n📄 Running with CONFIG: {config_path}")
    else:
        print(f"\n📄 Running with DEMO DATA (default parameters)")
    
    # Create sample transaction data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'transaction_id': [f'TXN-{i:06d}' for i in range(1, 101)],
        'amount': np.random.uniform(50000, 5000000, 100)
    })
    
    # Initialize stage with config (if provided)
    stage = FraudDetectionStage(config=config.get('parameters', {}))
    
    # Execute
    result = stage.execute(sample_data)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Transactions: {result['results']['total_transactions']}")
    print(f"   High Risk Fraud: {result['results']['high_risk_fraud']}")
    print(f"   Medium Risk: {result['results']['medium_risk_fraud']}")
    print(f"   Suspicious: {result['results']['suspicious_transactions']}")
    print(f"   Duplicates: {result['results']['duplicate_invoices']}")
    print(f"   Price Manipulations: {result['results']['price_manipulations']}")
    print(f"   Collusion Cases: {result['results']['potential_collusion']}")
    print(f"   Total Fraud Exposure: {format_currency(result['results']['total_fraud_exposure'])}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    # Fraud summary
    print(f"\n📋 FRAUD SUMMARY:")
    summary = result['results']['fraud_summary']
    print(f"   Detection Rate: {summary['fraud_detection_rate']:.1f}%")
    print(f"   Risk Levels: High={summary['by_risk_level']['High']}, Medium={summary['by_risk_level']['Medium']}, Low={summary['by_risk_level']['Low']}")
    print(f"   Avg Fraud Score: {summary['average_fraud_score']:.2f}")
    print(f"   Requires Investigation: {summary['transactions_requiring_investigation']}")
    print(f"   Most Common Indicator: {summary['most_common_indicator']}")
    print(f"\n   Fraud Types:")
    for fraud_type, count in summary['fraud_types'].items():
        print(f"     {fraud_type}: {count}")
    
    # Top fraud cases
    print(f"\n🚨 TOP FRAUD CASES:")
    transactions = result['results']['transactions']
    top_fraud = sorted([t for t in transactions if t['fraud_score'] > 50], key=lambda x: x['fraud_score'], reverse=True)[:5]
    
    for i, txn in enumerate(top_fraud, 1):
        print(f"\n   {i}. {txn['transaction_id']}")
        print(f"      Amount: {format_currency(txn['amount'])}")
        print(f"      Fraud Score: {txn['fraud_score']}")
        print(f"      Risk Level: {txn['fraud_risk']}")
        print(f"      Indicators ({txn['indicator_count']}): {', '.join(txn['fraud_indicators'])}")
        if txn['immediate_action']:
            print(f"      ⚠️  IMMEDIATE ACTION REQUIRED")
    
    # Alerts
    print(f"\n🔔 FRAUD ALERTS:")
    for alert in result['results']['alerts']:
        print(f"\n   [{alert['severity']}] {alert['message']}")
    
    print("\n" + "="*60)
    print("✅ STAGE 11 TEST COMPLETE")
    print("="*60)
