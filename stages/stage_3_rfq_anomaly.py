"""
Stage 3: RFQ Price & Specification Anomaly Detection
====================================================

Detects anomalies in RFQ/RFP responses:
- Price deviations (too high or suspiciously low)
- Specification mismatches
- Delivery timeline deviations
- Payment term violations

Uses: Rule Engine, Statistical Engine for anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

# Setup paths for imports
import sys
import os

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
from rule_engine import RuleEngine
from statistical_engine import StatisticalEngine
from formatters import format_currency, format_percentage
from logger import get_logger


class RFQAnomalyDetectionStage(BaseStage):
    """
    Detects pricing and specification anomalies in RFQ responses.
    """
    
    def __init__(self):
        """
        Initializes Stage 3 with anomaly detection engines.
        """
        super().__init__(stage_number=3)
        self.rule_engine = RuleEngine()
        self.stat_engine = StatisticalEngine()
        self.logger = get_logger(__name__)
        
        # Anomaly thresholds
        self.price_high_threshold = 30  # 30% above average
        self.price_low_threshold = 20   # 20% below average
        self.delivery_threshold = 15    # 15% deviation
    
    def _get_required_columns(self):
        """
        Returns required columns for RFQ anomaly detection.
        """
        return ['vendor_id', 'quoted_price']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes RFQ anomaly detection workflow.
        
        Args:
            data: DataFrame with RFQ response data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing detected anomalies
        """
        # Calculate statistics
        price_stats = self._calculate_price_statistics(data)
        
        # Detect price anomalies
        price_anomalies = self._detect_price_anomalies(data, price_stats)
        
        # Detect delivery anomalies
        delivery_anomalies = self._detect_delivery_anomalies(data)
        
        # Detect specification mismatches
        spec_anomalies = self._detect_specification_anomalies(data)
        
        # Combine all anomalies
        combined = self._combine_anomalies(
            price_anomalies, 
            delivery_anomalies, 
            spec_anomalies
        )
        
        # Classify severity
        classified = self._classify_severity(combined)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(classified)
        
        results = {
            'total_bids': len(data),
            'anomalies_detected': len(classified[classified['has_anomaly']]),
            'high_severity': len(classified[classified['severity'] == 'high']),
            'medium_severity': len(classified[classified['severity'] == 'medium']),
            'low_severity': len(classified[classified['severity'] == 'low']),
            'price_statistics': price_stats,
            'anomalies': classified.to_dict('records'),
            'recommendations': recommendations
        }
        
        self.logger.info(f"   Anomalies: {results['anomalies_detected']}/{results['total_bids']}")
        self.logger.info(f"   High Severity: {results['high_severity']}")
        
        return results
    
    def _calculate_price_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates statistical measures for quoted prices.
        
        Args:
            data: RFQ response DataFrame
            
        Returns:
            Dictionary with price statistics
        """
        prices = data['quoted_price'].values
        
        stats = {
            'mean': np.mean(prices),
            'median': np.median(prices),
            'std': np.std(prices),
            'min': np.min(prices),
            'max': np.max(prices),
            'q1': np.percentile(prices, 25),
            'q3': np.percentile(prices, 75)
        }
        
        # Calculate IQR for outlier detection
        stats['iqr'] = stats['q3'] - stats['q1']
        stats['lower_bound'] = stats['q1'] - (1.5 * stats['iqr'])
        stats['upper_bound'] = stats['q3'] + (1.5 * stats['iqr'])
        
        return stats
    
    def _detect_price_anomalies(self, data: pd.DataFrame, stats: Dict[str, float]) -> pd.DataFrame:
        """
        Detects price anomalies using statistical methods.
        
        Args:
            data: RFQ response DataFrame
            stats: Price statistics
            
        Returns:
            DataFrame with price anomaly flags
        """
        df = data.copy()
        
        # Calculate deviation from mean
        df['price_deviation_pct'] = ((df['quoted_price'] - stats['mean']) / stats['mean'] * 100)
        
        # Flag high prices
        df['price_too_high'] = df['price_deviation_pct'] > self.price_high_threshold
        
        # Flag suspiciously low prices
        df['price_too_low'] = df['price_deviation_pct'] < -self.price_low_threshold
        
        # Flag statistical outliers (IQR method)
        df['is_outlier'] = (
            (df['quoted_price'] < stats['lower_bound']) | 
            (df['quoted_price'] > stats['upper_bound'])
        )
        
        # Calculate Z-score
        df['z_score'] = (df['quoted_price'] - stats['mean']) / stats['std']
        df['extreme_outlier'] = np.abs(df['z_score']) > 3
        
        return df
    
    def _detect_delivery_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detects delivery timeline anomalies.
        
        Args:
            data: RFQ response DataFrame
            
        Returns:
            DataFrame with delivery anomaly flags
        """
        df = data.copy()
        
        # Check if delivery_days column exists
        if 'delivery_days' not in df.columns:
            df['delivery_anomaly'] = False
            return df
        
        # Calculate average delivery time
        avg_delivery = df['delivery_days'].mean()
        
        # Calculate deviation
        df['delivery_deviation_pct'] = (
            (df['delivery_days'] - avg_delivery) / avg_delivery * 100
        )
        
        # Flag deviations
        df['delivery_anomaly'] = (
            np.abs(df['delivery_deviation_pct']) > self.delivery_threshold
        )
        
        return df
    
    def _detect_specification_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detects specification mismatches.
        
        Args:
            data: RFQ response DataFrame
            
        Returns:
            DataFrame with specification anomaly flags
        """
        df = data.copy()
        
        # Initialize specification flags
        df['spec_mismatch'] = False
        
        # Check payment terms mismatch
        if 'payment_terms' in df.columns and 'required_payment_terms' in df.columns:
            df['spec_mismatch'] = df['payment_terms'] != df['required_payment_terms']
        
        # Check quantity mismatch
        if 'quoted_quantity' in df.columns and 'required_quantity' in df.columns:
            df['quantity_mismatch'] = df['quoted_quantity'] != df['required_quantity']
            df['spec_mismatch'] = df['spec_mismatch'] | df['quantity_mismatch']
        
        return df
    
    def _combine_anomalies(self, price_df: pd.DataFrame, delivery_df: pd.DataFrame, 
                          spec_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines all anomaly types into single DataFrame.
        
        Args:
            price_df: Price anomaly results
            delivery_df: Delivery anomaly results
            spec_df: Specification anomaly results
            
        Returns:
            Combined DataFrame with all anomaly flags
        """
        # Merge all dataframes
        combined = price_df.copy()
        
        # Add delivery anomalies if present
        if 'delivery_anomaly' in delivery_df.columns:
            combined['delivery_anomaly'] = delivery_df['delivery_anomaly']
        else:
            combined['delivery_anomaly'] = False
        
        # Add specification anomalies
        combined['spec_mismatch'] = spec_df['spec_mismatch']
        
        # Create master anomaly flag
        combined['has_anomaly'] = (
            combined['price_too_high'] | 
            combined['price_too_low'] | 
            combined['is_outlier'] |
            combined['delivery_anomaly'] |
            combined['spec_mismatch']
        )
        
        # Count total anomalies per bid
        combined['anomaly_count'] = (
            combined['price_too_high'].astype(int) +
            combined['price_too_low'].astype(int) +
            combined['delivery_anomaly'].astype(int) +
            combined['spec_mismatch'].astype(int)
        )
        
        return combined
    
    def _classify_severity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies anomaly severity levels.
        
        Args:
            data: DataFrame with anomaly flags
            
        Returns:
            DataFrame with severity classification
        """
        df = data.copy()
        
        # Initialize severity
        df['severity'] = 'none'
        
        # Low severity: 1 anomaly, not extreme
        df.loc[
            (df['anomaly_count'] == 1) & (~df['extreme_outlier']),
            'severity'
        ] = 'low'
        
        # Medium severity: 2 anomalies or 1 extreme
        df.loc[
            (df['anomaly_count'] == 2) | (df['extreme_outlier']),
            'severity'
        ] = 'medium'
        
        # High severity: 3+ anomalies
        df.loc[df['anomaly_count'] >= 3, 'severity'] = 'high'
        
        # Add recommendation flags
        df['requires_review'] = df['severity'].isin(['medium', 'high'])
        df['reject_bid'] = df['severity'] == 'high'
        
        return df
    
    def _generate_recommendations(self, data: pd.DataFrame) -> list:
        """
        Generates actionable recommendations.
        
        Args:
            data: DataFrame with classified anomalies
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # High severity bids
        high_severity = data[data['severity'] == 'high']
        if len(high_severity) > 0:
            recommendations.append({
                'type': 'reject',
                'count': len(high_severity),
                'vendors': high_severity['vendor_id'].tolist(),
                'message': f"🚨 REJECT {len(high_severity)} bids with HIGH severity anomalies"
            })
        
        # Medium severity bids
        medium_severity = data[data['severity'] == 'medium']
        if len(medium_severity) > 0:
            recommendations.append({
                'type': 'review',
                'count': len(medium_severity),
                'vendors': medium_severity['vendor_id'].tolist(),
                'message': f"⚠️  REVIEW {len(medium_severity)} bids with MEDIUM severity - seek clarification"
            })
        
        # Suspiciously low prices
        low_prices = data[data['price_too_low']]
        if len(low_prices) > 0:
            recommendations.append({
                'type': 'investigate',
                'count': len(low_prices),
                'vendors': low_prices['vendor_id'].tolist(),
                'message': f"🔍 INVESTIGATE {len(low_prices)} bids with suspiciously low prices"
            })
        
        # Clean bids
        clean_bids = data[~data['has_anomaly']]
        if len(clean_bids) > 0:
            recommendations.append({
                'type': 'approve',
                'count': len(clean_bids),
                'vendors': clean_bids['vendor_id'].tolist(),
                'message': f"✅ {len(clean_bids)} clean bids ready for evaluation"
            })
        
        return recommendations


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING STAGE 3: RFQ ANOMALY DETECTION")
    print("="*60)
    
    # Create sample RFQ response data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'vendor_id': [f'V{i:03d}' for i in range(1, 16)],
        'vendor_name': [f'Vendor {i}' for i in range(1, 16)],
        'quoted_price': [
            250000,  # Normal
            245000,  # Normal
            260000,  # Normal
            350000,  # Too high (40% above mean)
            180000,  # Too low (30% below mean)
            255000,  # Normal
            248000,  # Normal
            252000,  # Normal
            400000,  # Too high
            150000,  # Too low
            258000,  # Normal
            246000,  # Normal
            265000,  # Normal
            251000,  # Normal
            253000   # Normal
        ],
        'delivery_days': [30, 28, 32, 45, 25, 30, 31, 29, 50, 22, 30, 29, 31, 30, 32],
        'payment_terms': ['30 days'] * 15,
        'required_payment_terms': ['30 days'] * 15
    })
    
    # Add some specification mismatches
    sample_data.loc[3, 'payment_terms'] = '60 days'  # Mismatch
    sample_data.loc[8, 'payment_terms'] = '45 days'  # Mismatch
    
    # Initialize and execute stage
    stage = RFQAnomalyDetectionStage()
    result = stage.execute(sample_data)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Bids: {result['results']['total_bids']}")
    print(f"   Anomalies Detected: {result['results']['anomalies_detected']}")
    print(f"   High Severity: {result['results']['high_severity']}")
    print(f"   Medium Severity: {result['results']['medium_severity']}")
    print(f"   Low Severity: {result['results']['low_severity']}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    # Price statistics
    stats = result['results']['price_statistics']
    print(f"\n💰 PRICE STATISTICS:")
    print(f"   Average: {format_currency(stats['mean'])}")
    print(f"   Median: {format_currency(stats['median'])}")
    print(f"   Range: {format_currency(stats['min'])} - {format_currency(stats['max'])}")
    print(f"   Std Dev: {format_currency(stats['std'])}")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in result['results']['recommendations']:
        print(f"   {rec['message']}")
    
    # Anomalies detail
    anomalies = [a for a in result['results']['anomalies'] if a['has_anomaly']]
    if anomalies:
        print(f"\n🚨 DETECTED ANOMALIES:")
        for anomaly in anomalies[:5]:  # Show first 5
            print(f"\n   Vendor: {anomaly['vendor_id']}")
            print(f"   Price: {format_currency(anomaly['quoted_price'])}")
            print(f"   Deviation: {anomaly['price_deviation_pct']:.1f}%")
            print(f"   Severity: {anomaly['severity'].upper()}")
            flags = []
            if anomaly['price_too_high']:
                flags.append('Price Too High')
            if anomaly['price_too_low']:
                flags.append('Price Too Low')
            if anomaly['delivery_anomaly']:
                flags.append('Delivery Deviation')
            if anomaly['spec_mismatch']:
                flags.append('Spec Mismatch')
            print(f"   Flags: {', '.join(flags)}")
    
    print("\n" + "="*60)
    print("✅ STAGE 3 TEST COMPLETE")
    print("="*60)
