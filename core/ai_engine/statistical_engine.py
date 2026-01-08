"""
Statistical Engine for Procurement AI
Detects outliers using Z-Score and IQR methods
Calculates statistical metrics for RFQ/vendor analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class StatisticalEngine:
    """
    Statistical analysis engine for anomaly detection in procurement data
    Uses Z-score and IQR methods to identify outliers
    """
    
    def __init__(self, zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        """
        Initialize StatisticalEngine
        
        Args:
            zscore_threshold: Number of standard deviations for Z-score method (default: 3.0)
            iqr_multiplier: Multiplier for IQR range (default: 1.5)
        """
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
    
    
    def calculate_statistics(self, data: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a data series
        
        Args:
            data: Pandas Series of numerical values
            
        Returns:
            Dictionary with statistical metrics
        """
        # Remove NaN values
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'std': None,
                'variance': None,
                'min': None,
                'max': None,
                'q1': None,
                'q3': None,
                'iqr': None,
                'range': None
            }
        
        # Calculate statistics
        stats = {
            'count': len(clean_data),
            'mean': float(clean_data.mean()),
            'median': float(clean_data.median()),
            'std': float(clean_data.std()) if len(clean_data) > 1 else 0.0,  
            'variance': float(clean_data.var()) if len(clean_data) > 1 else 0.0,  
            'min': float(clean_data.min()),
            'max': float(clean_data.max()),
            'q1': float(clean_data.quantile(0.25)),
            'q3': float(clean_data.quantile(0.75)),
        }

        
        # Calculate IQR and range
        stats['iqr'] = stats['q3'] - stats['q1']
        stats['range'] = stats['max'] - stats['min']
        
        return stats
    
    
    def detect_outliers_zscore(self, data: pd.Series, column_name: str = 'value') -> pd.DataFrame:
        """
        Detect outliers using Z-score method
        
        Z-score = (value - mean) / std
        Outlier if |Z-score| > threshold
        
        Args:
            data: Pandas Series of numerical values
            column_name: Name for the value column in output
            
        Returns:
            DataFrame with original values, Z-scores, and outlier flags
        """
        # Create DataFrame from series
        df = pd.DataFrame({column_name: data})
        
        # Calculate statistics
        mean_val = data.mean()
        std_val = data.std()
        
        # Handle case where std is 0 (all values same)
        if std_val == 0:
            df['zscore'] = 0.0
            df['is_outlier_zscore'] = False
            df['outlier_type'] = 'none'
            df['severity'] = 'none'
            return df
        
        # Calculate Z-scores
        df['zscore'] = (df[column_name] - mean_val) / std_val
        
        # Identify outliers
        df['is_outlier_zscore'] = abs(df['zscore']) > self.zscore_threshold
        
        # Classify outlier type (high or low)
        df['outlier_type'] = 'none'
        df.loc[df['zscore'] > self.zscore_threshold, 'outlier_type'] = 'high'
        df.loc[df['zscore'] < -self.zscore_threshold, 'outlier_type'] = 'low'
        
        # Severity classification
        df['severity'] = 'none'
        df.loc[abs(df['zscore']) > self.zscore_threshold, 'severity'] = 'medium'
        df.loc[abs(df['zscore']) > self.zscore_threshold * 1.5, 'severity'] = 'high'
        
        return df
    
    
    def detect_outliers_iqr(self, data: pd.Series, column_name: str = 'value') -> pd.DataFrame:
        """
        Detect outliers using IQR (Interquartile Range) method
        
        IQR = Q3 - Q1
        Lower bound = Q1 - (IQR * multiplier)
        Upper bound = Q3 + (IQR * multiplier)
        
        Args:
            data: Pandas Series of numerical values
            column_name: Name for the value column in output
            
        Returns:
            DataFrame with original values, bounds, and outlier flags
        """
        # Create DataFrame from series
        df = pd.DataFrame({column_name: data})
        
        # Calculate quartiles and IQR
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        # Calculate bounds
        lower_bound = q1 - (self.iqr_multiplier * iqr)
        upper_bound = q3 + (self.iqr_multiplier * iqr)
        
        # Add bounds to DataFrame
        df['iqr'] = iqr
        df['lower_bound'] = lower_bound
        df['upper_bound'] = upper_bound
        
        # Identify outliers
        df['is_outlier_iqr'] = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
        
        # Classify outlier type
        df['outlier_type'] = 'none'
        df.loc[df[column_name] < lower_bound, 'outlier_type'] = 'low'
        df.loc[df[column_name] > upper_bound, 'outlier_type'] = 'high'
        
        # Calculate severity based on distance from bounds
        df['distance_from_bound'] = 0.0
        df.loc[df[column_name] < lower_bound, 'distance_from_bound'] = \
            abs(df.loc[df[column_name] < lower_bound, column_name] - lower_bound)
        df.loc[df[column_name] > upper_bound, 'distance_from_bound'] = \
            abs(df.loc[df[column_name] > upper_bound, column_name] - upper_bound)
        
        # Severity classification
        df['severity'] = 'none'
        df.loc[df['is_outlier_iqr'], 'severity'] = 'medium'
        df.loc[df['distance_from_bound'] > iqr, 'severity'] = 'high'
        
        return df
    
    
    def detect_outliers_combined(self, data: pd.Series, column_name: str = 'value') -> pd.DataFrame:
        """
        Detect outliers using both Z-score and IQR methods
        Combines results for comprehensive outlier detection
        
        Args:
            data: Pandas Series of numerical values
            column_name: Name for the value column in output
            
        Returns:
            DataFrame with results from both methods
        """
        # Get Z-score results
        zscore_df = self.detect_outliers_zscore(data, column_name)
        
        # Get IQR results
        iqr_df = self.detect_outliers_iqr(data, column_name)
        
        # Combine results
        combined_df = pd.DataFrame({column_name: data})
        combined_df['zscore'] = zscore_df['zscore']
        combined_df['is_outlier_zscore'] = zscore_df['is_outlier_zscore']
        combined_df['is_outlier_iqr'] = iqr_df['is_outlier_iqr']
        
        # Outlier if flagged by either method
        combined_df['is_outlier'] = (
            combined_df['is_outlier_zscore'] | combined_df['is_outlier_iqr']
        )
        
        # Outlier type from Z-score (primary)
        combined_df['outlier_type'] = zscore_df['outlier_type']
        
        # Combined severity (take maximum)
        severity_map = {'none': 0, 'medium': 1, 'high': 2}
        zscore_severity_num = zscore_df['severity'].map(severity_map)
        iqr_severity_num = iqr_df['severity'].map(severity_map)
        
        max_severity = pd.concat([zscore_severity_num, iqr_severity_num], axis=1).max(axis=1)
        reverse_map = {0: 'none', 1: 'medium', 2: 'high'}
        combined_df['severity'] = max_severity.map(reverse_map)
        
        # Detection method
        combined_df['detection_method'] = 'none'
        combined_df.loc[combined_df['is_outlier_zscore'] & combined_df['is_outlier_iqr'], 
                       'detection_method'] = 'both'
        combined_df.loc[combined_df['is_outlier_zscore'] & ~combined_df['is_outlier_iqr'], 
                       'detection_method'] = 'zscore_only'
        combined_df.loc[~combined_df['is_outlier_zscore'] & combined_df['is_outlier_iqr'], 
                       'detection_method'] = 'iqr_only'
        
        return combined_df
    
    
    def analyze_rfq_prices(self, rfq_data: pd.DataFrame, 
                          price_column: str = 'quoted_price',
                          vendor_column: str = 'vendor_id') -> pd.DataFrame:
        """
        Analyze RFQ/RFP bid prices for anomalies
        
        Args:
            rfq_data: DataFrame with vendor bids
            price_column: Column name for quoted prices
            vendor_column: Column name for vendor IDs
            
        Returns:
            DataFrame with statistical analysis and anomaly flags
        """
        result = rfq_data.copy()
        
        # Get outlier analysis
        outlier_analysis = self.detect_outliers_combined(
            rfq_data[price_column], 
            column_name=price_column
        )
        
        # Add outlier columns to result
        result['is_price_outlier'] = outlier_analysis['is_outlier']
        result['outlier_type'] = outlier_analysis['outlier_type']
        result['outlier_severity'] = outlier_analysis['severity']
        result['detection_method'] = outlier_analysis['detection_method']
        result['zscore'] = outlier_analysis['zscore']
        
        # Calculate statistics
        stats = self.calculate_statistics(rfq_data[price_column])
        result['price_mean'] = stats['mean']
        result['price_median'] = stats['median']
        result['price_std'] = stats['std']
        
        # Deviation percentage from mean
        if stats['mean'] and stats['mean'] > 0:
            result['deviation_pct'] = (
                (result[price_column] - stats['mean']) / stats['mean'] * 100
            )
        else:
            result['deviation_pct'] = 0.0
        
        return result
    
    
    def analyze_vendor_performance(self, vendor_data: pd.DataFrame,
                                   metric_column: str = 'delivery_score') -> pd.DataFrame:
        """
        Analyze vendor performance metrics for outliers
        
        Args:
            vendor_data: DataFrame with vendor performance data
            metric_column: Column to analyze (e.g., 'delivery_score', 'quality_score')
            
        Returns:
            DataFrame with performance analysis
        """
        result = vendor_data.copy()
        
        # Detect outliers
        outlier_analysis = self.detect_outliers_combined(
            vendor_data[metric_column],
            column_name=metric_column
        )
        
        # Add analysis columns
        result['is_performance_outlier'] = outlier_analysis['is_outlier']
        result['outlier_type'] = outlier_analysis['outlier_type']
        result['severity'] = outlier_analysis['severity']
        result['zscore'] = outlier_analysis['zscore']
        
        # Calculate statistics
        stats = self.calculate_statistics(vendor_data[metric_column])
        result['metric_mean'] = stats['mean']
        result['metric_median'] = stats['median']
        
        return result


# Example usage
if __name__ == "__main__":
    print("Statistical Engine - Example Usage")
    print("=" * 60)
    
    # Initialize engine
    engine = StatisticalEngine(zscore_threshold=3.0, iqr_multiplier=1.5)
    
    # Example: RFQ price analysis
    print("\n📊 RFQ Price Analysis")
    rfq_data = pd.DataFrame({
        'vendor_id': ['V001', 'V002', 'V003', 'V004', 'V005', 'V006'],
        'quoted_price': [250000, 260000, 255000, 450000, 248000, 150000]
    })
    
    result = engine.analyze_rfq_prices(rfq_data)
    print(result[['vendor_id', 'quoted_price', 'is_price_outlier', 
                  'outlier_type', 'outlier_severity', 'deviation_pct']])
    
    # Calculate statistics
    print("\n📈 Statistics Summary")
    stats = engine.calculate_statistics(rfq_data['quoted_price'])
    for key, value in stats.items():
        print(f"   {key}: {value}")
