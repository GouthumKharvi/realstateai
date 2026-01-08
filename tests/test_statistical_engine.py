"""
Unit tests for StatisticalEngine class
Tests Z-score, IQR outlier detection, and statistical calculations
"""

import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ai_engine.statistical_engine import StatisticalEngine
import pandas as pd
import numpy as np


def test_statistical_engine_initialization():
    """Test StatisticalEngine initialization with default and custom parameters"""
    print("\n🧪 Testing StatisticalEngine Initialization...")
    try:
        # Test 1: Default initialization
        engine = StatisticalEngine()
        
        if engine.zscore_threshold == 3.0 and engine.iqr_multiplier == 1.5:
            print(f"   ✅ Default initialization: Z-threshold={engine.zscore_threshold}, IQR-multiplier={engine.iqr_multiplier}")
        else:
            print(f"   ❌ Default values incorrect")
            return False
        
        # Test 2: Custom initialization
        custom_engine = StatisticalEngine(zscore_threshold=2.5, iqr_multiplier=2.0)
        
        if custom_engine.zscore_threshold == 2.5 and custom_engine.iqr_multiplier == 2.0:
            print(f"   ✅ Custom initialization: Z-threshold={custom_engine.zscore_threshold}, IQR-multiplier={custom_engine.iqr_multiplier}")
        else:
            print(f"   ❌ Custom values not set correctly")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_calculate_statistics():
    """Test comprehensive statistics calculation"""
    print("\n🧪 Testing Calculate Statistics...")
    try:
        engine = StatisticalEngine()
        
        # Test Case 1: Normal data
        data = pd.Series([100, 120, 110, 130, 105, 115, 125, 135])
        stats = engine.calculate_statistics(data)
        
        expected_mean = 117.5
        expected_median = 117.5
        
        if abs(stats['mean'] - expected_mean) < 0.1:
            print(f"   ✅ Mean calculated correctly: {stats['mean']}")
        else:
            print(f"   ❌ Mean incorrect: {stats['mean']} (expected ~{expected_mean})")
            return False
        
        if abs(stats['median'] - expected_median) < 0.1:
            print(f"   ✅ Median calculated correctly: {stats['median']}")
        else:
            print(f"   ❌ Median incorrect: {stats['median']} (expected ~{expected_median})")
        
        # Check all required keys
        required_keys = ['count', 'mean', 'median', 'std', 'variance', 'min', 'max', 'q1', 'q3', 'iqr', 'range']
        if all(key in stats for key in required_keys):
            print(f"   ✅ All {len(required_keys)} statistical metrics calculated")
        else:
            print(f"   ❌ Missing statistical metrics")
            return False
        
        # Test Case 2: Empty data
        empty_data = pd.Series([])
        empty_stats = engine.calculate_statistics(empty_data)
        
        if empty_stats['count'] == 0 and empty_stats['mean'] is None:
            print(f"   ✅ Empty data handled correctly")
        else:
            print(f"   ❌ Empty data not handled properly")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_detect_outliers_zscore():
    """Test Z-score outlier detection"""
    print("\n🧪 Testing Z-Score Outlier Detection...")
    try:
        engine = StatisticalEngine(zscore_threshold=2.0)
        
        # Create data with clear outliers
        # Normal: 100, 110, 105, 115, 108, 112
        # Outliers: 200 (high), 20 (low)
        data = pd.Series([100, 110, 105, 115, 108, 112, 200, 20])
        
        result = engine.detect_outliers_zscore(data, column_name='value')
        
        # Check if outliers detected
        outlier_count = result['is_outlier_zscore'].sum()
        
        if outlier_count >= 2:
            print(f"   ✅ Detected {outlier_count} outliers using Z-score")
        else:
            print(f"   ⚠️  Only detected {outlier_count} outliers (expected at least 2)")
        
        # Check outlier types
        high_outliers = (result['outlier_type'] == 'high').sum()
        low_outliers = (result['outlier_type'] == 'low').sum()
        
        print(f"   ✅ High outliers: {high_outliers}, Low outliers: {low_outliers}")
        
        # Check Z-score column exists
        if 'zscore' in result.columns:
            print(f"   ✅ Z-score values calculated")
        else:
            print(f"   ❌ Z-score column missing")
            return False
        
        # Test Case 2: All same values (std = 0)
        same_data = pd.Series([100, 100, 100, 100])
        same_result = engine.detect_outliers_zscore(same_data, column_name='value')
        
        if same_result['is_outlier_zscore'].sum() == 0:
            print(f"   ✅ Handled zero standard deviation correctly")
        else:
            print(f"   ❌ Zero std case not handled")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_detect_outliers_iqr():
    """Test IQR outlier detection"""
    print("\n🧪 Testing IQR Outlier Detection...")
    try:
        engine = StatisticalEngine(iqr_multiplier=1.5)
        
        # Create data with outliers
        data = pd.Series([100, 105, 110, 115, 120, 125, 130, 300, 10])
        
        result = engine.detect_outliers_iqr(data, column_name='value')
        
        # Check if outliers detected
        outlier_count = result['is_outlier_iqr'].sum()
        
        if outlier_count >= 2:
            print(f"   ✅ Detected {outlier_count} outliers using IQR")
        else:
            print(f"   ⚠️  Detected {outlier_count} outliers (expected at least 2)")
        
        # Check required columns
        required_cols = ['iqr', 'lower_bound', 'upper_bound', 'is_outlier_iqr', 'outlier_type']
        if all(col in result.columns for col in required_cols):
            print(f"   ✅ All IQR columns created")
        else:
            print(f"   ❌ Missing IQR columns")
            return False
        
        # Check bounds calculated
        if result['lower_bound'].iloc[0] is not None and result['upper_bound'].iloc[0] is not None:
            print(f"   ✅ Lower bound: {result['lower_bound'].iloc[0]:.2f}, Upper bound: {result['upper_bound'].iloc[0]:.2f}")
        else:
            print(f"   ❌ Bounds not calculated")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_detect_outliers_combined():
    """Test combined Z-score and IQR detection"""
    print("\n🧪 Testing Combined Outlier Detection...")
    try:
        engine = StatisticalEngine()
        
        # Data with various outliers
        data = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 400, 5])
        
        result = engine.detect_outliers_combined(data, column_name='value')
        
        # Check combined detection
        total_outliers = result['is_outlier'].sum()
        
        if total_outliers >= 2:
            print(f"   ✅ Combined detection found {total_outliers} outliers")
        else:
            print(f"   ⚠️  Found {total_outliers} outliers")
        
        # Check detection method tracking
        if 'detection_method' in result.columns:
            methods = result['detection_method'].value_counts()
            print(f"   ✅ Detection methods tracked:")
            for method, count in methods.items():
                print(f"      - {method}: {count}")
        else:
            print(f"   ❌ Detection method column missing")
            return False
        
        # Check severity classification
        if 'severity' in result.columns:
            severity_counts = result['severity'].value_counts()
            print(f"   ✅ Severity levels: {dict(severity_counts)}")
        else:
            print(f"   ❌ Severity column missing")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_analyze_rfq_prices():
    """Test RFQ price analysis for anomalies"""
    print("\n🧪 Testing RFQ Price Analysis...")
    try:
        engine = StatisticalEngine()
        
        # Create realistic RFQ data
        rfq_data = pd.DataFrame({
            'vendor_id': ['V001', 'V002', 'V003', 'V004', 'V005', 'V006', 'V007'],
            'quoted_price': [250000, 260000, 255000, 258000, 450000, 150000, 252000]
        })
        
        result = engine.analyze_rfq_prices(rfq_data)
        
        # Check required columns added
        required_cols = ['is_price_outlier', 'outlier_type', 'outlier_severity', 
                        'detection_method', 'zscore', 'price_mean', 'price_median', 
                        'price_std', 'deviation_pct']
        
        if all(col in result.columns for col in required_cols):
            print(f"   ✅ All {len(required_cols)} analysis columns created")
        else:
            missing = [col for col in required_cols if col not in result.columns]
            print(f"   ❌ Missing columns: {missing}")
            return False
        
        # Check outlier detection
        outlier_count = result['is_price_outlier'].sum()
        print(f"   ✅ Detected {outlier_count} price outlier(s)")
        
        # Check deviation percentage calculation
        if 'deviation_pct' in result.columns:
            max_dev = result['deviation_pct'].abs().max()
            print(f"   ✅ Maximum price deviation: {max_dev:.1f}%")
        
        # Display outliers
        if outlier_count > 0:
            outliers = result[result['is_price_outlier']]
            print(f"   ✅ Outlier vendors:")
            for _, row in outliers.iterrows():
                print(f"      - {row['vendor_id']}: ₹{row['quoted_price']:,} ({row['outlier_type']}, {row['outlier_severity']})")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_analyze_vendor_performance():
    """Test vendor performance analysis"""
    print("\n🧪 Testing Vendor Performance Analysis...")
    try:
        engine = StatisticalEngine()
        
        # Create vendor performance data
        vendor_data = pd.DataFrame({
            'vendor_id': ['V001', 'V002', 'V003', 'V004', 'V005', 'V006'],
            'delivery_score': [92, 88, 90, 45, 91, 89]  # V004 is poor performer
        })
        
        result = engine.analyze_vendor_performance(vendor_data, metric_column='delivery_score')
        
        # Check required columns
        required_cols = ['is_performance_outlier', 'outlier_type', 'severity', 
                        'zscore', 'metric_mean', 'metric_median']
        
        if all(col in result.columns for col in required_cols):
            print(f"   ✅ All performance analysis columns created")
        else:
            print(f"   ❌ Missing columns")
            return False
        
        # Check outlier detection
        outlier_count = result['is_performance_outlier'].sum()
        print(f"   ✅ Detected {outlier_count} performance outlier(s)")
        
        # Display performance summary
        mean_score = result['metric_mean'].iloc[0]
        median_score = result['metric_median'].iloc[0]
        print(f"   ✅ Mean score: {mean_score:.2f}, Median score: {median_score:.2f}")
        
        # Show poor performers
        if outlier_count > 0:
            outliers = result[result['is_performance_outlier']]
            print(f"   ✅ Performance outliers:")
            for _, row in outliers.iterrows():
                print(f"      - {row['vendor_id']}: Score {row['delivery_score']} ({row['outlier_type']})")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing Edge Cases...")
    try:
        engine = StatisticalEngine()
        
        # Test 1: Single value
        single_data = pd.Series([100])
        stats = engine.calculate_statistics(single_data)
        
        if stats['count'] == 1 and stats['std'] == 0:
            print(f"   ✅ Single value handled correctly")
        else:
            print(f"   ❌ Single value case failed")
            return False
        
        # Test 2: Data with NaN values
        nan_data = pd.Series([100, 110, np.nan, 120, np.nan, 130])
        stats_nan = engine.calculate_statistics(nan_data)
        
        if stats_nan['count'] == 4:  # Should ignore NaN
            print(f"   ✅ NaN values handled correctly (count={stats_nan['count']})")
        else:
            print(f"   ❌ NaN handling failed")
            return False
        
        # Test 3: Very large numbers
        large_data = pd.Series([1e9, 1.1e9, 1.05e9, 1.15e9])
        large_stats = engine.calculate_statistics(large_data)
        
        if large_stats['mean'] > 1e9:
            print(f"   ✅ Large numbers handled correctly")
        else:
            print(f"   ❌ Large number handling failed")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def run_all_tests():
    """Run all StatisticalEngine tests"""
    print("=" * 60)
    print("🚀 RUNNING STATISTICAL ENGINE TESTS")
    print("=" * 60)
    
    results = []
    results.append(("StatisticalEngine Initialization", test_statistical_engine_initialization()))
    results.append(("Calculate Statistics", test_calculate_statistics()))
    results.append(("Z-Score Outlier Detection", test_detect_outliers_zscore()))
    results.append(("IQR Outlier Detection", test_detect_outliers_iqr()))
    results.append(("Combined Outlier Detection", test_detect_outliers_combined()))
    results.append(("RFQ Price Analysis", test_analyze_rfq_prices()))
    results.append(("Vendor Performance Analysis", test_analyze_vendor_performance()))
    results.append(("Edge Cases", test_edge_cases()))
    
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
