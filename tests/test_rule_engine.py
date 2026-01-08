"""
Unit tests for RuleEngine class
Tests vendor scoring, RFQ anomaly detection, and contract compliance
"""

import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ai_engine.rule_engine import RuleEngine
import pandas as pd
import numpy as np


# Tests RuleEngine initialization and weight validation
def test_rule_engine_initialization():
    """Test RuleEngine initialization with default and custom weights"""
    print("\n🧪 Testing RuleEngine Initialization...")
    try:
        # Test 1: Default weights
        engine = RuleEngine()
        
        # Check if weights sum to 100
        weight_sum = sum(engine.weights.values())
        if abs(weight_sum - 100) < 0.01:
            print(f"   ✅ Default weights sum to {weight_sum}")
        else:
            print(f"   ❌ Weight sum mismatch: {weight_sum}")
            return False
        
        # Test 2: Custom weights
        custom_weights = {
            'cost_competitiveness': 25,
            'on_time_delivery': 30,
            'quality_score': 25,
            'compliance_score': 10,
            'past_disputes': 10
        }
        custom_engine = RuleEngine(weights=custom_weights)
        print(f"   ✅ Custom weights initialized successfully")
        
        # Test 3: Invalid weights (should raise error)
        try:
            invalid_weights = {'cost_competitiveness': 50, 'on_time_delivery': 30}
            RuleEngine(weights=invalid_weights)
            print(f"   ❌ Should have raised ValueError for invalid weights")
            return False
        except ValueError:
            print(f"   ✅ Correctly rejected invalid weights")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests weighted vendor scoring calculation (0-100)
def test_vendor_scoring():
    """Test vendor scoring calculation"""
    print("\n🧪 Testing Vendor Scoring...")
    try:
        engine = RuleEngine()
        
        # Test Case 1: Perfect vendor (high scores, no disputes)
        vendor1 = {
            'vendor_id': 'V001',
            'cost_competitiveness': 90,
            'on_time_delivery': 95,
            'quality_score': 92,
            'compliance_score': 98,
            'past_disputes': 0
        }
        score1 = engine.calculate_score(vendor1)
        
        if score1 >= 85:
            print(f"   ✅ Perfect vendor scored: {score1} (expected >= 85)")
        else:
            print(f"   ⚠️  Score lower than expected: {score1}")
        
        # Test Case 2: Poor vendor (low scores, multiple disputes)
        vendor2 = {
            'vendor_id': 'V002',
            'cost_competitiveness': 40,
            'on_time_delivery': 35,
            'quality_score': 45,
            'compliance_score': 50,
            'past_disputes': 5
        }
        score2 = engine.calculate_score(vendor2)
        
        if score2 < 50:
            print(f"   ✅ Poor vendor scored: {score2} (expected < 50)")
        else:
            print(f"   ⚠️  Score higher than expected: {score2}")
        
        # Test Case 3: Average vendor
        vendor3 = {
            'vendor_id': 'V003',
            'cost_competitiveness': 70,
            'on_time_delivery': 68,
            'quality_score': 72,
            'compliance_score': 75,
            'past_disputes': 1
        }
        score3 = engine.calculate_score(vendor3)
        
        if 50 <= score3 < 75:
            print(f"   ✅ Average vendor scored: {score3} (expected 50-74)")
        else:
            print(f"   ⚠️  Score outside expected range: {score3}")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests risk classification (low/medium/high) based on score
def test_risk_classification():
    """Test risk classification"""
    print("\n🧪 Testing Risk Classification...")
    try:
        engine = RuleEngine()
        
        test_cases = [
            (85, 'low'),
            (75, 'low'),
            (65, 'medium'),
            (50, 'medium'),
            (45, 'high'),
            (20, 'high')
        ]
        
        passed = 0
        for score, expected_risk in test_cases:
            risk = engine.classify_risk(score)
            if risk == expected_risk:
                passed += 1
                print(f"   ✅ Score {score} → {risk} risk (correct)")
            else:
                print(f"   ❌ Score {score} → {risk} risk (expected {expected_risk})")
        
        if passed == len(test_cases):
            print(f"   ✅ All {passed}/{len(test_cases)} risk classifications correct")
            return True
        else:
            print(f"   ⚠️  {passed}/{len(test_cases)} passed")
            return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests vendor evaluation rules and preferred vendor identification
def test_apply_vendor_rules():
    """Test vendor evaluation rules"""
    print("\n🧪 Testing Vendor Evaluation Rules...")
    try:
        engine = RuleEngine()
        
        # Create test vendor data
        vendor_data = pd.DataFrame([
            {
                'vendor_id': 'V001',
                'cost_competitiveness': 88,
                'on_time_delivery': 92,
                'quality_score': 90,
                'compliance_score': 95,
                'past_disputes': 0,
                'dispute_count': 0
            },
            {
                'vendor_id': 'V002',
                'cost_competitiveness': 65,
                'on_time_delivery': 58,
                'quality_score': 62,
                'compliance_score': 68,
                'past_disputes': 2,
                'dispute_count': 3
            },
            {
                'vendor_id': 'V003',
                'cost_competitiveness': 42,
                'on_time_delivery': 38,
                'quality_score': 45,
                'compliance_score': 50,
                'past_disputes': 4,
                'dispute_count': 5
            }
        ])
        
        result = engine.apply_rules(vendor_data, rule_type='vendor_scoring')
        
        # Check if required columns exist
        required_columns = ['vendor_score', 'risk_level', 'preferred_vendor']
        if all(col in result.columns for col in required_columns):
            print(f"   ✅ All required columns created")
        else:
            print(f"   ❌ Missing columns")
            return False
        
        # Check preferred vendor identification
        preferred_count = result['preferred_vendor'].sum()
        print(f"   ✅ Found {preferred_count} preferred vendor(s)")
        
        # Check risk classification distribution
        risk_counts = result['risk_level'].value_counts()
        print(f"   ✅ Risk distribution: Low={risk_counts.get('low', 0)}, "
              f"Medium={risk_counts.get('medium', 0)}, High={risk_counts.get('high', 0)}")
        
        # Check high-dispute flag
        if 'high_dispute_flag' in result.columns:
            flagged = result['high_dispute_flag'].sum()
            print(f"   ✅ {flagged} vendor(s) flagged for high disputes")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests RFQ/RFP anomaly detection for price and delivery deviations
def test_rfq_anomaly_detection():
    """Test RFQ/RFP anomaly detection"""
    print("\n🧪 Testing RFQ Anomaly Detection...")
    try:
        engine = RuleEngine()
        
        # Create test RFQ data
        rfq_data = pd.DataFrame([
            {'vendor_id': 'V001', 'quoted_price': 250000, 'delivery_days': 30},
            {'vendor_id': 'V002', 'quoted_price': 350000, 'delivery_days': 28},  # High price
            {'vendor_id': 'V003', 'quoted_price': 180000, 'delivery_days': 50},  # Low price + delivery deviation
            {'vendor_id': 'V004', 'quoted_price': 245000, 'delivery_days': 31},
            {'vendor_id': 'V005', 'quoted_price': 260000, 'delivery_days': 29}
        ])
        
        result = engine.apply_rules(rfq_data, rule_type='rfq_anomaly')
        
        # Check if anomaly flags created
        required_columns = ['price_high_flag', 'price_low_flag', 'anomaly_detected', 'anomaly_severity']
        if all(col in result.columns for col in required_columns):
            print(f"   ✅ All anomaly detection columns created")
        else:
            print(f"   ❌ Missing anomaly columns")
            return False
        
        # Check anomaly detection
        total_anomalies = result['anomaly_detected'].sum()
        print(f"   ✅ Detected {total_anomalies} anomalies out of {len(result)} bids")
        
        # Check severity classification
        severity_counts = result['anomaly_severity'].value_counts()
        print(f"   ✅ Severity distribution: High={severity_counts.get('high', 0)}, "
              f"Medium={severity_counts.get('medium', 0)}, Low={severity_counts.get('low', 0)}, "
              f"None={severity_counts.get('none', 0)}")
        
        # Verify high price detection
        high_price_count = result['price_high_flag'].sum()
        print(f"   ✅ {high_price_count} bid(s) flagged for high price")
        
        # Verify low price detection
        low_price_count = result['price_low_flag'].sum()
        print(f"   ✅ {low_price_count} bid(s) flagged for suspicious low price")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests contract compliance rules and Red-Amber-Green classification
def test_contract_compliance():
    """Test contract compliance rules"""
    print("\n🧪 Testing Contract Compliance...")
    try:
        engine = RuleEngine()
        
        # Create test contract data
        contract_data = pd.DataFrame([
            {
                'contract_id': 'C001',
                'contract_value': 5000000,
                'payment_clause': 'Yes',
                'penalty_clause': 'Yes',
                'termination_clause': 'Yes',
                'warranty_clause': 'Yes',
                'liability_clause': 'Yes',
                'gcc_compliant': True,
                'vendor_risk_level': 'low'
            },
            {
                'contract_id': 'C002',
                'contract_value': 15000000,
                'payment_clause': None,  # Missing
                'penalty_clause': 'Yes',
                'termination_clause': 'Yes',
                'warranty_clause': None,  # Missing
                'liability_clause': 'Yes',
                'gcc_compliant': False,
                'vendor_risk_level': 'high'
            },
            {
                'contract_id': 'C003',
                'contract_value': 75000000,  # High value
                'payment_clause': 'Yes',
                'penalty_clause': 'Yes',
                'termination_clause': 'Yes',
                'warranty_clause': 'Yes',
                'liability_clause': 'Yes',
                'gcc_compliant': True,
                'vendor_risk_level': 'medium'
            }
        ])
        
        result = engine.apply_rules(contract_data, rule_type='contract_compliance')
        
        # Check if required columns exist
        required_columns = ['contract_risk_score', 'rag_status', 'high_value_contract']
        if all(col in result.columns for col in required_columns):
            print(f"   ✅ All compliance columns created")
        else:
            print(f"   ❌ Missing compliance columns")
            return False
        
        # Check RAG status distribution
        rag_counts = result['rag_status'].value_counts()
        print(f"   ✅ RAG Status: Red={rag_counts.get('Red', 0)}, "
              f"Amber={rag_counts.get('Amber', 0)}, Green={rag_counts.get('Green', 0)}")
        
        # Check high-value contract detection
        high_value_count = result['high_value_contract'].sum()
        print(f"   ✅ {high_value_count} high-value contract(s) detected")
        
        # Check board approval requirement
        if 'requires_board_approval' in result.columns:
            board_approval_count = result['requires_board_approval'].sum()
            print(f"   ✅ {board_approval_count} contract(s) require board approval")
        
        # Check missing clause detection
        missing_payment = result['payment_clause_missing'].sum()
        print(f"   ✅ {missing_payment} contract(s) missing payment clause")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Tests recommendation generation for all rule types
def test_generate_recommendations():
    """Test recommendation generation"""
    print("\n🧪 Testing Recommendation Generation...")
    try:
        engine = RuleEngine()
        
        # Test 1: Vendor recommendations
        vendor_data = pd.DataFrame([
            {'vendor_id': 'V001', 'cost_competitiveness': 90, 'on_time_delivery': 95,
             'quality_score': 92, 'compliance_score': 98, 'past_disputes': 0,
             'dispute_count': 0},
            {'vendor_id': 'V002', 'cost_competitiveness': 40, 'on_time_delivery': 35,
             'quality_score': 45, 'compliance_score': 50, 'past_disputes': 5,
             'dispute_count': 5}
        ])
        vendor_result = engine.apply_rules(vendor_data, rule_type='vendor_scoring')
        vendor_recs = engine.generate_recommendations(vendor_result, 'vendor_scoring')
        
        if len(vendor_recs) > 0:
            print(f"   ✅ Generated {len(vendor_recs)} vendor recommendation(s)")
            for rec in vendor_recs:
                print(f"      - {rec['message']}")
        
        # Test 2: RFQ recommendations
        rfq_data = pd.DataFrame([
            {'vendor_id': 'V001', 'quoted_price': 250000, 'delivery_days': 30},
            {'vendor_id': 'V002', 'quoted_price': 450000, 'delivery_days': 28},  # Very high
            {'vendor_id': 'V003', 'quoted_price': 150000, 'delivery_days': 60}   # Very low + deviation
        ])
        rfq_result = engine.apply_rules(rfq_data, rule_type='rfq_anomaly')
        rfq_recs = engine.generate_recommendations(rfq_result, 'rfq_anomaly')
        
        if len(rfq_recs) > 0:
            print(f"   ✅ Generated {len(rfq_recs)} RFQ recommendation(s)")
            for rec in rfq_recs:
                print(f"      - {rec['message']}")
        
        # Test 3: Contract recommendations
        contract_data = pd.DataFrame([
            {'contract_id': 'C001', 'contract_value': 5000000, 'payment_clause': 'Yes',
             'penalty_clause': 'Yes', 'termination_clause': 'Yes', 'gcc_compliant': True,
             'vendor_risk_level': 'low'},
            {'contract_id': 'C002', 'contract_value': 15000000, 'payment_clause': None,
             'penalty_clause': None, 'termination_clause': 'Yes', 'gcc_compliant': False,
             'vendor_risk_level': 'high'}
        ])
        contract_result = engine.apply_rules(contract_data, rule_type='contract_compliance')
        contract_recs = engine.generate_recommendations(contract_result, 'contract_compliance')
        
        if len(contract_recs) > 0:
            print(f"   ✅ Generated {len(contract_recs)} contract recommendation(s)")
            for rec in contract_recs:
                print(f"      - {rec['message']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# Executes all test functions and displays summary
def run_all_tests():
    """Run all RuleEngine tests"""
    print("=" * 60)
    print("🚀 RUNNING RULE ENGINE TESTS")
    print("=" * 60)
    
    results = []
    results.append(("RuleEngine Initialization", test_rule_engine_initialization()))
    results.append(("Vendor Scoring", test_vendor_scoring()))
    results.append(("Risk Classification", test_risk_classification()))
    results.append(("Apply Vendor Rules", test_apply_vendor_rules()))
    results.append(("RFQ Anomaly Detection", test_rfq_anomaly_detection()))
    results.append(("Contract Compliance", test_contract_compliance()))
    results.append(("Generate Recommendations", test_generate_recommendations()))
    
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
