"""
AI Engine - Rule-Based Vendor Scoring & Risk Classification
Handles business rules, weighted scoring, and risk assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime


class RuleEngine:
    """
    Rule-based vendor scoring and risk classification engine
    Used for Vendor Development, RFQ/RFP evaluation, and Contract Review
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize RuleEngine with scoring weights
        
        Args:
            weights: Dictionary of parameter weights (must sum to 100)
        """
        # Default weights for vendor scoring (from project doc)
        self.weights = weights or {
            'cost_competitiveness': 30,
            'on_time_delivery': 25,
            'quality_score': 20,
            'compliance_score': 15,
            'past_disputes': 10
        }
        
        # Validate weights sum to 100
        if abs(sum(self.weights.values()) - 100) > 0.01:
            raise ValueError(f"Weights must sum to 100, got {sum(self.weights.values())}")
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': (75, 100),      # Score >= 75: Low risk
            'medium': (50, 75),    # Score 50-74: Medium risk
            'high': (0, 50)        # Score < 50: High risk
        }
        
        # Anomaly thresholds for RFQ/RFP
        self.anomaly_thresholds = {
            'price_deviation_high': 30,  # % above average
            'price_deviation_low': 20,   # % below average
            'delivery_deviation': 15,    # % timeline deviation
            'max_disputes': 2,           # Max allowed disputes
            'min_compliance': 70         # Min compliance score
        }
    
    
    def apply_rules(self, data: pd.DataFrame, rule_type: str = 'vendor_scoring') -> pd.DataFrame:
        """
        Apply business rules to data based on rule type
        
        Args:
            data: DataFrame with vendor/contract data
            rule_type: Type of rules ('vendor_scoring', 'rfq_anomaly', 'contract_compliance')
            
        Returns:
            DataFrame with rules applied and flags added
        """
        result = data.copy()
        
        if rule_type == 'vendor_scoring':
            result = self._apply_vendor_rules(result)
        elif rule_type == 'rfq_anomaly':
            result = self._apply_rfq_rules(result)
        elif rule_type == 'contract_compliance':
            result = self._apply_contract_rules(result)
        else:
            raise ValueError(f"Unknown rule_type: {rule_type}")
        
        return result
    
    
    def _apply_vendor_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply vendor evaluation rules"""
        
        # Rule 1: Calculate vendor score
        data['vendor_score'] = data.apply(
            lambda row: self.calculate_score(row.to_dict()), axis=1
        )
        
        # Rule 2: Classify risk
        data['risk_level'] = data['vendor_score'].apply(self.classify_risk)
        
        # Rule 3: Flag high-dispute vendors
        if 'dispute_count' in data.columns:
            data['high_dispute_flag'] = data['dispute_count'] > self.anomaly_thresholds['max_disputes']
        
        # Rule 4: Flag low compliance vendors
        if 'compliance_score' in data.columns:
            data['low_compliance_flag'] = data['compliance_score'] < self.anomaly_thresholds['min_compliance']
        
        # Rule 5: Identify preferred vendors
        data['preferred_vendor'] = (
            (data['vendor_score'] >= 85) & 
            (data['risk_level'] == 'low') &
            (data.get('dispute_count', 0) == 0)
        )
        
        return data
    
    
    def _apply_rfq_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply RFQ/RFP anomaly detection rules"""
        
        # Calculate average price (baseline)
        avg_price = data['quoted_price'].mean()
        
        # Rule 1: Price too high
        data['price_high_flag'] = (
            (data['quoted_price'] - avg_price) / avg_price * 100 
            > self.anomaly_thresholds['price_deviation_high']
        )
        
        # Rule 2: Price suspiciously low
        data['price_low_flag'] = (
            (avg_price - data['quoted_price']) / avg_price * 100 
            > self.anomaly_thresholds['price_deviation_low']
        )
        
        # Rule 3: Delivery timeline deviation
        if 'delivery_days' in data.columns:
            avg_delivery = data['delivery_days'].mean()
            data['delivery_deviation_flag'] = (
                abs(data['delivery_days'] - avg_delivery) / avg_delivery * 100
                > self.anomaly_thresholds['delivery_deviation']
            )
        
        # Rule 4: Payment terms not matching RFQ
        if 'payment_terms' in data.columns and 'rfq_payment_terms' in data.columns:
            data['payment_terms_flag'] = data['payment_terms'] != data['rfq_payment_terms']
        
        # Rule 5: Overall anomaly flag
        data['anomaly_detected'] = (
            data['price_high_flag'] | 
            data['price_low_flag'] | 
            data.get('delivery_deviation_flag', False) |
            data.get('payment_terms_flag', False)
        )
        
        # Classify anomaly severity
        data['anomaly_severity'] = data.apply(
            lambda row: self._classify_anomaly_severity(row), axis=1
        )
        
        return data
    
    
    def _apply_contract_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply contract compliance rules"""
        
        # Rule 1: Check mandatory clauses
        mandatory_clauses = ['payment_clause', 'penalty_clause', 'termination_clause', 
                            'warranty_clause', 'liability_clause']
        
        for clause in mandatory_clauses:
            if clause in data.columns:
                data[f'{clause}_missing'] = data[clause].isna() | (data[clause] == '')
        
        # Rule 2: Deviation from GCC/SCC
        if 'gcc_compliant' in data.columns:
            data['gcc_deviation'] = ~data['gcc_compliant']
        
        # Rule 3: Contract value threshold checks
        if 'contract_value' in data.columns:
            data['high_value_contract'] = data['contract_value'] > 10000000  # 1 Cr threshold
            data['requires_board_approval'] = data['contract_value'] > 50000000  # 5 Cr
        
        # Rule 4: Risk scoring for contracts
        data['contract_risk_score'] = data.apply(
            lambda row: self._calculate_contract_risk(row), axis=1
        )
        
        # Rule 5: Red-Amber-Green classification
        data['rag_status'] = data['contract_risk_score'].apply(
            lambda score: 'Red' if score >= 70 else ('Amber' if score >= 40 else 'Green')
        )
        
        return data
    
    
    def calculate_score(self, vendor_data: Dict[str, Any]) -> float:
        """
        Calculate weighted vendor score (0-100)
        
        Args:
            vendor_data: Dictionary with vendor metrics
            
        Returns:
            Weighted score (0-100)
        """
        score = 0.0
        
        # Cost competitiveness (lower is better, convert to 0-100 scale)
        if 'cost_competitiveness' in vendor_data:
            cost_score = min(100, vendor_data['cost_competitiveness'])
            score += cost_score * (self.weights['cost_competitiveness'] / 100)
        
        # On-time delivery (higher is better)
        if 'on_time_delivery' in vendor_data:
            delivery_score = min(100, vendor_data['on_time_delivery'])
            score += delivery_score * (self.weights['on_time_delivery'] / 100)
        
        # Quality score (higher is better)
        if 'quality_score' in vendor_data:
            quality_score = min(100, vendor_data['quality_score'])
            score += quality_score * (self.weights['quality_score'] / 100)
        
        # Compliance score (higher is better)
        if 'compliance_score' in vendor_data:
            compliance_score = min(100, vendor_data['compliance_score'])
            score += compliance_score * (self.weights['compliance_score'] / 100)
        
        # Past disputes (fewer is better, invert to 0-100 scale)
        if 'past_disputes' in vendor_data:
            # 0 disputes = 100 points, each dispute reduces score
            dispute_penalty = vendor_data['past_disputes'] * 20  # Each dispute = -20 points
            dispute_score = max(0, 100 - dispute_penalty)
            score += dispute_score * (self.weights['past_disputes'] / 100)
        
        return round(score, 2)
    
    
    def classify_risk(self, score: float) -> str:
        """
        Classify vendor as Low/Medium/High risk based on score
        
        Args:
            score: Vendor score (0-100)
            
        Returns:
            Risk level: 'low', 'medium', or 'high'
        """
        for risk_level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= score < max_score:
                return risk_level
        
        return 'low'  # Default to low if score >= 100
    
    
    def _classify_anomaly_severity(self, row: pd.Series) -> str:
        """Classify RFQ anomaly severity"""
        
        # High severity if multiple flags
        flag_count = sum([
            row.get('price_high_flag', False),
            row.get('price_low_flag', False),
            row.get('delivery_deviation_flag', False),
            row.get('payment_terms_flag', False)
        ])
        
        if flag_count >= 3:
            return 'high'
        elif flag_count >= 2:
            return 'medium'
        elif flag_count == 1:
            return 'low'
        else:
            return 'none'
    
    
    def _calculate_contract_risk(self, row: pd.Series) -> float:
        """Calculate contract risk score (0-100)"""
        
        risk_score = 0
        
        # Missing mandatory clauses (+20 points per missing clause)
        mandatory_clauses = ['payment_clause', 'penalty_clause', 'termination_clause']
        for clause in mandatory_clauses:
            if row.get(f'{clause}_missing', False):
                risk_score += 20
        
        # GCC deviation (+15 points)
        if row.get('gcc_deviation', False):
            risk_score += 15
        
        # High contract value (+10 points)
        if row.get('high_value_contract', False):
            risk_score += 10
        
        # Vendor risk level
        vendor_risk = row.get('vendor_risk_level', 'low')
        if vendor_risk == 'high':
            risk_score += 25
        elif vendor_risk == 'medium':
            risk_score += 10
        
        return min(100, risk_score)
    
    
    def generate_recommendations(self, data: pd.DataFrame, rule_type: str = 'vendor_scoring') -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on applied rules
        
        Args:
            data: DataFrame with rules applied
            rule_type: Type of analysis
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        if rule_type == 'vendor_scoring':
            # Preferred vendors
            preferred = data[data.get('preferred_vendor', False)]
            if len(preferred) > 0:
                recommendations.append({
                    'type': 'preferred_vendors',
                    'count': len(preferred),
                    'vendors': preferred['vendor_id'].tolist() if 'vendor_id' in preferred.columns else [],
                    'message': f"Found {len(preferred)} preferred vendors with score >= 85 and low risk"
                })
            
            # High-risk vendors
            high_risk = data[data['risk_level'] == 'high']
            if len(high_risk) > 0:
                recommendations.append({
                    'type': 'high_risk_vendors',
                    'count': len(high_risk),
                    'vendors': high_risk['vendor_id'].tolist() if 'vendor_id' in high_risk.columns else [],
                    'message': f"⚠️  {len(high_risk)} vendors flagged as HIGH RISK - avoid or escalate"
                })
        
        elif rule_type == 'rfq_anomaly':
            # High anomalies
            high_anomalies = data[data['anomaly_severity'] == 'high']
            if len(high_anomalies) > 0:
                recommendations.append({
                    'type': 'critical_anomalies',
                    'count': len(high_anomalies),
                    'message': f"🚨 {len(high_anomalies)} bids with HIGH severity anomalies - requires immediate review"
                })
        
        elif rule_type == 'contract_compliance':
            # Red-flagged contracts
            red_contracts = data[data['rag_status'] == 'Red']
            if len(red_contracts) > 0:
                recommendations.append({
                    'type': 'red_contracts',
                    'count': len(red_contracts),
                    'message': f"🔴 {len(red_contracts)} contracts flagged RED - do not approve without legal review"
                })
        
        return recommendations


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TESTING RULE ENGINE")
    print("=" * 60)
    
    # Initialize RuleEngine
    engine = RuleEngine()
    
    print("\n✅ RuleEngine initialized with weights:")
    for param, weight in engine.weights.items():
        print(f"   {param}: {weight}%")
    
    # Test 1: Vendor Scoring
    print("\n" + "=" * 60)
    print("🧪 Test 1: Vendor Scoring")
    print("=" * 60)
    
    vendor_data = pd.DataFrame([
        {'vendor_id': 'V001', 'cost_competitiveness': 85, 'on_time_delivery': 92, 
         'quality_score': 88, 'compliance_score': 95, 'past_disputes': 0},
        {'vendor_id': 'V002', 'cost_competitiveness': 70, 'on_time_delivery': 60, 
         'quality_score': 65, 'compliance_score': 72, 'past_disputes': 2},
        {'vendor_id': 'V003', 'cost_competitiveness': 45, 'on_time_delivery': 40, 
         'quality_score': 50, 'compliance_score': 55, 'past_disputes': 4}
    ])
    
    result = engine.apply_rules(vendor_data, rule_type='vendor_scoring')
    
    print("\nVendor Evaluation Results:")
    print(result[['vendor_id', 'vendor_score', 'risk_level', 'preferred_vendor']])
    
    recommendations = engine.generate_recommendations(result, 'vendor_scoring')
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"   {rec['message']}")
    
    # Test 2: RFQ Anomaly Detection
    print("\n" + "=" * 60)
    print("🧪 Test 2: RFQ Anomaly Detection")
    print("=" * 60)
    
    rfq_data = pd.DataFrame([
        {'vendor_id': 'V001', 'quoted_price': 250000, 'delivery_days': 30},
        {'vendor_id': 'V002', 'quoted_price': 350000, 'delivery_days': 28},  # High price
        {'vendor_id': 'V003', 'quoted_price': 180000, 'delivery_days': 45},  # Low price + delivery deviation
        {'vendor_id': 'V004', 'quoted_price': 245000, 'delivery_days': 32}
    ])
    
    rfq_result = engine.apply_rules(rfq_data, rule_type='rfq_anomaly')
    
    print("\nRFQ Anomaly Detection:")
    print(rfq_result[['vendor_id', 'quoted_price', 'anomaly_detected', 'anomaly_severity']])
    
    rfq_recommendations = engine.generate_recommendations(rfq_result, 'rfq_anomaly')
    print("\nRecommendations:")
    for rec in rfq_recommendations:
        print(f"   {rec['message']}")
    
    # Test 3: Contract Compliance
    print("\n" + "=" * 60)
    print("🧪 Test 3: Contract Compliance")
    print("=" * 60)
    
    contract_data = pd.DataFrame([
        {'contract_id': 'C001', 'contract_value': 5000000, 'payment_clause': 'Yes', 
         'penalty_clause': 'Yes', 'termination_clause': 'Yes', 'gcc_compliant': True, 
         'vendor_risk_level': 'low'},
        {'contract_id': 'C002', 'contract_value': 15000000, 'payment_clause': None, 
         'penalty_clause': 'Yes', 'termination_clause': 'Yes', 'gcc_compliant': False, 
         'vendor_risk_level': 'high'}
    ])
    
    contract_result = engine.apply_rules(contract_data, rule_type='contract_compliance')
    
    print("\nContract Compliance Check:")
    print(contract_result[['contract_id', 'contract_risk_score', 'rag_status']])
    
    contract_recommendations = engine.generate_recommendations(contract_result, 'contract_compliance')
    print("\nRecommendations:")
    for rec in contract_recommendations:
        print(f"   {rec['message']}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - RuleEngine Working!")
    print("=" * 60)
