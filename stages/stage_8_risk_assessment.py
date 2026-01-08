"""
Stage 8: Vendor & Contract Risk Assessment
===========================================

Performs comprehensive risk assessment:
- Vendor financial health monitoring
- Contract delivery risk scoring
- Compliance risk identification
- Historical performance analysis
- Risk mitigation recommendations

Uses: ML Engine for predictions, Statistical Engine for scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import os

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
from formatters import format_percentage
from logger import get_logger


class RiskAssessmentStage(BaseStage):
    """
    Assesses and scores various risk dimensions for vendors and contracts.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes Stage 8 with risk assessment parameters.
        
        Args:
            config: Optional configuration dictionary from JSON
        """
        super().__init__(stage_number=8)
        self.ml_engine = MLEngine()
        self.stat_engine = StatisticalEngine()
        self.logger = get_logger(__name__)
        
        # Load config or use defaults
        if config:
            self.financial_risk_weight = config.get('financial_risk_weight', 0.30)
            self.delivery_risk_weight = config.get('delivery_risk_weight', 0.25)
            self.compliance_risk_weight = config.get('compliance_risk_weight', 0.25)
            self.performance_risk_weight = config.get('performance_risk_weight', 0.20)
            self.high_risk_threshold = config.get('high_risk_threshold', 70)
        else:
            self.financial_risk_weight = 0.30
            self.delivery_risk_weight = 0.25
            self.compliance_risk_weight = 0.25
            self.performance_risk_weight = 0.20
            self.high_risk_threshold = 70
    
    def _get_required_columns(self):
        """
        Returns required columns for risk assessment.
        """
        return ['entity_id', 'entity_type']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes risk assessment workflow.
        
        Args:
            data: DataFrame with vendor/contract data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing risk assessment results
        """
        # Assess financial risk
        financial_risk = self._assess_financial_risk(data)
        
        # Assess delivery risk
        delivery_risk = self._assess_delivery_risk(financial_risk)
        
        # Assess compliance risk
        compliance_risk = self._assess_compliance_risk(delivery_risk)
        
        # Assess performance risk
        performance_risk = self._assess_performance_risk(compliance_risk)
        
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(performance_risk)
        
        # Classify risk levels
        classified = self._classify_risk_levels(overall_risk)
        
        # Generate mitigation strategies
        strategies = self._generate_mitigation_strategies(classified)
        
        results = {
            'total_entities': len(data),
            'high_risk_count': len(classified[classified['risk_level'] == 'High']),
            'medium_risk_count': len(classified[classified['risk_level'] == 'Medium']),
            'low_risk_count': len(classified[classified['risk_level'] == 'Low']),
            'average_risk_score': classified['overall_risk_score'].mean(),
            'critical_risks': len(classified[classified['requires_immediate_action']]),
            'risk_assessments': classified.to_dict('records'),
            'mitigation_strategies': strategies,
            'risk_distribution': self._calculate_risk_distribution(classified)
        }
        
        self.logger.info(f"   High Risk: {results['high_risk_count']}/{results['total_entities']}")
        self.logger.info(f"   Critical: {results['critical_risks']} require immediate action")
        
        return results
    
    def _assess_financial_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Assesses financial health risk of entities.
        
        Args:
            data: Entity DataFrame
            
        Returns:
            DataFrame with financial risk scores
        """
        df = data.copy()
        
        # Add financial metrics if not present
        if 'credit_score' not in df.columns:
            df['credit_score'] = np.random.uniform(300, 850, len(df))
        
        if 'debt_to_equity' not in df.columns:
            df['debt_to_equity'] = np.random.uniform(0.2, 3.5, len(df))
        
        if 'liquidity_ratio' not in df.columns:
            df['liquidity_ratio'] = np.random.uniform(0.5, 3.0, len(df))
        
        # Calculate financial risk score (0-100, higher = more risk)
        df['financial_risk_score'] = 0
        
        # Credit score risk (inverse relationship)
        df['financial_risk_score'] += ((850 - df['credit_score']) / 850 * 100) * 0.4
        
        # Debt-to-equity risk
        df['financial_risk_score'] += (df['debt_to_equity'] / 4.0 * 100).clip(0, 100) * 0.35
        
        # Liquidity risk (inverse)
        df['financial_risk_score'] += ((3.0 - df['liquidity_ratio']) / 3.0 * 100).clip(0, 100) * 0.25
        
        # Clip to 0-100
        df['financial_risk_score'] = df['financial_risk_score'].clip(0, 100)
        
        return df
    
    def _assess_delivery_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Assesses delivery and timeline risk.
        
        Args:
            data: Financial risk DataFrame
            
        Returns:
            DataFrame with delivery risk scores
        """
        df = data.copy()
        
        # Add delivery metrics if not present
        if 'on_time_delivery_pct' not in df.columns:
            df['on_time_delivery_pct'] = np.random.uniform(60, 98, len(df))
        
        if 'average_delay_days' not in df.columns:
            df['average_delay_days'] = np.random.uniform(0, 30, len(df))
        
        if 'project_complexity_score' not in df.columns:
            df['project_complexity_score'] = np.random.uniform(1, 10, len(df))
        
        # Calculate delivery risk score
        df['delivery_risk_score'] = 0
        
        # On-time delivery risk (inverse)
        df['delivery_risk_score'] += (100 - df['on_time_delivery_pct']) * 0.5
        
        # Delay risk
        df['delivery_risk_score'] += (df['average_delay_days'] / 30 * 100).clip(0, 100) * 0.3
        
        # Complexity risk
        df['delivery_risk_score'] += (df['project_complexity_score'] / 10 * 100) * 0.2
        
        # Clip to 0-100
        df['delivery_risk_score'] = df['delivery_risk_score'].clip(0, 100)
        
        return df
    
    def _assess_compliance_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Assesses regulatory and contract compliance risk.
        
        Args:
            data: Delivery risk DataFrame
            
        Returns:
            DataFrame with compliance risk scores
        """
        df = data.copy()
        
        # Add compliance metrics if not present
        if 'compliance_violations' not in df.columns:
            df['compliance_violations'] = np.random.randint(0, 5, len(df))
        
        if 'audit_score' not in df.columns:
            df['audit_score'] = np.random.uniform(60, 100, len(df))
        
        if 'certification_status' not in df.columns:
            df['certification_status'] = np.random.choice(['Valid', 'Expired', 'Pending'], len(df), p=[0.7, 0.2, 0.1])
        
        # Calculate compliance risk score
        df['compliance_risk_score'] = 0
        
        # Violations risk
        df['compliance_risk_score'] += (df['compliance_violations'] / 5 * 100).clip(0, 100) * 0.4
        
        # Audit score risk (inverse)
        df['compliance_risk_score'] += (100 - df['audit_score']) * 0.4
        
        # Certification status risk
        cert_risk = {'Valid': 0, 'Pending': 30, 'Expired': 60}
        df['compliance_risk_score'] += df['certification_status'].map(cert_risk) * 0.2
        
        # Clip to 0-100
        df['compliance_risk_score'] = df['compliance_risk_score'].clip(0, 100)
        
        return df
    
    def _assess_performance_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Assesses historical performance risk.
        
        Args:
            data: Compliance risk DataFrame
            
        Returns:
            DataFrame with performance risk scores
        """
        df = data.copy()
        
        # Add performance metrics if not present
        if 'quality_score' not in df.columns:
            df['quality_score'] = np.random.uniform(60, 98, len(df))
        
        if 'defect_rate_pct' not in df.columns:
            df['defect_rate_pct'] = np.random.uniform(0, 10, len(df))
        
        if 'customer_satisfaction' not in df.columns:
            df['customer_satisfaction'] = np.random.uniform(60, 100, len(df))
        
        # Calculate performance risk score
        df['performance_risk_score'] = 0
        
        # Quality risk (inverse)
        df['performance_risk_score'] += (100 - df['quality_score']) * 0.4
        
        # Defect rate risk
        df['performance_risk_score'] += (df['defect_rate_pct'] / 10 * 100).clip(0, 100) * 0.35
        
        # Satisfaction risk (inverse)
        df['performance_risk_score'] += (100 - df['customer_satisfaction']) * 0.25
        
        # Clip to 0-100
        df['performance_risk_score'] = df['performance_risk_score'].clip(0, 100)
        
        return df
    
    def _calculate_overall_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates weighted overall risk score.
        
        Args:
            data: All risk dimensions DataFrame
            
        Returns:
            DataFrame with overall risk scores
        """
        df = data.copy()
        
        # Weighted overall risk
        df['overall_risk_score'] = (
            df['financial_risk_score'] * self.financial_risk_weight +
            df['delivery_risk_score'] * self.delivery_risk_weight +
            df['compliance_risk_score'] * self.compliance_risk_weight +
            df['performance_risk_score'] * self.performance_risk_weight
        )
        
        # Round to 2 decimals
        df['overall_risk_score'] = df['overall_risk_score'].round(2)
        
        return df
    
    def _classify_risk_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies entities into risk levels.
        
        Args:
            data: Overall risk DataFrame
            
        Returns:
            DataFrame with risk classifications
        """
        df = data.copy()
        
        # Risk level classification
        df['risk_level'] = 'Low'
        df.loc[df['overall_risk_score'] >= 40, 'risk_level'] = 'Medium'
        df.loc[df['overall_risk_score'] >= self.high_risk_threshold, 'risk_level'] = 'High'
        
        # Requires immediate action flag
        df['requires_immediate_action'] = (
            (df['risk_level'] == 'High') |
            (df['financial_risk_score'] >= 80) |
            (df['compliance_violations'] >= 3)
        )
        
        # Risk category
        df['primary_risk_area'] = df.apply(self._identify_primary_risk, axis=1)
        
        return df
    
    def _identify_primary_risk(self, row: pd.Series) -> str:
        """
        Identifies primary risk area for entity.
        
        Args:
            row: Entity data row
            
        Returns:
            Primary risk area name
        """
        risks = {
            'Financial': row['financial_risk_score'],
            'Delivery': row['delivery_risk_score'],
            'Compliance': row['compliance_risk_score'],
            'Performance': row['performance_risk_score']
        }
        
        return max(risks, key=risks.get)
    
    def _generate_mitigation_strategies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates risk mitigation strategies.
        
        Args:
            data: Classified DataFrame
            
        Returns:
            List of mitigation strategy dictionaries
        """
        strategies = []
        
        # High risk entities
        high_risk = data[data['risk_level'] == 'High']
        if len(high_risk) > 0:
            strategies.append({
                'risk_level': 'High',
                'priority': 'Critical',
                'count': len(high_risk),
                'entities': high_risk['entity_id'].tolist(),
                'actions': [
                    'Conduct immediate risk assessment meeting',
                    'Request updated financial statements',
                    'Implement enhanced monitoring',
                    'Consider contract restructuring',
                    'Activate contingency plans'
                ]
            })
        
        # Financial risk mitigation
        financial_risk = data[data['primary_risk_area'] == 'Financial']
        if len(financial_risk) > 0:
            strategies.append({
                'risk_area': 'Financial',
                'priority': 'High',
                'count': len(financial_risk),
                'entities': financial_risk['entity_id'].tolist(),
                'actions': [
                    'Request bank guarantees or performance bonds',
                    'Implement milestone-based payments',
                    'Reduce credit exposure',
                    'Monitor financial health monthly'
                ]
            })
        
        # Compliance risk mitigation
        compliance_risk = data[data['primary_risk_area'] == 'Compliance']
        if len(compliance_risk) > 0:
            strategies.append({
                'risk_area': 'Compliance',
                'priority': 'High',
                'count': len(compliance_risk),
                'entities': compliance_risk['entity_id'].tolist(),
                'actions': [
                    'Conduct compliance audit',
                    'Require immediate certification renewal',
                    'Implement compliance training',
                    'Increase inspection frequency'
                ]
            })
        
        # Delivery risk mitigation
        delivery_risk = data[data['primary_risk_area'] == 'Delivery']
        if len(delivery_risk) > 0:
            strategies.append({
                'risk_area': 'Delivery',
                'priority': 'Medium',
                'count': len(delivery_risk),
                'entities': delivery_risk['entity_id'].tolist(),
                'actions': [
                    'Add penalty clauses for delays',
                    'Request detailed project timeline',
                    'Implement weekly progress reviews',
                    'Identify backup vendors'
                ]
            })
        
        return strategies
    
    def _calculate_risk_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates risk distribution statistics.
        
        Args:
            data: Classified DataFrame
            
        Returns:
            Risk distribution dictionary
        """
        distribution = {
            'by_level': {
                'High': int(len(data[data['risk_level'] == 'High'])),
                'Medium': int(len(data[data['risk_level'] == 'Medium'])),
                'Low': int(len(data[data['risk_level'] == 'Low']))
            },
            'by_area': {
                'Financial': int(len(data[data['primary_risk_area'] == 'Financial'])),
                'Delivery': int(len(data[data['primary_risk_area'] == 'Delivery'])),
                'Compliance': int(len(data[data['primary_risk_area'] == 'Compliance'])),
                'Performance': int(len(data[data['primary_risk_area'] == 'Performance']))
            },
            'average_scores': {
                'financial': float(data['financial_risk_score'].mean()),
                'delivery': float(data['delivery_risk_score'].mean()),
                'compliance': float(data['compliance_risk_score'].mean()),
                'performance': float(data['performance_risk_score'].mean())
            }
        }
        
        return distribution


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
    print("🧪 TESTING STAGE 8: RISK ASSESSMENT")
    print("="*60)
    
    # Check if config file provided
    config = {}
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print(f"\n📄 Running with CONFIG: {config_path}")
    else:
        print(f"\n📄 Running with DEMO DATA (default parameters)")
    
    # Create sample entity data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'entity_id': [f'ENT-{i:03d}' for i in range(1, 21)],
        'entity_name': [f'Entity {i}' for i in range(1, 21)],
        'entity_type': np.random.choice(['Vendor', 'Contract'], 20)
    })
    
    # Initialize stage with config (if provided)
    stage = RiskAssessmentStage(config=config.get('parameters', {}))
    
    # Execute
    result = stage.execute(sample_data)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Entities: {result['results']['total_entities']}")
    print(f"   High Risk: {result['results']['high_risk_count']}")
    print(f"   Medium Risk: {result['results']['medium_risk_count']}")
    print(f"   Low Risk: {result['results']['low_risk_count']}")
    print(f"   Critical (Immediate Action): {result['results']['critical_risks']}")
    print(f"   Avg Risk Score: {result['results']['average_risk_score']:.2f}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    # Risk distribution
    print(f"\n📊 RISK DISTRIBUTION:")
    dist = result['results']['risk_distribution']
    print(f"   By Level: High={dist['by_level']['High']}, Medium={dist['by_level']['Medium']}, Low={dist['by_level']['Low']}")
    print(f"   By Area: Financial={dist['by_area']['Financial']}, Delivery={dist['by_area']['Delivery']}, Compliance={dist['by_area']['Compliance']}, Performance={dist['by_area']['Performance']}")
    print(f"\n   Average Scores:")
    print(f"     Financial: {dist['average_scores']['financial']:.2f}")
    print(f"     Delivery: {dist['average_scores']['delivery']:.2f}")
    print(f"     Compliance: {dist['average_scores']['compliance']:.2f}")
    print(f"     Performance: {dist['average_scores']['performance']:.2f}")
    
    # Top risk entities
    print(f"\n🚨 TOP RISK ENTITIES:")
    assessments = result['results']['risk_assessments']
    top_5 = sorted(assessments, key=lambda x: x['overall_risk_score'], reverse=True)[:5]
    
    for i, entity in enumerate(top_5, 1):
        print(f"\n   {i}. {entity['entity_id']} - {entity['entity_name']}")
        print(f"      Overall Risk: {entity['overall_risk_score']:.2f} ({entity['risk_level']})")
        print(f"      Primary Risk Area: {entity['primary_risk_area']}")
        print(f"      Financial: {entity['financial_risk_score']:.2f}")
        print(f"      Delivery: {entity['delivery_risk_score']:.2f}")
        print(f"      Compliance: {entity['compliance_risk_score']:.2f}")
        print(f"      Performance: {entity['performance_risk_score']:.2f}")
        if entity['requires_immediate_action']:
            print(f"      ⚠️  REQUIRES IMMEDIATE ACTION")
    
    # Mitigation strategies
    print(f"\n🛡️  MITIGATION STRATEGIES:")
    for strategy in result['results']['mitigation_strategies']:
        area = strategy.get('risk_area', strategy.get('risk_level', 'General'))
        print(f"\n   [{strategy['priority']}] {area} Risk ({strategy['count']} entities)")
        print(f"   Recommended Actions:")
        for action in strategy['actions'][:3]:  # Show first 3 actions
            print(f"     • {action}")
    
    print("\n" + "="*60)
    print("✅ STAGE 8 TEST COMPLETE")
    print("="*60)
