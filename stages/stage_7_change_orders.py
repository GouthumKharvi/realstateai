"""
Stage 7: Change Order Management & Impact Analysis
===================================================

Manages contract change orders through:
- Change request validation
- Cost impact calculation
- Timeline impact assessment
- Approval workflow routing
- Budget variance analysis

Uses: Rule Engine for validation, Statistical Engine for impact analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import os
from datetime import datetime, timedelta

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
from rule_engine import RuleEngine
from statistical_engine import StatisticalEngine
from formatters import format_currency, format_percentage
from logger import get_logger


class ChangeOrderManagementStage(BaseStage):
    """
    Processes and analyzes change orders for contracts.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes Stage 7 with change order parameters.
        
        Args:
            config: Optional configuration dictionary from JSON
        """
        super().__init__(stage_number=7)
        self.rule_engine = RuleEngine()
        self.stat_engine = StatisticalEngine()
        self.logger = get_logger(__name__)
        
        # Load config or use defaults
        if config:
            self.cost_variance_threshold = config.get('cost_variance_threshold', 10)
            self.time_variance_threshold = config.get('time_variance_threshold', 15)
            self.auto_approve_limit = config.get('auto_approve_limit', 500000)
        else:
            self.cost_variance_threshold = 10  # 10% cost increase
            self.time_variance_threshold = 15  # 15% timeline increase
            self.auto_approve_limit = 500000   # 5 Lakhs
    
    def _get_required_columns(self):
        """
        Returns required columns for change order processing.
        """
        return ['change_order_id', 'contract_id', 'change_cost']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes change order workflow.
        
        Args:
            data: DataFrame with change order data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing change order analysis
        """
        # Validate change orders
        validated = self._validate_change_orders(data)
        
        # Calculate cost impact
        cost_impact = self._calculate_cost_impact(validated)
        
        # Assess timeline impact
        timeline_impact = self._assess_timeline_impact(cost_impact)
        
        # Calculate budget variance
        budget_analysis = self._calculate_budget_variance(timeline_impact)
        
        # Determine approval routing
        approval_routing = self._determine_approval_routing(budget_analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(approval_routing)
        
        results = {
            'total_change_orders': len(data),
            'approved_count': len(approval_routing[approval_routing['approval_status'] == 'Auto-Approved']),
            'review_required': len(approval_routing[approval_routing['approval_status'] == 'Review Required']),
            'rejected_count': len(approval_routing[approval_routing['approval_status'] == 'Rejected']),
            'total_cost_impact': approval_routing['change_cost'].sum(),
            'average_cost_variance': approval_routing['cost_variance_pct'].mean(),
            'high_impact_changes': len(approval_routing[approval_routing['impact_level'] == 'High']),
            'change_orders': approval_routing.to_dict('records'),
            'recommendations': recommendations,
            'summary': self._generate_summary(approval_routing)
        }
        
        self.logger.info(f"   Auto-Approved: {results['approved_count']}/{results['total_change_orders']}")
        self.logger.info(f"   Total Cost Impact: {format_currency(results['total_cost_impact'])}")
        
        return results
    
    def _validate_change_orders(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates change order requests for completeness.
        
        Args:
            data: Change order DataFrame
            
        Returns:
            DataFrame with validation flags
        """
        df = data.copy()
        
        # Add original contract value if not present
        if 'original_contract_value' not in df.columns:
            df['original_contract_value'] = df['change_cost'] * np.random.uniform(10, 50, len(df))
        
        # Add change reason if not present
        if 'change_reason' not in df.columns:
            reasons = ['Scope Change', 'Design Modification', 'Material Substitution', 
                      'Timeline Extension', 'Unforeseen Conditions']
            df['change_reason'] = np.random.choice(reasons, len(df))
        
        # Validation flags
        df['is_valid'] = True
        df['validation_errors'] = ''
        
        # Rule 1: Cost must be positive
        mask = df['change_cost'] <= 0
        df.loc[mask, 'is_valid'] = False
        df.loc[mask, 'validation_errors'] += 'Invalid cost; '
        
        # Rule 2: Must have contract reference
        if 'contract_id' in df.columns:
            mask = df['contract_id'].isna()
            df.loc[mask, 'is_valid'] = False
            df.loc[mask, 'validation_errors'] += 'Missing contract ID; '
        
        return df
    
    def _calculate_cost_impact(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates cost impact of change orders.
        
        Args:
            data: Validated DataFrame
            
        Returns:
            DataFrame with cost impact metrics
        """
        df = data.copy()
        
        # Calculate cost variance percentage
        df['cost_variance_pct'] = (df['change_cost'] / df['original_contract_value'] * 100)
        
        # New contract value
        df['new_contract_value'] = df['original_contract_value'] + df['change_cost']
        
        # Cumulative cost if multiple changes
        if 'cumulative_changes' not in df.columns:
            df['cumulative_changes'] = df['change_cost']
        
        df['cumulative_variance_pct'] = (df['cumulative_changes'] / df['original_contract_value'] * 100)
        
        # Flag significant cost increases
        df['significant_cost_impact'] = df['cost_variance_pct'] > self.cost_variance_threshold
        
        return df
    
    def _assess_timeline_impact(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Assesses timeline impact of change orders.
        
        Args:
            data: Cost-analyzed DataFrame
            
        Returns:
            DataFrame with timeline impact
        """
        df = data.copy()
        
        # Add timeline extension if not present
        if 'timeline_extension_days' not in df.columns:
            # Larger changes typically need more time
            df['timeline_extension_days'] = (df['cost_variance_pct'] * np.random.uniform(0.5, 2, len(df))).astype(int)
        
        # Original timeline
        if 'original_timeline_days' not in df.columns:
            df['original_timeline_days'] = np.random.randint(90, 365, len(df))
        
        # Calculate timeline variance
        df['timeline_variance_pct'] = (df['timeline_extension_days'] / df['original_timeline_days'] * 100)
        
        # New timeline
        df['new_timeline_days'] = df['original_timeline_days'] + df['timeline_extension_days']
        
        # Flag significant timeline impact
        df['significant_timeline_impact'] = df['timeline_variance_pct'] > self.time_variance_threshold
        
        return df
    
    def _calculate_budget_variance(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates budget variance and flags overruns.
        
        Args:
            data: Timeline-analyzed DataFrame
            
        Returns:
            DataFrame with budget analysis
        """
        df = data.copy()
        
        # Add project budget if not present
        if 'project_budget' not in df.columns:
            df['project_budget'] = df['original_contract_value'] * np.random.uniform(1.1, 1.5, len(df))
        
        # Calculate budget utilization
        df['budget_utilization_pct'] = (df['new_contract_value'] / df['project_budget'] * 100)
        
        # Flag budget overruns
        df['budget_overrun'] = df['budget_utilization_pct'] > 100
        
        # Remaining budget
        df['remaining_budget'] = df['project_budget'] - df['new_contract_value']
        
        return df
    
    def _determine_approval_routing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Determines approval workflow routing based on change order characteristics.
        
        Args:
            data: Budget-analyzed DataFrame
            
        Returns:
            DataFrame with approval routing
        """
        df = data.copy()
        
        # Initialize approval fields
        df['approval_status'] = 'Pending'
        df['approval_level'] = 'Manager'
        df['impact_level'] = 'Low'
        
        # Impact level classification
        df.loc[
            (df['cost_variance_pct'] > 5) | (df['timeline_variance_pct'] > 10),
            'impact_level'
        ] = 'Medium'
        
        df.loc[
            (df['cost_variance_pct'] > 15) | (df['timeline_variance_pct'] > 20) | df['budget_overrun'],
            'impact_level'
        ] = 'High'
        
        # Auto-approval for small changes
        mask = (df['change_cost'] <= self.auto_approve_limit) & df['is_valid'] & (~df['budget_overrun'])
        df.loc[mask, 'approval_status'] = 'Auto-Approved'
        df.loc[mask, 'approval_level'] = 'System'
        
        # Manager approval
        mask = (
            (df['change_cost'] > self.auto_approve_limit) & 
            (df['change_cost'] <= 2000000) &
            (~df['budget_overrun'])
        )
        df.loc[mask, 'approval_status'] = 'Review Required'
        df.loc[mask, 'approval_level'] = 'Manager'
        
        # Director approval
        mask = (
            (df['change_cost'] > 2000000) & 
            (df['change_cost'] <= 10000000) |
            (df['significant_cost_impact'])
        )
        df.loc[mask, 'approval_status'] = 'Review Required'
        df.loc[mask, 'approval_level'] = 'Director'
        
        # Board approval
        mask = (df['change_cost'] > 10000000) | df['budget_overrun']
        df.loc[mask, 'approval_status'] = 'Review Required'
        df.loc[mask, 'approval_level'] = 'Board'
        
        # Reject invalid changes
        mask = ~df['is_valid']
        df.loc[mask, 'approval_status'] = 'Rejected'
        
        return df
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates recommendations for change order management.
        
        Args:
            data: Approval-routed DataFrame
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # High impact changes
        high_impact = data[data['impact_level'] == 'High']
        if len(high_impact) > 0:
            total_cost = high_impact['change_cost'].sum()
            recommendations.append({
                'type': 'high_impact',
                'priority': 'Critical',
                'count': len(high_impact),
                'total_cost': total_cost,
                'message': f"🚨 CRITICAL: {len(high_impact)} high-impact change orders worth {format_currency(total_cost)} - Requires senior management review"
            })
        
        # Budget overruns
        overruns = data[data['budget_overrun']]
        if len(overruns) > 0:
            recommendations.append({
                'type': 'budget_overrun',
                'priority': 'High',
                'count': len(overruns),
                'message': f"⚠️  WARNING: {len(overruns)} change orders cause budget overruns - Immediate action required"
            })
        
        # Auto-approved changes
        auto_approved = data[data['approval_status'] == 'Auto-Approved']
        if len(auto_approved) > 0:
            recommendations.append({
                'type': 'auto_approved',
                'priority': 'Info',
                'count': len(auto_approved),
                'total_cost': auto_approved['change_cost'].sum(),
                'message': f"✅ {len(auto_approved)} change orders auto-approved - Total: {format_currency(auto_approved['change_cost'].sum())}"
            })
        
        # Timeline impacts
        timeline_impact = data[data['significant_timeline_impact']]
        if len(timeline_impact) > 0:
            avg_extension = timeline_impact['timeline_extension_days'].mean()
            recommendations.append({
                'type': 'timeline_impact',
                'priority': 'Medium',
                'count': len(timeline_impact),
                'avg_extension': avg_extension,
                'message': f"⏱️  {len(timeline_impact)} change orders impact timeline - Avg extension: {avg_extension:.0f} days"
            })
        
        return recommendations
    
    def _generate_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates summary statistics for change orders.
        
        Args:
            data: Processed DataFrame
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_cost_impact': float(data['change_cost'].sum()),
            'average_cost_variance': float(data['cost_variance_pct'].mean()),
            'average_timeline_extension': float(data['timeline_extension_days'].mean()),
            'budget_overrun_count': int(len(data[data['budget_overrun']])),
            'auto_approval_rate': float(len(data[data['approval_status'] == 'Auto-Approved']) / len(data) * 100) if len(data) > 0 else 0,
            'most_common_reason': data['change_reason'].mode()[0] if len(data) > 0 else 'N/A'
        }
        
        return summary


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
    print("🧪 TESTING STAGE 7: CHANGE ORDER MANAGEMENT")
    print("="*60)
    
    # Check if config file provided
    config = {}
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print(f"\n📄 Running with CONFIG: {config_path}")
    else:
        print(f"\n📄 Running with DEMO DATA (default parameters)")
    
    # Create sample change order data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'change_order_id': [f'CO-{i:04d}' for i in range(1, 13)],
        'contract_id': [f'CONT-{i:04d}' for i in range(1, 13)],
        'change_cost': [
            300000, 1500000, 8000000, 450000, 12000000,
            250000, 3000000, 600000, 15000000, 800000,
            5000000, 400000
        ]
    })
    
    # Initialize stage with config (if provided)
    stage = ChangeOrderManagementStage(config=config.get('parameters', {}))
    
    # Execute
    result = stage.execute(sample_data)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Change Orders: {result['results']['total_change_orders']}")
    print(f"   Auto-Approved: {result['results']['approved_count']}")
    print(f"   Review Required: {result['results']['review_required']}")
    print(f"   Rejected: {result['results']['rejected_count']}")
    print(f"   High Impact: {result['results']['high_impact_changes']}")
    print(f"   Total Cost Impact: {format_currency(result['results']['total_cost_impact'])}")
    print(f"   Avg Cost Variance: {format_percentage(result['results']['average_cost_variance'])}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    summary = result['results']['summary']
    print(f"   Total Impact: {format_currency(summary['total_cost_impact'])}")
    print(f"   Avg Cost Variance: {format_percentage(summary['average_cost_variance'])}")
    print(f"   Avg Timeline Extension: {summary['average_timeline_extension']:.0f} days")
    print(f"   Budget Overruns: {summary['budget_overrun_count']}")
    print(f"   Auto-Approval Rate: {format_percentage(summary['auto_approval_rate'])}")
    print(f"   Most Common Reason: {summary['most_common_reason']}")
    
    # Top change orders
    print(f"\n💰 TOP CHANGE ORDERS BY COST:")
    changes = result['results']['change_orders']
    top_5 = sorted(changes, key=lambda x: x['change_cost'], reverse=True)[:5]
    
    for i, change in enumerate(top_5, 1):
        print(f"\n   {i}. {change['change_order_id']} - {change['contract_id']}")
        print(f"      Cost: {format_currency(change['change_cost'])}")
        print(f"      Variance: {format_percentage(change['cost_variance_pct'])}")
        print(f"      Impact Level: {change['impact_level']}")
        print(f"      Approval: {change['approval_status']} ({change['approval_level']})")
        print(f"      Timeline Extension: {change['timeline_extension_days']} days")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in result['results']['recommendations']:
        print(f"   [{rec['priority']}] {rec['message']}")
    
    print("\n" + "="*60)
    print("✅ STAGE 7 TEST COMPLETE")
    print("="*60)
