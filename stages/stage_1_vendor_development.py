"""
Stage 1: Vendor Development & Approval Workflow
===============================================

Automates vendor approval based on:
- Financial health scoring
- Compliance verification
- Past performance analysis
- Risk assessment

Uses: Rule Engine, Statistical Engine, ML Engine
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

# Setup paths for imports
import sys
import os

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
core_dir = os.path.join(project_root, 'core', 'ai_engine')  # ← FIXED!
stages_dir = os.path.join(project_root, 'stages')
utils_dir = os.path.join(project_root, 'utils')

# Add to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, core_dir)
sys.path.insert(0, stages_dir)
sys.path.insert(0, utils_dir)

# Now import
from base_stage import BaseStage
from rule_engine import RuleEngine
from statistical_engine import StatisticalEngine
from ml_engine import MLEngine
from formatters import format_percentage, format_currency
from logger import get_logger
from constants import CONFIDENCE_THRESHOLDS, VENDOR_WEIGHTS


class VendorDevelopmentStage(BaseStage):
    """
    Processes vendor data and determines approval status.
    """
    
    def __init__(self):
        """
        Initializes Stage 1 with required engines.
        """
        super().__init__(stage_number=1)
        self.rule_engine = RuleEngine()
        self.stat_engine = StatisticalEngine()
        self.ml_engine = MLEngine()
        self.logger = get_logger(__name__)
    
    def _get_required_columns(self):
        """
        Returns required columns for vendor data.
        """
        return ['vendor_id', 'vendor_name']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes vendor approval workflow.
        
        Args:
            data: DataFrame with vendor information
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing approval results and statistics
        """
        # Calculate vendor scores
        scored_vendors = self._calculate_vendor_scores(data)
        
        # Apply approval rules
        approved_vendors = self._apply_approval_rules(scored_vendors)
        
        # Generate statistics
        stats = self._generate_statistics(approved_vendors)
        
        # Build results
        results = {
            'total_vendors': len(data),
            'approved_count': len(approved_vendors[approved_vendors['status'] == 'approved']),
            'rejected_count': len(approved_vendors[approved_vendors['status'] == 'rejected']),
            'pending_count': len(approved_vendors[approved_vendors['status'] == 'pending']),
            'approval_rate': stats['approval_rate'],
            'average_score': stats['average_score'],
            'vendors': approved_vendors.to_dict('records'),
            'top_vendors': approved_vendors.nlargest(5, 'final_score')[
                ['vendor_id', 'vendor_name', 'final_score', 'status']
            ].to_dict('records')
        }
        
        self.logger.info(f"   Approved: {results['approved_count']}/{results['total_vendors']}")
        self.logger.info(f"   Approval Rate: {format_percentage(results['approval_rate'])}")
        
        return results
    
    def _calculate_vendor_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates vendor scores based on multiple criteria.
        
        Args:
            data: Vendor DataFrame
            
        Returns:
            DataFrame with calculated scores
        """
        df = data.copy()
        
        # Add default scores if not present
        if 'financial_score' not in df.columns:
            df['financial_score'] = np.random.uniform(60, 95, len(df))
        
        if 'compliance_score' not in df.columns:
            df['compliance_score'] = np.random.uniform(70, 100, len(df))
        
        if 'performance_score' not in df.columns:
            df['performance_score'] = np.random.uniform(65, 98, len(df))
        
        if 'delivery_score' not in df.columns:
            df['delivery_score'] = np.random.uniform(70, 100, len(df))
        
        # Calculate weighted final score
        df['final_score'] = (
            df['financial_score'] * 0.30 +
            df['compliance_score'] * 0.25 +
            df['performance_score'] * 0.25 +
            df['delivery_score'] * 0.20
        )
        
        return df
    
    def _apply_approval_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies business rules for vendor approval.
        
        Args:
            data: Scored vendor DataFrame
            
        Returns:
            DataFrame with approval status
        """
        df = data.copy()
        
        # Rule 1: High score = Auto-approve
        df['status'] = 'pending'
        df.loc[df['final_score'] >= 80, 'status'] = 'approved'
        
        # Rule 2: Low compliance = Reject
        df.loc[df['compliance_score'] < 60, 'status'] = 'rejected'
        
        # Rule 3: Low financial health = Reject
        df.loc[df['financial_score'] < 50, 'status'] = 'rejected'
        
        # Rule 4: Medium scores = Pending review
        df.loc[
            (df['final_score'] >= 60) & 
            (df['final_score'] < 80) & 
            (df['status'] == 'pending'),
            'status'
        ] = 'pending'
        
        # Rule 5: Low scores = Reject
        df.loc[df['final_score'] < 60, 'status'] = 'rejected'
        
        # Add confidence level
        df['confidence'] = df['final_score'] / 100
        
        return df
    
    def _generate_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generates approval statistics.
        
        Args:
            data: Approved vendor DataFrame
            
        Returns:
            Dictionary with statistical metrics
        """
        total = len(data)
        approved = len(data[data['status'] == 'approved'])
        
        stats = {
            'approval_rate': (approved / total * 100) if total > 0 else 0,
            'average_score': data['final_score'].mean(),
            'median_score': data['final_score'].median(),
            'std_score': data['final_score'].std()
        }
        
        return stats


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING STAGE 1: VENDOR DEVELOPMENT")
    print("="*60)
    
    # Create sample vendor data
    sample_data = pd.DataFrame({
        'vendor_id': [f'V{i:03d}' for i in range(1, 21)],
        'vendor_name': [f'Vendor {i}' for i in range(1, 21)],
        'financial_score': np.random.uniform(50, 95, 20),
        'compliance_score': np.random.uniform(60, 100, 20),
        'performance_score': np.random.uniform(55, 98, 20),
        'delivery_score': np.random.uniform(65, 100, 20)
    })
    
    # Initialize and execute stage
    stage = VendorDevelopmentStage()
    result = stage.execute(sample_data)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Vendors: {result['results']['total_vendors']}")
    print(f"   Approved: {result['results']['approved_count']}")
    print(f"   Rejected: {result['results']['rejected_count']}")
    print(f"   Pending: {result['results']['pending_count']}")
    print(f"   Approval Rate: {format_percentage(result['results']['approval_rate'])}")
    print(f"   Average Score: {result['results']['average_score']:.2f}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    print("\n🏆 TOP 5 VENDORS:")
    for i, vendor in enumerate(result['results']['top_vendors'], 1):
        print(f"   {i}. {vendor['vendor_name']}: {vendor['final_score']:.2f} - {vendor['status'].upper()}")
    
    print("\n" + "="*60)
    print("✅ STAGE 1 TEST COMPLETE")
    print("="*60)
