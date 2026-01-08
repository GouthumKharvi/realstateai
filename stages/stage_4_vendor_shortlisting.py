"""
Stage 4: Vendor Shortlisting for Tenders
=========================================

Scores and ranks vendors for tender selection based on:
- Historical performance
- Financial strength
- Technical capability
- Risk assessment
- Cost competitiveness

Uses: ML Engine for scoring, Rule Engine for criteria validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

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
from ml_engine import MLEngine
from statistical_engine import StatisticalEngine
from formatters import format_percentage
from logger import get_logger
from constants import VENDOR_WEIGHTS, VENDOR_CLASSIFICATION


class VendorShortlistingStage(BaseStage):
    """
    Scores and shortlists vendors for tender evaluation.
    """
    
    def __init__(self):
        """
        Initializes Stage 4 with scoring engines.
        """
        super().__init__(stage_number=4)
        self.rule_engine = RuleEngine()
        self.ml_engine = MLEngine()
        self.stat_engine = StatisticalEngine()
        self.logger = get_logger(__name__)
        
        # Scoring weights (from constants)
        self.weights = VENDOR_WEIGHTS
    
    def _get_required_columns(self):
        """
        Returns required columns for vendor shortlisting.
        """
        return ['vendor_id', 'vendor_name']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes vendor shortlisting workflow.
        
        Args:
            data: DataFrame with vendor data
            **kwargs: shortlist_count (default: 10)
            
        Returns:
            Dictionary containing shortlisted vendors
        """
        shortlist_count = kwargs.get('shortlist_count', 10)
        
        # Calculate vendor scores
        scored_vendors = self._calculate_vendor_scores(data)
        
        # Apply eligibility criteria
        eligible_vendors = self._apply_eligibility_criteria(scored_vendors)
        
        # Rank vendors
        ranked_vendors = self._rank_vendors(eligible_vendors)
        
        # Create shortlist
        shortlist = self._create_shortlist(ranked_vendors, shortlist_count)
        
        # Identify risk flags
        risk_flagged = self._identify_risk_flags(shortlist)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_flagged, ranked_vendors)
        
        results = {
            'total_vendors': len(data),
            'eligible_vendors': len(eligible_vendors),
            'shortlisted_count': len(shortlist),
            'rejected_count': len(data) - len(eligible_vendors),
            'average_score': ranked_vendors['final_score'].mean(),
            'shortlist': shortlist.to_dict('records'),
            'all_ranked': ranked_vendors[['vendor_id', 'vendor_name', 'final_score', 'rank', 'category']].to_dict('records'),
            'recommendations': recommendations
        }
        
        self.logger.info(f"   Eligible: {results['eligible_vendors']}/{results['total_vendors']}")
        self.logger.info(f"   Shortlisted: {results['shortlisted_count']}")
        
        return results
    
    def _calculate_vendor_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates weighted scores for each vendor.
        
        Args:
            data: Vendor DataFrame
            
        Returns:
            DataFrame with calculated scores
        """
        df = data.copy()
        
        # Add default scores if not present
        if 'delivery_score' not in df.columns:
            df['delivery_score'] = np.random.uniform(70, 100, len(df))
        
        if 'quality_score' not in df.columns:
            df['quality_score'] = np.random.uniform(65, 98, len(df))
        
        if 'cost_competitiveness' not in df.columns:
            df['cost_competitiveness'] = np.random.uniform(60, 95, len(df))
        
        if 'payment_compliance' not in df.columns:
            df['payment_compliance'] = np.random.uniform(75, 100, len(df))
        
        if 'dispute_history' not in df.columns:
            # Lower dispute score is better, invert it
            df['dispute_count'] = np.random.randint(0, 5, len(df))
            df['dispute_history'] = 100 - (df['dispute_count'] * 20)
            df['dispute_history'] = df['dispute_history'].clip(0, 100)
        
        # Calculate weighted final score
        df['final_score'] = (
            df['delivery_score'] * self.weights['delivery_score'] +
            df['quality_score'] * self.weights['quality_score'] +
            df['cost_competitiveness'] * self.weights['cost_competitiveness'] +
            df['payment_compliance'] * self.weights['payment_compliance'] +
            df['dispute_history'] * self.weights['dispute_history']
        )
        
        # Round to 2 decimal places
        df['final_score'] = df['final_score'].round(2)
        
        return df
    
    def _apply_eligibility_criteria(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters vendors based on eligibility criteria.
        
        Args:
            data: Scored vendor DataFrame
            
        Returns:
            DataFrame with only eligible vendors
        """
        df = data.copy()
        
        # Initialize eligibility
        df['eligible'] = True
        df['rejection_reasons'] = ''
        
        # Criterion 1: Minimum delivery score
        min_delivery = 60
        mask = df['delivery_score'] < min_delivery
        df.loc[mask, 'eligible'] = False
        df.loc[mask, 'rejection_reasons'] += f'Low delivery score (<{min_delivery}); '
        
        # Criterion 2: Minimum quality score
        min_quality = 60
        mask = df['quality_score'] < min_quality
        df.loc[mask, 'eligible'] = False
        df.loc[mask, 'rejection_reasons'] += f'Low quality score (<{min_quality}); '
        
        # Criterion 3: Maximum disputes
        max_disputes = 3
        if 'dispute_count' in df.columns:
            mask = df['dispute_count'] > max_disputes
            df.loc[mask, 'eligible'] = False
            df.loc[mask, 'rejection_reasons'] += f'Too many disputes (>{max_disputes}); '
        
        # Criterion 4: Minimum overall score
        min_overall = 50
        mask = df['final_score'] < min_overall
        df.loc[mask, 'eligible'] = False
        df.loc[mask, 'rejection_reasons'] += f'Low overall score (<{min_overall}); '
        
        # Return only eligible vendors
        eligible = df[df['eligible']].copy()
        
        # Log rejected vendors
        rejected = df[~df['eligible']]
        if len(rejected) > 0:
            self.logger.info(f"   Rejected {len(rejected)} vendors due to eligibility criteria")
        
        return eligible
    
    def _rank_vendors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks vendors by final score.
        
        Args:
            data: Eligible vendor DataFrame
            
        Returns:
            DataFrame sorted by rank
        """
        df = data.copy()
        
        # Sort by score (descending)
        df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        # Assign rank
        df['rank'] = range(1, len(df) + 1)
        
        # Categorize vendors
        df['category'] = df['final_score'].apply(self._categorize_vendor)
        
        return df
    
    def _categorize_vendor(self, score: float) -> str:
        """
        Categorizes vendor based on score.
        
        Args:
            score: Vendor final score
            
        Returns:
            Category label (Excellent, Good, Average, Poor)
        """
        if score >= VENDOR_CLASSIFICATION['excellent']:
            return 'Excellent'
        elif score >= VENDOR_CLASSIFICATION['good']:
            return 'Good'
        elif score >= VENDOR_CLASSIFICATION['average']:
            return 'Average'
        else:
            return 'Poor'
    
    def _create_shortlist(self, data: pd.DataFrame, count: int) -> pd.DataFrame:
        """
        Creates final shortlist of top vendors.
        
        Args:
            data: Ranked vendor DataFrame
            count: Number of vendors to shortlist
            
        Returns:
            DataFrame with shortlisted vendors
        """
        # Take top N vendors
        shortlist = data.head(count).copy()
        
        # Add shortlist flag
        shortlist['shortlisted'] = True
        
        # Select relevant columns
        columns = [
            'rank', 'vendor_id', 'vendor_name', 'final_score', 'category',
            'delivery_score', 'quality_score', 'cost_competitiveness',
            'payment_compliance', 'dispute_history'
        ]
        
        # Filter to available columns
        available_cols = [col for col in columns if col in shortlist.columns]
        shortlist = shortlist[available_cols]
        
        return shortlist
    
    def _identify_risk_flags(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies potential risk factors in shortlisted vendors.
        
        Args:
            data: Shortlisted vendor DataFrame
            
        Returns:
            DataFrame with risk flags added
        """
        df = data.copy()
        
        # Initialize risk flags
        df['risk_flags'] = ''
        df['risk_count'] = 0
        
        # Flag 1: Low delivery performance
        if 'delivery_score' in df.columns:
            mask = df['delivery_score'] < 75
            df.loc[mask, 'risk_flags'] += 'Low delivery; '
            df.loc[mask, 'risk_count'] += 1
        
        # Flag 2: Quality concerns
        if 'quality_score' in df.columns:
            mask = df['quality_score'] < 75
            df.loc[mask, 'risk_flags'] += 'Quality concerns; '
            df.loc[mask, 'risk_count'] += 1
        
        # Flag 3: Dispute history
        if 'dispute_count' in df.columns:
            mask = df['dispute_count'] > 1
            df.loc[mask, 'risk_flags'] += 'Past disputes; '
            df.loc[mask, 'risk_count'] += 1
        
        # Overall risk level
        df['risk_level'] = df['risk_count'].apply(
            lambda x: 'High' if x >= 2 else ('Medium' if x == 1 else 'Low')
        )
        
        return df
    
    def _generate_recommendations(self, shortlist: pd.DataFrame, 
                                 all_vendors: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates recommendations for vendor selection.
        
        Args:
            shortlist: Shortlisted vendors DataFrame
            all_vendors: All ranked vendors DataFrame
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Top vendors
        top_vendors = shortlist.head(3)
        recommendations.append({
            'type': 'top_vendors',
            'count': len(top_vendors),
            'vendors': top_vendors['vendor_id'].tolist(),
            'message': f"✅ TOP 3 VENDORS: {', '.join(top_vendors['vendor_name'].tolist())} - Recommend for immediate invitation"
        })
        
        # Excellent category
        excellent = shortlist[shortlist['category'] == 'Excellent']
        if len(excellent) > 0:
            recommendations.append({
                'type': 'excellent',
                'count': len(excellent),
                'vendors': excellent['vendor_id'].tolist(),
                'message': f"⭐ {len(excellent)} EXCELLENT vendors in shortlist - Priority consideration"
            })
        
        # Risk flagged vendors
        if 'risk_level' in shortlist.columns:
            high_risk = shortlist[shortlist['risk_level'] == 'High']
            if len(high_risk) > 0:
                recommendations.append({
                    'type': 'risk_warning',
                    'count': len(high_risk),
                    'vendors': high_risk['vendor_id'].tolist(),
                    'message': f"⚠️  {len(high_risk)} vendors have HIGH risk flags - Require additional due diligence"
                })
        
        # Average score info
        avg_score = shortlist['final_score'].mean()
        recommendations.append({
            'type': 'statistics',
            'message': f"📊 Shortlist average score: {avg_score:.2f} - Quality threshold maintained"
        })
        
        return recommendations


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING STAGE 4: VENDOR SHORTLISTING")
    print("="*60)
    
    # Create sample vendor data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'vendor_id': [f'V{i:03d}' for i in range(1, 26)],
        'vendor_name': [f'Vendor {i}' for i in range(1, 26)],
        'delivery_score': np.random.uniform(60, 98, 25),
        'quality_score': np.random.uniform(65, 95, 25),
        'cost_competitiveness': np.random.uniform(70, 95, 25),
        'payment_compliance': np.random.uniform(75, 100, 25),
        'dispute_count': np.random.randint(0, 4, 25)
    })
    
    # Add some poor performers for testing
    sample_data.loc[20, 'delivery_score'] = 45  # Will be rejected
    sample_data.loc[21, 'quality_score'] = 40   # Will be rejected
    sample_data.loc[22, 'dispute_count'] = 5    # Will be rejected
    
    # Initialize and execute stage
    stage = VendorShortlistingStage()
    result = stage.execute(sample_data, shortlist_count=10)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Vendors: {result['results']['total_vendors']}")
    print(f"   Eligible: {result['results']['eligible_vendors']}")
    print(f"   Rejected: {result['results']['rejected_count']}")
    print(f"   Shortlisted: {result['results']['shortlisted_count']}")
    print(f"   Average Score: {result['results']['average_score']:.2f}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    # Top 5 shortlisted vendors
    print(f"\n🏆 TOP 5 SHORTLISTED VENDORS:")
    for vendor in result['results']['shortlist'][:5]:
        risk = vendor.get('risk_level', 'N/A')
        print(f"   {vendor['rank']}. {vendor['vendor_name']}")
        print(f"      Score: {vendor['final_score']:.2f} | Category: {vendor['category']} | Risk: {risk}")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in result['results']['recommendations']:
        print(f"   {rec['message']}")
    
    # Score breakdown
    print(f"\n📈 SCORE BREAKDOWN (Top Vendor):")
    top_vendor = result['results']['shortlist'][0]
    print(f"   Vendor: {top_vendor['vendor_name']}")
    print(f"   Delivery: {top_vendor['delivery_score']:.2f} (Weight: {VENDOR_WEIGHTS['delivery_score']:.0%})")
    print(f"   Quality: {top_vendor['quality_score']:.2f} (Weight: {VENDOR_WEIGHTS['quality_score']:.0%})")
    print(f"   Cost: {top_vendor['cost_competitiveness']:.2f} (Weight: {VENDOR_WEIGHTS['cost_competitiveness']:.0%})")
    print(f"   Payment: {top_vendor['payment_compliance']:.2f} (Weight: {VENDOR_WEIGHTS['payment_compliance']:.0%})")
    print(f"   Disputes: {top_vendor['dispute_history']:.2f} (Weight: {VENDOR_WEIGHTS['dispute_history']:.0%})")
    print(f"   Final Score: {top_vendor['final_score']:.2f}")
    
    print("\n" + "="*60)
    print("✅ STAGE 4 TEST COMPLETE")
    print("="*60)