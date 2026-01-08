"""
Stage 5: AI-Powered Negotiation Insights
=========================================

Provides negotiation support through:
- Historical price trend analysis
- Optimal counter-offer calculation
- Negotiation success probability prediction
- Risk assessment for negotiation strategies
- Market benchmark comparison

Uses: Statistical Engine, ML Engine for predictions
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
from statistical_engine import StatisticalEngine
from ml_engine import MLEngine
from formatters import format_currency, format_percentage
from logger import get_logger


class NegotiationInsightsStage(BaseStage):
    """
    Provides AI-powered negotiation recommendations and insights.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes Stage 5 with negotiation parameters.
        
        Args:
            config: Optional configuration dictionary from JSON
        """
        super().__init__(stage_number=5)
        self.stat_engine = StatisticalEngine()
        self.ml_engine = MLEngine()
        self.logger = get_logger(__name__)
        
        # Load config or use defaults
        if config:
            self.target_savings_pct = config.get('target_savings_pct', 10)
            self.max_negotiation_rounds = config.get('max_negotiation_rounds', 3)
            self.min_acceptable_discount = config.get('min_acceptable_discount', 5)
            self.market_variance_threshold = config.get('market_variance_threshold', 15)
        else:
            # Default parameters
            self.target_savings_pct = 10
            self.max_negotiation_rounds = 3
            self.min_acceptable_discount = 5
            self.market_variance_threshold = 15
    
    def _get_required_columns(self):
        """
        Returns required columns for negotiation analysis.
        """
        return ['vendor_id', 'current_price']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes negotiation insights workflow.
        
        Args:
            data: DataFrame with vendor pricing data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing negotiation recommendations
        """
        # Analyze price trends
        price_analysis = self._analyze_price_trends(data)
        
        # Calculate optimal counter-offers
        counter_offers = self._calculate_counter_offers(price_analysis)
        
        # Predict negotiation success
        predictions = self._predict_negotiation_success(counter_offers)
        
        # Assess negotiation risks
        risk_assessment = self._assess_negotiation_risks(predictions)
        
        # Generate negotiation strategies
        strategies = self._generate_negotiation_strategies(risk_assessment)
        
        # Create recommendations
        recommendations = self._create_recommendations(strategies)
        
        results = {
            'total_vendors': len(data),
            'negotiable_vendors': len(strategies[strategies['is_negotiable']]),
            'total_potential_savings': strategies['potential_savings'].sum(),
            'average_discount_achievable': strategies['recommended_discount_pct'].mean(),
            'high_success_probability': len(strategies[strategies['success_probability'] >= 0.7]),
            'negotiation_insights': strategies.to_dict('records'),
            'recommendations': recommendations,
            'market_benchmark': self._calculate_market_benchmark(data)
        }
        
        self.logger.info(f"   Negotiable: {results['negotiable_vendors']}/{results['total_vendors']}")
        self.logger.info(f"   Potential Savings: {format_currency(results['total_potential_savings'])}")
        
        return results
    
    def _analyze_price_trends(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes historical price trends for vendors.
        
        Args:
            data: Vendor pricing DataFrame
            
        Returns:
            DataFrame with trend analysis
        """
        df = data.copy()
        
        # Add historical data if not present (simulate for demo)
        if 'historical_avg_price' not in df.columns:
            # Simulate historical prices (slightly higher than current)
            df['historical_avg_price'] = df['current_price'] * np.random.uniform(1.05, 1.20, len(df))
        
        if 'market_avg_price' not in df.columns:
            # Simulate market average
            overall_avg = df['current_price'].mean()
            df['market_avg_price'] = overall_avg * np.random.uniform(0.95, 1.05, len(df))
        
        # Calculate price variance from market
        df['price_vs_market_pct'] = ((df['current_price'] - df['market_avg_price']) / 
                                      df['market_avg_price'] * 100)
        
        # Calculate historical trend
        df['price_trend_pct'] = ((df['current_price'] - df['historical_avg_price']) / 
                                  df['historical_avg_price'] * 100)
        
        # Identify if price is above market
        df['above_market'] = df['current_price'] > df['market_avg_price']
        
        return df
    
    def _calculate_counter_offers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates optimal counter-offer prices.
        
        Args:
            data: Price analysis DataFrame
            
        Returns:
            DataFrame with counter-offer calculations
        """
        df = data.copy()
        
        # Base counter-offer: Target savings percentage
        df['target_price'] = df['current_price'] * (1 - self.target_savings_pct / 100)
        
        # Adjust based on market position
        # If already below market, reduce discount expectation
        df.loc[~df['above_market'], 'target_price'] = (
            df.loc[~df['above_market'], 'current_price'] * 
            (1 - self.min_acceptable_discount / 100)
        )
        
        # Calculate recommended discount
        df['recommended_discount_pct'] = ((df['current_price'] - df['target_price']) / 
                                           df['current_price'] * 100)
        
        # Calculate potential savings
        df['potential_savings'] = df['current_price'] - df['target_price']
        
        # Conservative offer (start high, negotiate down)
        df['conservative_offer'] = df['current_price'] * (1 - self.min_acceptable_discount / 100)
        
        # Aggressive offer (maximum reasonable discount)
        df['aggressive_offer'] = df['market_avg_price'] * 0.95
        
        return df
    
    def _predict_negotiation_success(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts probability of successful negotiation.
        
        Args:
            data: Counter-offer DataFrame
            
        Returns:
            DataFrame with success predictions
        """
        df = data.copy()
        
        # Initialize success probability
        df['success_probability'] = 0.5  # Base 50%
        
        # Factor 1: Price vs market (higher above market = higher success)
        df.loc[df['price_vs_market_pct'] > 10, 'success_probability'] += 0.2
        df.loc[df['price_vs_market_pct'] > 20, 'success_probability'] += 0.1
        
        # Factor 2: Historical trend (decreasing prices = higher success)
        df.loc[df['price_trend_pct'] < 0, 'success_probability'] += 0.15
        
        # Factor 3: Discount reasonableness (smaller discount = higher success)
        df.loc[df['recommended_discount_pct'] < 10, 'success_probability'] += 0.1
        df.loc[df['recommended_discount_pct'] > 20, 'success_probability'] -= 0.2
        
        # Add vendor relationship score if available
        if 'vendor_relationship_score' not in df.columns:
            df['vendor_relationship_score'] = np.random.uniform(60, 95, len(df))
        
        # Factor 4: Vendor relationship (good relationship = higher success)
        df['success_probability'] += (df['vendor_relationship_score'] - 70) / 100
        
        # Clip to 0-1 range
        df['success_probability'] = df['success_probability'].clip(0.1, 0.95)
        
        return df
    
    def _assess_negotiation_risks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Assesses risks associated with negotiation.
        
        Args:
            data: Prediction DataFrame
            
        Returns:
            DataFrame with risk assessment
        """
        df = data.copy()
        
        # Initialize risk flags
        df['risk_flags'] = ''
        df['risk_score'] = 0
        
        # Risk 1: Vendor already below market (may not budge)
        mask = df['price_vs_market_pct'] < -5
        df.loc[mask, 'risk_flags'] += 'Already below market; '
        df.loc[mask, 'risk_score'] += 30
        
        # Risk 2: Aggressive discount requested
        mask = df['recommended_discount_pct'] > 15
        df.loc[mask, 'risk_flags'] += 'Aggressive discount; '
        df.loc[mask, 'risk_score'] += 25
        
        # Risk 3: Low vendor relationship score
        mask = df['vendor_relationship_score'] < 70
        df.loc[mask, 'risk_flags'] += 'Weak relationship; '
        df.loc[mask, 'risk_score'] += 20
        
        # Risk 4: Prices already declining
        mask = df['price_trend_pct'] < -10
        df.loc[mask, 'risk_flags'] += 'Price declining trend; '
        df.loc[mask, 'risk_score'] += 15
        
        # Classify risk level
        df['risk_level'] = 'Low'
        df.loc[df['risk_score'] >= 30, 'risk_level'] = 'Medium'
        df.loc[df['risk_score'] >= 50, 'risk_level'] = 'High'
        
        return df
    
    def _generate_negotiation_strategies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates specific negotiation strategies for each vendor.
        
        Args:
            data: Risk assessment DataFrame
            
        Returns:
            DataFrame with negotiation strategies
        """
        df = data.copy()
        
        # Determine if negotiation is worthwhile
        df['is_negotiable'] = (
            (df['potential_savings'] > 0) & 
            (df['success_probability'] >= 0.3) &
            (df['risk_level'] != 'High')
        )
        
        # Generate strategy text
        df['strategy'] = ''
        
        # High success probability
        mask = (df['success_probability'] >= 0.7) & df['is_negotiable']
        df.loc[mask, 'strategy'] = (
            'Start with aggressive offer at ' + 
            df.loc[mask, 'aggressive_offer'].apply(lambda x: format_currency(x)) +
            '. High success probability.'
        )
        
        # Medium success probability
        mask = (df['success_probability'] >= 0.4) & (df['success_probability'] < 0.7) & df['is_negotiable']
        df.loc[mask, 'strategy'] = (
            'Start with conservative offer at ' + 
            df.loc[mask, 'conservative_offer'].apply(lambda x: format_currency(x)) +
            '. Negotiate gradually over ' + str(self.max_negotiation_rounds) + ' rounds.'
        )
        
        # Low success probability
        mask = (df['success_probability'] < 0.4) & df['is_negotiable']
        df.loc[mask, 'strategy'] = (
            'Limited negotiation potential. Focus on non-price terms (payment, delivery). ' +
            'Maximum discount: ' + df.loc[mask, 'min_acceptable_discount'].apply(lambda x: f"{x}%") if 'min_acceptable_discount' in df.columns else '5%'
        )
        
        # Not negotiable
        mask = ~df['is_negotiable']
        df.loc[mask, 'strategy'] = 'Not recommended for negotiation. Accept current price or seek alternatives.'
        
        # Add recommended approach
        df['recommended_approach'] = df.apply(self._determine_approach, axis=1)
        
        return df
    
    def _determine_approach(self, row: pd.Series) -> str:
        """
        Determines negotiation approach based on vendor profile.
        
        Args:
            row: Vendor data row
            
        Returns:
            Recommended approach string
        """
        if not row['is_negotiable']:
            return 'No Negotiation'
        
        if row['success_probability'] >= 0.7:
            return 'Aggressive Negotiation'
        elif row['success_probability'] >= 0.5:
            return 'Standard Negotiation'
        else:
            return 'Conservative Negotiation'
    
    def _create_recommendations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Creates actionable recommendations.
        
        Args:
            data: Strategy DataFrame
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # High-value negotiations
        high_value = data[
            (data['is_negotiable']) & 
            (data['potential_savings'] > data['potential_savings'].quantile(0.75))
        ].nlargest(3, 'potential_savings')
        
        if len(high_value) > 0:
            total_savings = high_value['potential_savings'].sum()
            recommendations.append({
                'type': 'high_value',
                'priority': 'High',
                'vendors': high_value['vendor_id'].tolist(),
                'message': f"💰 TOP PRIORITY: Negotiate with {len(high_value)} vendors for potential savings of {format_currency(total_savings)}"
            })
        
        # Quick wins (high success, moderate savings)
        quick_wins = data[
            (data['success_probability'] >= 0.7) & 
            (data['is_negotiable'])
        ]
        
        if len(quick_wins) > 0:
            recommendations.append({
                'type': 'quick_wins',
                'priority': 'Medium',
                'vendors': quick_wins['vendor_id'].tolist(),
                'message': f"✅ QUICK WINS: {len(quick_wins)} vendors with high success probability (≥70%)"
            })
        
        # Risky negotiations
        risky = data[data['risk_level'] == 'High']
        if len(risky) > 0:
            recommendations.append({
                'type': 'risk_warning',
                'priority': 'Warning',
                'vendors': risky['vendor_id'].tolist(),
                'message': f"⚠️  CAUTION: {len(risky)} vendors flagged as HIGH RISK - Consider alternative strategies"
            })
        
        # Market insights
        above_market = data[data['above_market']]
        if len(above_market) > 0:
            avg_premium = above_market['price_vs_market_pct'].mean()
            recommendations.append({
                'type': 'market_insight',
                'priority': 'Info',
                'message': f"📊 MARKET INSIGHT: {len(above_market)} vendors priced {avg_premium:.1f}% above market - Strong negotiation leverage"
            })
        
        return recommendations
    
    def _calculate_market_benchmark(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates market benchmark statistics.
        
        Args:
            data: Vendor pricing DataFrame
            
        Returns:
            Market benchmark dictionary
        """
        prices = data['current_price'].values
        
        benchmark = {
            'market_average': float(np.mean(prices)),
            'market_median': float(np.median(prices)),
            'market_std': float(np.std(prices)),
            'lowest_price': float(np.min(prices)),
            'highest_price': float(np.max(prices)),
            'price_range_pct': float((np.max(prices) - np.min(prices)) / np.mean(prices) * 100)
        }
        
        return benchmark


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
    import sys
    
    print("="*60)
    print("🧪 TESTING STAGE 5: NEGOTIATION INSIGHTS")
    print("="*60)
    
    # Check if config file provided
    config = {}
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print(f"\n📄 Running with CONFIG: {config_path}")
    else:
        print(f"\n📄 Running with DEMO DATA (default parameters)")
    
    # Create sample vendor data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'vendor_id': [f'V{i:03d}' for i in range(1, 11)],
        'vendor_name': [f'Vendor {i}' for i in range(1, 11)],
        'current_price': [
            2500000,  # V001
            2750000,  # V002 - High price
            2200000,  # V003 - Low price
            2600000,  # V004
            3000000,  # V005 - Very high
            2400000,  # V006
            2550000,  # V007
            2100000,  # V008 - Low
            2650000,  # V009
            2800000   # V010 - High
        ],
        'vendor_relationship_score': np.random.uniform(60, 95, 10)
    })
    
    # Initialize stage with config (if provided)
    stage = NegotiationInsightsStage(config=config.get('parameters', {}))
    
    # Execute
    result = stage.execute(sample_data)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Vendors: {result['results']['total_vendors']}")
    print(f"   Negotiable: {result['results']['negotiable_vendors']}")
    print(f"   Potential Savings: {format_currency(result['results']['total_potential_savings'])}")
    print(f"   Avg Achievable Discount: {format_percentage(result['results']['average_discount_achievable'])}")
    print(f"   High Success Probability: {result['results']['high_success_probability']}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    # Market benchmark
    print(f"\n💹 MARKET BENCHMARK:")
    benchmark = result['results']['market_benchmark']
    print(f"   Average: {format_currency(benchmark['market_average'])}")
    print(f"   Median: {format_currency(benchmark['market_median'])}")
    print(f"   Range: {format_currency(benchmark['lowest_price'])} - {format_currency(benchmark['highest_price'])}")
    print(f"   Variance: {format_percentage(benchmark['price_range_pct'])}")
    
    # Top negotiation opportunities
    print(f"\n💰 TOP NEGOTIATION OPPORTUNITIES:")
    insights = result['results']['negotiation_insights']
    negotiable = [v for v in insights if v['is_negotiable']]
    top_5 = sorted(negotiable, key=lambda x: x['potential_savings'], reverse=True)[:5]
    
    for i, vendor in enumerate(top_5, 1):
        print(f"\n   {i}. {vendor['vendor_name']} ({vendor['vendor_id']})")
        print(f"      Current Price: {format_currency(vendor['current_price'])}")
        print(f"      Target Price: {format_currency(vendor['target_price'])}")
        print(f"      Potential Savings: {format_currency(vendor['potential_savings'])}")
        print(f"      Success Probability: {format_percentage(vendor['success_probability'])}")
        print(f"      Risk Level: {vendor['risk_level']}")
        print(f"      Approach: {vendor['recommended_approach']}")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in result['results']['recommendations']:
        print(f"   [{rec.get('priority', 'Info')}] {rec['message']}")
    
    print("\n" + "="*60)
    print("✅ STAGE 5 TEST COMPLETE")
    print("="*60)
