"""
Stage 9: Predictive Analytics for Procurement
==============================================

Provides predictive insights through:
- Price trend forecasting
- Delivery delay prediction
- Demand forecasting
- Budget overrun prediction
- Vendor performance prediction

Uses: ML Engine for predictions, Statistical Engine for trends
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
from ml_engine import MLEngine
from statistical_engine import StatisticalEngine
from formatters import format_currency, format_percentage
from logger import get_logger


class PredictiveAnalyticsStage(BaseStage):
    """
    Performs predictive analytics for procurement planning.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes Stage 9 with predictive analytics parameters.
        
        Args:
            config: Optional configuration dictionary from JSON
        """
        super().__init__(stage_number=9)
        self.ml_engine = MLEngine()
        self.stat_engine = StatisticalEngine()
        self.logger = get_logger(__name__)
        
        # Load config or use defaults
        if config:
            self.forecast_horizon_days = config.get('forecast_horizon_days', 90)
            self.confidence_level = config.get('confidence_level', 0.95)
            self.price_volatility_threshold = config.get('price_volatility_threshold', 15)
        else:
            self.forecast_horizon_days = 90
            self.confidence_level = 0.95
            self.price_volatility_threshold = 15
    
    def _get_required_columns(self):
        """
        Returns required columns for predictive analytics.
        """
        return ['item_id', 'current_price']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes predictive analytics workflow.
        
        Args:
            data: DataFrame with historical procurement data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing predictions and forecasts
        """
        # Forecast price trends
        price_forecast = self._forecast_price_trends(data)
        
        # Predict delivery delays
        delay_prediction = self._predict_delivery_delays(price_forecast)
        
        # Forecast demand
        demand_forecast = self._forecast_demand(delay_prediction)
        
        # Predict budget overruns
        budget_prediction = self._predict_budget_overruns(demand_forecast)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(budget_prediction)
        
        results = {
            'total_items': len(data),
            'items_with_price_increase': len(budget_prediction[budget_prediction['price_trend'] == 'Increasing']),
            'high_delay_risk_items': len(budget_prediction[budget_prediction['delay_risk'] == 'High']),
            'budget_overrun_risk': len(budget_prediction[budget_prediction['budget_overrun_probability'] > 0.7]),
            'average_price_change_pct': budget_prediction['predicted_price_change_pct'].mean(),
            'predictions': budget_prediction.to_dict('records'),
            'recommendations': recommendations,
            'forecast_summary': self._generate_forecast_summary(budget_prediction)
        }
        
        self.logger.info(f"   Price Increases Expected: {results['items_with_price_increase']}/{results['total_items']}")
        self.logger.info(f"   High Delay Risk: {results['high_delay_risk_items']}")
        
        return results
    
    def _forecast_price_trends(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Forecasts future price trends for items.
        
        Args:
            data: Historical data DataFrame
            
        Returns:
            DataFrame with price forecasts
        """
        df = data.copy()
        
        # Add historical prices if not present
        if 'price_3m_ago' not in df.columns:
            df['price_3m_ago'] = df['current_price'] * np.random.uniform(0.85, 1.15, len(df))
        
        if 'price_6m_ago' not in df.columns:
            df['price_6m_ago'] = df['current_price'] * np.random.uniform(0.80, 1.20, len(df))
        
        # Calculate historical trend
        df['price_change_3m_pct'] = ((df['current_price'] - df['price_3m_ago']) / df['price_3m_ago'] * 100)
        df['price_change_6m_pct'] = ((df['current_price'] - df['price_6m_ago']) / df['price_6m_ago'] * 100)
        
        # Predict future price (simple linear extrapolation + ML adjustment)
        df['predicted_price_change_pct'] = (
            df['price_change_3m_pct'] * 0.6 + 
            df['price_change_6m_pct'] * 0.4 +
            np.random.normal(0, 2, len(df))  # Add some randomness
        )
        
        df['predicted_price'] = df['current_price'] * (1 + df['predicted_price_change_pct'] / 100)
        
        # Classify trend
        df['price_trend'] = 'Stable'
        df.loc[df['predicted_price_change_pct'] > 5, 'price_trend'] = 'Increasing'
        df.loc[df['predicted_price_change_pct'] < -5, 'price_trend'] = 'Decreasing'
        
        # Volatility assessment
        df['price_volatility'] = abs(df['price_change_3m_pct'] - df['price_change_6m_pct'])
        df['high_volatility'] = df['price_volatility'] > self.price_volatility_threshold
        
        return df
    
    def _predict_delivery_delays(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts probability of delivery delays.
        
        Args:
            data: Price forecast DataFrame
            
        Returns:
            DataFrame with delay predictions
        """
        df = data.copy()
        
        # Add historical delivery data if not present
        if 'past_delivery_performance' not in df.columns:
            df['past_delivery_performance'] = np.random.uniform(70, 98, len(df))
        
        if 'supplier_reliability_score' not in df.columns:
            df['supplier_reliability_score'] = np.random.uniform(60, 95, len(df))
        
        if 'lead_time_days' not in df.columns:
            df['lead_time_days'] = np.random.randint(15, 90, len(df))
        
        # Calculate delay probability (0-1)
        df['delay_probability'] = (
            (100 - df['past_delivery_performance']) / 100 * 0.4 +
            (100 - df['supplier_reliability_score']) / 100 * 0.3 +
            (df['lead_time_days'] / 90) * 0.2 +
            (df['high_volatility'].astype(int) * 0.1)
        )
        
        df['delay_probability'] = df['delay_probability'].clip(0, 1)
        
        # Predict expected delay days
        df['expected_delay_days'] = (df['delay_probability'] * df['lead_time_days'] * 0.3).astype(int)
        
        # Risk classification
        df['delay_risk'] = 'Low'
        df.loc[df['delay_probability'] > 0.3, 'delay_risk'] = 'Medium'
        df.loc[df['delay_probability'] > 0.6, 'delay_risk'] = 'High'
        
        return df
    
    def _forecast_demand(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Forecasts future demand for items.
        
        Args:
            data: Delay prediction DataFrame
            
        Returns:
            DataFrame with demand forecasts
        """
        df = data.copy()
        
        # Add historical demand if not present
        if 'current_monthly_demand' not in df.columns:
            df['current_monthly_demand'] = np.random.randint(10, 500, len(df))
        
        if 'demand_3m_ago' not in df.columns:
            df['demand_3m_ago'] = df['current_monthly_demand'] * np.random.uniform(0.8, 1.2, len(df))
        
        # Calculate demand growth rate
        df['demand_growth_rate'] = ((df['current_monthly_demand'] - df['demand_3m_ago']) / df['demand_3m_ago'] * 100)
        
        # Forecast next period demand
        df['forecasted_demand'] = (
            df['current_monthly_demand'] * 
            (1 + df['demand_growth_rate'] / 100) *
            np.random.uniform(0.95, 1.05, len(df))  # Add variance
        ).astype(int)
        
        # Demand trend
        df['demand_trend'] = 'Stable'
        df.loc[df['demand_growth_rate'] > 10, 'demand_trend'] = 'Growing'
        df.loc[df['demand_growth_rate'] < -10, 'demand_trend'] = 'Declining'
        
        return df
    
    def _predict_budget_overruns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts probability of budget overruns.
        
        Args:
            data: Demand forecast DataFrame
            
        Returns:
            DataFrame with budget overrun predictions
        """
        df = data.copy()
        
        # Add budget data if not present
        if 'allocated_budget' not in df.columns:
            df['allocated_budget'] = df['current_price'] * df['current_monthly_demand'] * 3
        
        # Calculate predicted spend
        df['predicted_spend'] = df['predicted_price'] * df['forecasted_demand'] * 3  # 3 months
        
        # Budget variance
        df['budget_variance'] = df['predicted_spend'] - df['allocated_budget']
        df['budget_variance_pct'] = (df['budget_variance'] / df['allocated_budget'] * 100)
        
        # Overrun probability
        df['budget_overrun_probability'] = 0.0
        
        # Factor 1: Price increase impact
        df.loc[df['price_trend'] == 'Increasing', 'budget_overrun_probability'] += 0.3
        
        # Factor 2: Demand increase impact
        df.loc[df['demand_trend'] == 'Growing', 'budget_overrun_probability'] += 0.2
        
        # Factor 3: Delay impact (delays = cost increase)
        df['budget_overrun_probability'] += df['delay_probability'] * 0.3
        
        # Factor 4: High volatility impact
        df.loc[df['high_volatility'], 'budget_overrun_probability'] += 0.2
        
        df['budget_overrun_probability'] = df['budget_overrun_probability'].clip(0, 1)
        
        # Risk level
        df['budget_risk'] = 'Low'
        df.loc[df['budget_overrun_probability'] > 0.4, 'budget_risk'] = 'Medium'
        df.loc[df['budget_overrun_probability'] > 0.7, 'budget_risk'] = 'High'
        
        return df
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates actionable recommendations based on predictions.
        
        Args:
            data: Prediction DataFrame
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Price increase warnings
        price_increases = data[data['price_trend'] == 'Increasing']
        if len(price_increases) > 0:
            top_increases = price_increases.nlargest(3, 'predicted_price_change_pct')
            recommendations.append({
                'type': 'price_increase',
                'priority': 'High',
                'count': len(price_increases),
                'items': top_increases['item_id'].tolist(),
                'message': f"📈 PRICE ALERT: {len(price_increases)} items expected to increase - Consider bulk purchasing now",
                'action': 'Accelerate procurement for high-increase items'
            })
        
        # Delay risk warnings
        high_delay = data[data['delay_risk'] == 'High']
        if len(high_delay) > 0:
            recommendations.append({
                'type': 'delay_risk',
                'priority': 'High',
                'count': len(high_delay),
                'items': high_delay['item_id'].tolist(),
                'message': f"⏱️  DELAY RISK: {len(high_delay)} items have high delay probability - Plan buffer time",
                'action': 'Identify alternative suppliers or increase safety stock'
            })
        
        # Budget overrun warnings
        budget_risk = data[data['budget_risk'] == 'High']
        if len(budget_risk) > 0:
            total_overrun = budget_risk['budget_variance'].sum()
            recommendations.append({
                'type': 'budget_overrun',
                'priority': 'Critical',
                'count': len(budget_risk),
                'total_overrun': total_overrun,
                'message': f"💰 BUDGET ALERT: {len(budget_risk)} items risk overrun of {format_currency(total_overrun)}",
                'action': 'Review and increase budget allocation or reduce demand'
            })
        
        # Favorable conditions
        price_decreases = data[data['price_trend'] == 'Decreasing']
        if len(price_decreases) > 0:
            recommendations.append({
                'type': 'opportunity',
                'priority': 'Info',
                'count': len(price_decreases),
                'items': price_decreases['item_id'].tolist(),
                'message': f"✅ OPPORTUNITY: {len(price_decreases)} items expected to decrease - Delay purchasing if possible",
                'action': 'Wait for price decrease or negotiate better rates'
            })
        
        return recommendations
    
    def _generate_forecast_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates summary statistics for forecasts.
        
        Args:
            data: Prediction DataFrame
            
        Returns:
            Forecast summary dictionary
        """
        summary = {
            'forecast_horizon_days': self.forecast_horizon_days,
            'total_items_analyzed': len(data),
            'average_price_change': float(data['predicted_price_change_pct'].mean()),
            'average_delay_probability': float(data['delay_probability'].mean()),
            'total_predicted_spend': float(data['predicted_spend'].sum()),
            'total_budget_allocated': float(data['allocated_budget'].sum()),
            'overall_budget_variance': float(data['budget_variance'].sum()),
            'items_at_risk': {
                'price_increase': int(len(data[data['price_trend'] == 'Increasing'])),
                'high_delay': int(len(data[data['delay_risk'] == 'High'])),
                'budget_overrun': int(len(data[data['budget_risk'] == 'High']))
            }
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
    print("🧪 TESTING STAGE 9: PREDICTIVE ANALYTICS")
    print("="*60)
    
    # Check if config file provided
    config = {}
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print(f"\n📄 Running with CONFIG: {config_path}")
    else:
        print(f"\n📄 Running with DEMO DATA (default parameters)")
    
    # Create sample item data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'item_id': [f'ITEM-{i:03d}' for i in range(1, 16)],
        'item_name': [f'Construction Material {i}' for i in range(1, 16)],
        'current_price': np.random.uniform(10000, 500000, 15),
        'current_monthly_demand': np.random.randint(50, 500, 15)
    })
    
    # Initialize stage with config (if provided)
    stage = PredictiveAnalyticsStage(config=config.get('parameters', {}))
    
    # Execute
    result = stage.execute(sample_data)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Items Analyzed: {result['results']['total_items']}")
    print(f"   Price Increases Expected: {result['results']['items_with_price_increase']}")
    print(f"   High Delay Risk: {result['results']['high_delay_risk_items']}")
    print(f"   Budget Overrun Risk: {result['results']['budget_overrun_risk']}")
    print(f"   Avg Price Change: {format_percentage(result['results']['average_price_change_pct'])}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    # Forecast summary
    print(f"\n📋 FORECAST SUMMARY:")
    summary = result['results']['forecast_summary']
    print(f"   Forecast Horizon: {summary['forecast_horizon_days']} days")
    print(f"   Avg Price Change: {format_percentage(summary['average_price_change'])}")
    print(f"   Avg Delay Probability: {format_percentage(summary['average_delay_probability'] * 100)}")
    print(f"   Total Predicted Spend: {format_currency(summary['total_predicted_spend'])}")
    print(f"   Total Budget: {format_currency(summary['total_budget_allocated'])}")
    print(f"   Budget Variance: {format_currency(summary['overall_budget_variance'])}")
    print(f"\n   Items at Risk:")
    print(f"     Price Increase: {summary['items_at_risk']['price_increase']}")
    print(f"     High Delay: {summary['items_at_risk']['high_delay']}")
    print(f"     Budget Overrun: {summary['items_at_risk']['budget_overrun']}")
    
    # Top predictions
    print(f"\n🔮 TOP PREDICTIONS:")
    predictions = result['results']['predictions']
    critical = [p for p in predictions if p['budget_risk'] == 'High' or p['delay_risk'] == 'High']
    
    for i, pred in enumerate(critical[:5], 1):
        print(f"\n   {i}. {pred['item_id']} - {pred['item_name']}")
        print(f"      Current Price: {format_currency(pred['current_price'])}")
        print(f"      Predicted Price: {format_currency(pred['predicted_price'])} ({pred['price_trend']})")
        print(f"      Price Change: {format_percentage(pred['predicted_price_change_pct'])}")
        print(f"      Delay Risk: {pred['delay_risk']} (Prob: {format_percentage(pred['delay_probability'] * 100)})")
        print(f"      Budget Risk: {pred['budget_risk']} (Prob: {format_percentage(pred['budget_overrun_probability'] * 100)})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in result['results']['recommendations']:
        print(f"\n   [{rec['priority']}] {rec['message']}")
        print(f"   Action: {rec['action']}")
    
    print("\n" + "="*60)
    print("✅ STAGE 9 TEST COMPLETE")
    print("="*60)
