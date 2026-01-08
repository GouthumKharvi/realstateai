"""
Stages Package - AI-Enabled Procurement Processing Stages
=========================================================

This package contains the 11 processing stages for real estate
procurement automation:

Stage 1:  Vendor Development & Approval Workflow
Stage 2:  Automated Document Creation (Contracts, POs)
Stage 3:  RFQ Price & Specification Anomaly Detection
Stage 4:  Vendor Shortlisting for Tenders
Stage 5:  AI-Driven Negotiation Insights
Stage 6:  Contract Review & Compliance Checks
Stage 7:  Change Order Analysis
Stage 8:  Risk Assessment for Procurement Decisions
Stage 9:  Predictive Insights for Vendor Shortlisting
Stage 10: Spend Analysis & Optimization Suggestions
Stage 11: Invoice Fraud Detection

Each stage:
- Inherits from BaseStage
- Uses engines (Rule, Statistical, ML)
- Validates inputs
- Generates structured outputs
- Logs execution details

Usage:
    from stages import VendorDevelopmentStage
    
    stage = VendorDevelopmentStage()
    result = stage.execute(vendor_data)
"""

from .base_stage import BaseStage

# Import all stages (will be added as we build them)
try:
    from .stage_1_vendor_development import VendorDevelopmentStage
except ImportError:
    VendorDevelopmentStage = None

try:
    from .stage_2_document_automation import DocumentAutomationStage
except ImportError:
    DocumentAutomationStage = None

try:
    from .stage_3_rfq_anomaly import RFQAnomalyDetectionStage
except ImportError:
    RFQAnomalyDetectionStage = None

try:
    from .stage_4_vendor_shortlisting import VendorShortlistingStage
except ImportError:
    VendorShortlistingStage = None

try:
    from .stage_5_negotiation import NegotiationInsightsStage
except ImportError:
    NegotiationInsightsStage = None

try:
    from .stage_6_contract_review import ContractReviewStage
except ImportError:
    ContractReviewStage = None

try:
    from .stage_7_change_orders import ChangeOrderAnalysisStage
except ImportError:
    ChangeOrderAnalysisStage = None

try:
    from .stage_8_risk_assessment import RiskAssessmentStage
except ImportError:
    RiskAssessmentStage = None

try:
    from .stage_9_predictive import PredictiveAnalyticsStage
except ImportError:
    PredictiveAnalyticsStage = None

try:
    from .stage_10_spend_analysis import SpendAnalysisStage
except ImportError:
    SpendAnalysisStage = None

try:
    from .stage_11_fraud_detection import FraudDetectionStage
except ImportError:
    FraudDetectionStage = None


# Stage registry for dynamic access
STAGE_REGISTRY = {
    1: VendorDevelopmentStage,
    2: DocumentAutomationStage,
    3: RFQAnomalyDetectionStage,
    4: VendorShortlistingStage,
    5: NegotiationInsightsStage,
    6: ContractReviewStage,
    7: ChangeOrderAnalysisStage,
    8: RiskAssessmentStage,
    9: PredictiveAnalyticsStage,
    10: SpendAnalysisStage,
    11: FraudDetectionStage
}

# Get available stages (non-None)
AVAILABLE_STAGES = {k: v for k, v in STAGE_REGISTRY.items() if v is not None}


def get_stage(stage_number: int):
    """
    Get a stage class by number.
    
    Args:
        stage_number: Stage number (1-11)
        
    Returns:
        Stage class or None if not available
        
    Example:
        >>> StageClass = get_stage(4)
        >>> stage = StageClass()
        >>> result = stage.execute(data)
    """
    return STAGE_REGISTRY.get(stage_number)


def list_available_stages():
    """
    List all available (imported) stages.
    
    Returns:
        Dictionary of stage number to class
        
    Example:
        >>> stages = list_available_stages()
        >>> print(f"Available: {list(stages.keys())}")
    """
    return AVAILABLE_STAGES


__version__ = '1.0.0'
__author__ = 'Assisto.tech Internship Team'

__all__ = [
    # Base class
    'BaseStage',
    
    # Stage classes
    'VendorDevelopmentStage',
    'DocumentAutomationStage',
    'RFQAnomalyDetectionStage',
    'VendorShortlistingStage',
    'NegotiationInsightsStage',
    'ContractReviewStage',
    'ChangeOrderAnalysisStage',
    'RiskAssessmentStage',
    'PredictiveAnalyticsStage',
    'SpendAnalysisStage',
    'FraudDetectionStage',
    
    # Helper functions
    'get_stage',
    'list_available_stages',
    
    # Registry
    'STAGE_REGISTRY',
    'AVAILABLE_STAGES'
]
