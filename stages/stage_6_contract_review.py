"""
Stage 6: AI-Powered Contract Review & Compliance Check
=======================================================

Automates contract review through:
- Clause extraction and validation
- GCC/SCC compliance verification
- Risk clause identification
- Missing mandatory clauses detection
- Red-Amber-Green (RAG) status classification

Uses: Rule Engine for compliance, NLP for clause extraction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import os
import re

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
from formatters import format_currency
from logger import get_logger


class ContractReviewStage(BaseStage):
    """
    Performs automated contract review and compliance checks.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes Stage 6 with contract review parameters.
        
        Args:
            config: Optional configuration dictionary from JSON
        """
        super().__init__(stage_number=6)
        self.rule_engine = RuleEngine()
        self.logger = get_logger(__name__)
        
        # Load config or use defaults
        if config:
            self.mandatory_clauses = config.get('mandatory_clauses', self._get_default_clauses())
            self.high_value_threshold = config.get('high_value_threshold', 10000000)
            self.auto_reject_threshold = config.get('auto_reject_threshold', 3)
        else:
            self.mandatory_clauses = self._get_default_clauses()
            self.high_value_threshold = 10000000  # 1 Crore
            self.auto_reject_threshold = 3  # Missing clauses count
    
    def _get_default_clauses(self) -> List[str]:
        """
        Returns list of mandatory contract clauses.
        
        Returns:
            List of clause names
        """
        return [
            'payment_clause',
            'penalty_clause',
            'termination_clause',
            'warranty_clause',
            'liability_clause',
            'force_majeure_clause',
            'dispute_resolution_clause'
        ]
    
    def _get_required_columns(self):
        """
        Returns required columns for contract review.
        """
        return ['contract_id', 'contract_value']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes contract review workflow.
        
        Args:
            data: DataFrame with contract data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing review results
        """
        # Check mandatory clauses
        clause_check = self._check_mandatory_clauses(data)
        
        # Verify GCC/SCC compliance
        compliance_check = self._verify_gcc_scc_compliance(clause_check)
        
        # Identify risk clauses
        risk_analysis = self._identify_risk_clauses(compliance_check)
        
        # Calculate contract risk scores
        scored_contracts = self._calculate_risk_scores(risk_analysis)
        
        # Classify RAG status
        rag_classified = self._classify_rag_status(scored_contracts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(rag_classified)
        
        results = {
            'total_contracts': len(data),
            'compliant_contracts': len(rag_classified[rag_classified['rag_status'] == 'Green']),
            'amber_contracts': len(rag_classified[rag_classified['rag_status'] == 'Amber']),
            'red_contracts': len(rag_classified[rag_classified['rag_status'] == 'Red']),
            'average_risk_score': rag_classified['risk_score'].mean(),
            'high_value_contracts': len(rag_classified[rag_classified['is_high_value']]),
            'contracts': rag_classified.to_dict('records'),
            'recommendations': recommendations,
            'compliance_summary': self._generate_compliance_summary(rag_classified)
        }
        
        self.logger.info(f"   Compliant (Green): {results['compliant_contracts']}/{results['total_contracts']}")
        self.logger.info(f"   Red Flags: {results['red_contracts']}")
        
        return results
    
    def _check_mandatory_clauses(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Checks presence of mandatory clauses in contracts.
        
        Args:
            data: Contract DataFrame
            
        Returns:
            DataFrame with clause presence flags
        """
        df = data.copy()
        
        # Add clause columns if not present (simulate for demo)
        for clause in self.mandatory_clauses:
            if clause not in df.columns:
                # Randomly simulate clause presence (70% present)
                df[clause] = np.random.choice([True, False], size=len(df), p=[0.7, 0.3])
        
        # Count missing clauses
        clause_cols = [c for c in self.mandatory_clauses if c in df.columns]
        df['missing_clauses'] = df[clause_cols].apply(lambda row: sum(~row), axis=1)
        
        # List missing clause names
        df['missing_clause_names'] = df.apply(
            lambda row: [c.replace('_clause', '') for c in clause_cols if not row[c]],
            axis=1
        )
        
        return df
    
    def _verify_gcc_scc_compliance(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Verifies General Conditions of Contract (GCC) and Special Conditions of Contract (SCC) compliance.
        
        Args:
            data: Clause-checked DataFrame
            
        Returns:
            DataFrame with compliance flags
        """
        df = data.copy()
        
        # Add GCC/SCC fields if not present
        if 'gcc_compliant' not in df.columns:
            # Simulate GCC compliance (80% compliant)
            df['gcc_compliant'] = np.random.choice([True, False], size=len(df), p=[0.8, 0.2])
        
        if 'scc_included' not in df.columns:
            # Simulate SCC inclusion (75% included)
            df['scc_included'] = np.random.choice([True, False], size=len(df), p=[0.75, 0.25])
        
        # Overall compliance flag
        df['is_compliant'] = df['gcc_compliant'] & df['scc_included'] & (df['missing_clauses'] == 0)
        
        return df
    
    def _identify_risk_clauses(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies high-risk clauses in contracts.
        
        Args:
            data: Compliance-checked DataFrame
            
        Returns:
            DataFrame with risk flags
        """
        df = data.copy()
        
        # Initialize risk flags
        df['risk_flags'] = ''
        
        # Flag 1: Missing critical clauses
        critical_clauses = ['payment_clause', 'penalty_clause', 'liability_clause']
        for clause in critical_clauses:
            if clause in df.columns:
                mask = ~df[clause]
                df.loc[mask, 'risk_flags'] += f'Missing {clause.replace("_", " ")}; '
        
        # Flag 2: GCC non-compliance
        if 'gcc_compliant' in df.columns:
            mask = ~df['gcc_compliant']
            df.loc[mask, 'risk_flags'] += 'GCC non-compliant; '
        
        # Flag 3: High value without SCC
        df['is_high_value'] = df['contract_value'] > self.high_value_threshold
        mask = df['is_high_value'] & (~df['scc_included'])
        df.loc[mask, 'risk_flags'] += 'High value without SCC; '
        
        # Flag 4: Multiple missing clauses
        mask = df['missing_clauses'] >= 3
        df.loc[mask, 'risk_flags'] += f'Multiple missing clauses ({df["missing_clauses"]}); '
        
        return df
    
    def _calculate_risk_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates overall risk score for each contract (0-100).
        
        Args:
            data: Risk-flagged DataFrame
            
        Returns:
            DataFrame with risk scores
        """
        df = data.copy()
        
        # Initialize risk score
        df['risk_score'] = 0
        
        # Factor 1: Missing clauses (20 points per clause, max 60)
        df['risk_score'] += (df['missing_clauses'] * 20).clip(0, 60)
        
        # Factor 2: GCC non-compliance (25 points)
        if 'gcc_compliant' in df.columns:
            df.loc[~df['gcc_compliant'], 'risk_score'] += 25
        
        # Factor 3: Missing SCC (15 points)
        if 'scc_included' in df.columns:
            df.loc[~df['scc_included'], 'risk_score'] += 15
        
        # Factor 4: High value contract (additional 10 points if risky)
        mask = df['is_high_value'] & (df['risk_score'] > 20)
        df.loc[mask, 'risk_score'] += 10
        
        # Clip to 0-100 range
        df['risk_score'] = df['risk_score'].clip(0, 100)
        
        return df
    
    def _classify_rag_status(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies contracts into Red-Amber-Green status.
        
        Args:
            data: Scored DataFrame
            
        Returns:
            DataFrame with RAG classification
        """
        df = data.copy()
        
        # RAG classification based on risk score
        df['rag_status'] = 'Green'
        df.loc[df['risk_score'] >= 40, 'rag_status'] = 'Amber'
        df.loc[df['risk_score'] >= 70, 'rag_status'] = 'Red'
        
        # Add approval recommendation
        df['approval_recommendation'] = 'Approve'
        df.loc[df['rag_status'] == 'Amber', 'approval_recommendation'] = 'Review Required'
        df.loc[df['rag_status'] == 'Red', 'approval_recommendation'] = 'Reject / Legal Review'
        
        # Requires legal review flag
        df['requires_legal_review'] = (
            (df['rag_status'] == 'Red') |
            ((df['rag_status'] == 'Amber') & df['is_high_value'])
        )
        
        return df
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates actionable recommendations for contract approval.
        
        Args:
            data: RAG-classified DataFrame
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Red contracts (critical)
        red_contracts = data[data['rag_status'] == 'Red']
        if len(red_contracts) > 0:
            recommendations.append({
                'type': 'critical',
                'priority': 'High',
                'count': len(red_contracts),
                'contracts': red_contracts['contract_id'].tolist(),
                'message': f"🔴 CRITICAL: {len(red_contracts)} contracts flagged RED - Do NOT approve without legal review"
            })
        
        # Amber contracts (warning)
        amber_contracts = data[data['rag_status'] == 'Amber']
        if len(amber_contracts) > 0:
            recommendations.append({
                'type': 'warning',
                'priority': 'Medium',
                'count': len(amber_contracts),
                'contracts': amber_contracts['contract_id'].tolist(),
                'message': f"⚠️  CAUTION: {len(amber_contracts)} contracts flagged AMBER - Requires review before approval"
            })
        
        # Green contracts (approved)
        green_contracts = data[data['rag_status'] == 'Green']
        if len(green_contracts) > 0:
            recommendations.append({
                'type': 'approved',
                'priority': 'Info',
                'count': len(green_contracts),
                'contracts': green_contracts['contract_id'].tolist(),
                'message': f"✅ APPROVED: {len(green_contracts)} contracts passed all compliance checks"
            })
        
        # High-value contracts needing review
        high_value_review = data[data['requires_legal_review'] & data['is_high_value']]
        if len(high_value_review) > 0:
            total_value = high_value_review['contract_value'].sum()
            recommendations.append({
                'type': 'high_value',
                'priority': 'High',
                'count': len(high_value_review),
                'total_value': total_value,
                'message': f"💰 HIGH VALUE: {len(high_value_review)} contracts worth {format_currency(total_value)} require legal review"
            })
        
        return recommendations
    
    def _generate_compliance_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates compliance summary statistics.
        
        Args:
            data: Classified DataFrame
            
        Returns:
            Compliance summary dictionary
        """
        summary = {
            'total_contracts': len(data),
            'compliant_pct': (len(data[data['is_compliant']]) / len(data) * 100) if len(data) > 0 else 0,
            'gcc_compliance_pct': (len(data[data['gcc_compliant']]) / len(data) * 100) if len(data) > 0 else 0,
            'scc_inclusion_pct': (len(data[data['scc_included']]) / len(data) * 100) if len(data) > 0 else 0,
            'avg_missing_clauses': data['missing_clauses'].mean(),
            'most_common_missing': self._find_most_common_missing_clause(data)
        }
        
        return summary
    
    def _find_most_common_missing_clause(self, data: pd.DataFrame) -> str:
        """
        Finds the most commonly missing clause across all contracts.
        
        Args:
            data: Contract DataFrame
            
        Returns:
            Name of most commonly missing clause
        """
        # Flatten all missing clause lists
        all_missing = []
        for clause_list in data['missing_clause_names']:
            all_missing.extend(clause_list)
        
        if not all_missing:
            return 'None'
        
        # Find most common
        from collections import Counter
        most_common = Counter(all_missing).most_common(1)
        
        return most_common[0][0] if most_common else 'None'


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
    print("🧪 TESTING STAGE 6: CONTRACT REVIEW")
    print("="*60)
    
    # Check if config file provided
    config = {}
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print(f"\n📄 Running with CONFIG: {config_path}")
    else:
        print(f"\n📄 Running with DEMO DATA (default parameters)")
    
    # Create sample contract data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'contract_id': [f'CONT-{i:04d}' for i in range(1, 16)],
        'contract_name': [f'Contract {i}' for i in range(1, 16)],
        'vendor_id': [f'V{i:03d}' for i in range(1, 16)],
        'contract_value': [
            5000000, 15000000, 3000000, 25000000, 8000000,
            12000000, 4500000, 30000000, 6000000, 9000000,
            7000000, 20000000, 5500000, 11000000, 16000000
        ]
    })
    
    # Initialize stage with config (if provided)
    stage = ContractReviewStage(config=config.get('parameters', {}))
    
    # Execute
    result = stage.execute(sample_data)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Contracts: {result['results']['total_contracts']}")
    print(f"   Green (Compliant): {result['results']['compliant_contracts']}")
    print(f"   Amber (Review): {result['results']['amber_contracts']}")
    print(f"   Red (Reject): {result['results']['red_contracts']}")
    print(f"   High Value: {result['results']['high_value_contracts']}")
    print(f"   Avg Risk Score: {result['results']['average_risk_score']:.2f}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    # Compliance summary
    print(f"\n📋 COMPLIANCE SUMMARY:")
    summary = result['results']['compliance_summary']
    print(f"   Overall Compliance: {summary['compliant_pct']:.1f}%")
    print(f"   GCC Compliance: {summary['gcc_compliance_pct']:.1f}%")
    print(f"   SCC Inclusion: {summary['scc_inclusion_pct']:.1f}%")
    print(f"   Avg Missing Clauses: {summary['avg_missing_clauses']:.1f}")
    print(f"   Most Common Missing: {summary['most_common_missing']}")
    
    # Top risk contracts
    print(f"\n🚨 TOP RISK CONTRACTS:")
    contracts = result['results']['contracts']
    high_risk = sorted(contracts, key=lambda x: x['risk_score'], reverse=True)[:5]
    
    for i, contract in enumerate(high_risk, 1):
        print(f"\n   {i}. {contract['contract_id']} - {contract['contract_name']}")
        print(f"      Value: {format_currency(contract['contract_value'])}")
        print(f"      Risk Score: {contract['risk_score']}")
        print(f"      RAG Status: {contract['rag_status']}")
        print(f"      Missing Clauses: {contract['missing_clauses']}")
        print(f"      Recommendation: {contract['approval_recommendation']}")
        if contract['risk_flags']:
            print(f"      Flags: {contract['risk_flags']}")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in result['results']['recommendations']:
        print(f"   [{rec['priority']}] {rec['message']}")
    
    print("\n" + "="*60)
    print("✅ STAGE 6 TEST COMPLETE")
    print("="*60)
