"""
Stage 10: Digital Record Management & Compliance
=================================================

Manages digital records and ensures compliance:
- Document indexing and retrieval
- Compliance tracking
- Audit trail generation
- Record retention management
- Version control

Uses: Data processing for organization, compliance validation
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
from formatters import format_date
from logger import get_logger


class RecordManagementStage(BaseStage):
    """
    Manages digital records and ensures compliance with retention policies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes Stage 10 with record management parameters.
        
        Args:
            config: Optional configuration dictionary from JSON
        """
        super().__init__(stage_number=10)
        self.logger = get_logger(__name__)
        
        # Load config or use defaults
        if config:
            self.retention_years = config.get('retention_years', 7)
            self.archive_threshold_days = config.get('archive_threshold_days', 365)
            self.compliance_check_enabled = config.get('compliance_check_enabled', True)
        else:
            self.retention_years = 7
            self.archive_threshold_days = 365
            self.compliance_check_enabled = True
    
    def _get_required_columns(self):
        """
        Returns required columns for record management.
        """
        return ['record_id', 'document_type']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes record management workflow.
        
        Args:
            data: DataFrame with document records
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing record management results
        """
        # Index documents
        indexed = self._index_documents(data)
        
        # Check retention compliance
        retention_check = self._check_retention_compliance(indexed)
        
        # Identify archival candidates
        archival = self._identify_archival_candidates(retention_check)
        
        # Generate audit trail
        audit_trail = self._generate_audit_trail(archival)
        
        # Check version control
        version_check = self._check_version_control(audit_trail)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(version_check)
        
        results = {
            'total_records': len(data),
            'compliant_records': len(version_check[version_check['is_compliant']]),
            'non_compliant': len(version_check[~version_check['is_compliant']]),
            'archival_candidates': len(version_check[version_check['archival_candidate']]),
            'records_to_delete': len(version_check[version_check['deletion_candidate']]),
            'compliance_rate': (len(version_check[version_check['is_compliant']]) / len(data) * 100) if len(data) > 0 else 0,
            'records': version_check.to_dict('records'),
            'recommendations': recommendations,
            'summary': self._generate_summary(version_check)
        }
        
        self.logger.info(f"   Compliant Records: {results['compliant_records']}/{results['total_records']}")
        self.logger.info(f"   Archival Candidates: {results['archival_candidates']}")
        
        return results
    
    def _index_documents(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates searchable index for documents.
        
        Args:
            data: Document records DataFrame
            
        Returns:
            DataFrame with indexed metadata
        """
        df = data.copy()
        
        # Add metadata if not present
        if 'creation_date' not in df.columns:
            df['creation_date'] = pd.to_datetime('2024-01-01') - pd.to_timedelta(np.random.randint(0, 2000, len(df)), unit='D')
        
        if 'last_modified' not in df.columns:
            df['last_modified'] = df['creation_date'] + pd.to_timedelta(np.random.randint(0, 365, len(df)), unit='D')
        
        if 'document_status' not in df.columns:
            df['document_status'] = np.random.choice(['Active', 'Archived', 'Pending'], len(df), p=[0.6, 0.3, 0.1])
        
        # Calculate document age
        df['age_days'] = (datetime.now() - df['creation_date']).dt.days
        df['age_years'] = (df['age_days'] / 365).round(1)
        
        # Create search tags
        df['search_tags'] = df.apply(
            lambda row: f"{row['document_type']},{row['document_status']},age_{int(row['age_years'])}y",
            axis=1
        )
        
        return df
    
    def _check_retention_compliance(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Checks if records comply with retention policies.
        
        Args:
            data: Indexed DataFrame
            
        Returns:
            DataFrame with compliance flags
        """
        df = data.copy()
        
        # Define retention periods by document type
        retention_rules = {
            'Contract': 7,
            'Invoice': 7,
            'Purchase Order': 5,
            'RFQ Response': 3,
            'Vendor Agreement': 7,
            'Change Order': 5
        }
        
        # Get retention period for each document
        df['retention_period_years'] = df['document_type'].map(retention_rules).fillna(self.retention_years)
        
        # Check if within retention period
        df['within_retention'] = df['age_years'] <= df['retention_period_years']
        
        # Check if retention expired
        df['retention_expired'] = df['age_years'] > df['retention_period_years']
        
        # Days until expiry
        df['days_until_expiry'] = ((df['retention_period_years'] * 365) - df['age_days']).clip(lower=0)
        
        return df
    
    def _identify_archival_candidates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies documents ready for archival.
        
        Args:
            data: Retention-checked DataFrame
            
        Returns:
            DataFrame with archival flags
        """
        df = data.copy()
        
        # Archival candidates: Old but within retention
        df['archival_candidate'] = (
            (df['age_days'] > self.archive_threshold_days) &
            (df['within_retention']) &
            (df['document_status'] == 'Active')
        )
        
        # Deletion candidates: Past retention period
        df['deletion_candidate'] = df['retention_expired']
        
        # Priority level for archival
        df['archival_priority'] = 'None'
        df.loc[df['archival_candidate'] & (df['age_days'] > 730), 'archival_priority'] = 'High'  # 2+ years
        df.loc[df['archival_candidate'] & (df['age_days'] <= 730), 'archival_priority'] = 'Medium'
        
        return df
    
    def _generate_audit_trail(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates audit trail information for records.
        
        Args:
            data: Archival-flagged DataFrame
            
        Returns:
            DataFrame with audit trail
        """
        df = data.copy()
        
        # Add audit fields if not present
        if 'last_accessed_by' not in df.columns:
            users = ['Admin', 'Procurement Manager', 'Finance Team', 'Legal Team', 'Auditor']
            df['last_accessed_by'] = np.random.choice(users, len(df))
        
        if 'last_access_date' not in df.columns:
            df['last_access_date'] = df['last_modified'] + pd.to_timedelta(np.random.randint(0, 180, len(df)), unit='D')
        
        if 'access_count' not in df.columns:
            df['access_count'] = np.random.randint(1, 50, len(df))
        
        # Days since last access
        df['days_since_access'] = (datetime.now() - df['last_access_date']).dt.days
        
        # Audit flag (frequently accessed = higher importance)
        df['high_importance'] = df['access_count'] > 20
        
        return df
    
    def _check_version_control(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Checks version control status of documents.
        
        Args:
            data: Audit trail DataFrame
            
        Returns:
            DataFrame with version control flags
        """
        df = data.copy()
        
        # Add version info if not present
        if 'version' not in df.columns:
            df['version'] = np.random.choice(['v1.0', 'v1.1', 'v2.0', 'v2.1', 'v3.0'], len(df))
        
        if 'is_latest_version' not in df.columns:
            df['is_latest_version'] = np.random.choice([True, False], len(df), p=[0.8, 0.2])
        
        # Compliance check
        df['is_compliant'] = (
            df['within_retention'] &
            df['is_latest_version'] &
            (df['document_status'] != 'Pending')
        )
        
        # Compliance issues
        df['compliance_issues'] = ''
        df.loc[~df['within_retention'], 'compliance_issues'] += 'Retention expired; '
        df.loc[~df['is_latest_version'], 'compliance_issues'] += 'Outdated version; '
        df.loc[df['document_status'] == 'Pending', 'compliance_issues'] += 'Pending status; '
        
        return df
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates record management recommendations.
        
        Args:
            data: Version-checked DataFrame
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Archival recommendations
        archival = data[data['archival_candidate']]
        if len(archival) > 0:
            recommendations.append({
                'type': 'archival',
                'priority': 'Medium',
                'count': len(archival),
                'records': archival['record_id'].tolist()[:10],  # First 10
                'message': f"📦 ARCHIVAL: {len(archival)} records ready for archival - Free up active storage",
                'action': 'Move to archival storage system'
            })
        
        # Deletion recommendations
        deletion = data[data['deletion_candidate']]
        if len(deletion) > 0:
            recommendations.append({
                'type': 'deletion',
                'priority': 'High',
                'count': len(deletion),
                'records': deletion['record_id'].tolist()[:10],
                'message': f"🗑️  DELETION: {len(deletion)} records past retention period - Delete per policy",
                'action': 'Schedule for secure deletion after legal review'
            })
        
        # Compliance issues
        non_compliant = data[~data['is_compliant']]
        if len(non_compliant) > 0:
            recommendations.append({
                'type': 'compliance',
                'priority': 'High',
                'count': len(non_compliant),
                'records': non_compliant['record_id'].tolist()[:10],
                'message': f"⚠️  COMPLIANCE: {len(non_compliant)} records have compliance issues - Immediate action required",
                'action': 'Review and update to latest version or proper status'
            })
        
        # Version updates needed
        outdated = data[~data['is_latest_version']]
        if len(outdated) > 0:
            recommendations.append({
                'type': 'version_update',
                'priority': 'Medium',
                'count': len(outdated),
                'records': outdated['record_id'].tolist()[:10],
                'message': f"🔄 VERSION: {len(outdated)} records need version update",
                'action': 'Update to latest version and archive old versions'
            })
        
        return recommendations
    
    def _generate_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates summary statistics for record management.
        
        Args:
            data: Processed DataFrame
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_records': len(data),
            'by_status': {
                'Active': int(len(data[data['document_status'] == 'Active'])),
                'Archived': int(len(data[data['document_status'] == 'Archived'])),
                'Pending': int(len(data[data['document_status'] == 'Pending']))
            },
            'by_type': data['document_type'].value_counts().to_dict(),
            'compliance_rate': float((len(data[data['is_compliant']]) / len(data) * 100) if len(data) > 0 else 0),
            'average_age_years': float(data['age_years'].mean()),
            'archival_candidates': int(len(data[data['archival_candidate']])),
            'deletion_candidates': int(len(data[data['deletion_candidate']])),
            'storage_optimization_potential': f"{int(len(data[data['archival_candidate']]) / len(data) * 100)}%"
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
    print("🧪 TESTING STAGE 10: RECORD MANAGEMENT")
    print("="*60)
    
    # Check if config file provided
    config = {}
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print(f"\n📄 Running with CONFIG: {config_path}")
    else:
        print(f"\n📄 Running with DEMO DATA (default parameters)")
    
    # Create sample record data
    np.random.seed(42)
    
    doc_types = ['Contract', 'Invoice', 'Purchase Order', 'RFQ Response', 'Vendor Agreement', 'Change Order']
    
    sample_data = pd.DataFrame({
        'record_id': [f'REC-{i:05d}' for i in range(1, 51)],
        'document_type': np.random.choice(doc_types, 50),
        'document_name': [f'Document {i}' for i in range(1, 51)]
    })
    
    # Initialize stage with config (if provided)
    stage = RecordManagementStage(config=config.get('parameters', {}))
    
    # Execute
    result = stage.execute(sample_data)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"   Status: {result['status']}")
    print(f"   Total Records: {result['results']['total_records']}")
    print(f"   Compliant: {result['results']['compliant_records']}")
    print(f"   Non-Compliant: {result['results']['non_compliant']}")
    print(f"   Archival Candidates: {result['results']['archival_candidates']}")
    print(f"   Deletion Candidates: {result['results']['records_to_delete']}")
    print(f"   Compliance Rate: {format_percentage(result['results']['compliance_rate'])}")
    print(f"   Duration: {result['duration_seconds']}s")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    summary = result['results']['summary']
    print(f"   Total Records: {summary['total_records']}")
    print(f"   Status: Active={summary['by_status']['Active']}, Archived={summary['by_status']['Archived']}, Pending={summary['by_status']['Pending']}")
    print(f"   Compliance Rate: {format_percentage(summary['compliance_rate'])}")
    print(f"   Average Age: {summary['average_age_years']:.1f} years")
    print(f"   Storage Optimization: {summary['storage_optimization_potential']} can be archived")
    print(f"\n   By Document Type:")
    for doc_type, count in summary['by_type'].items():
        print(f"     {doc_type}: {count}")
    
    # Records needing action
    print(f"\n📌 RECORDS NEEDING ACTION:")
    records = result['results']['records']
    action_needed = [r for r in records if not r['is_compliant'] or r['archival_candidate'] or r['deletion_candidate']]
    
    for i, record in enumerate(action_needed[:5], 1):
        print(f"\n   {i}. {record['record_id']} - {record['document_type']}")
        print(f"      Age: {record['age_years']:.1f} years")
        print(f"      Status: {record['document_status']}")
        print(f"      Compliant: {'Yes' if record['is_compliant'] else 'No'}")
        if record['archival_candidate']:
            print(f"      ✅ Archival Candidate (Priority: {record['archival_priority']})")
        if record['deletion_candidate']:
            print(f"      🗑️  Deletion Candidate (Retention Expired)")
        if record['compliance_issues']:
            print(f"      Issues: {record['compliance_issues']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in result['results']['recommendations']:
        print(f"\n   [{rec['priority']}] {rec['message']}")
        print(f"   Action: {rec['action']}")
    
    print("\n" + "="*60)
    print("✅ STAGE 10 TEST COMPLETE")
    print("="*60)
