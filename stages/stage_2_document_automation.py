"""
Stage 2: Automated Document Creation (Contracts, POs)
=====================================================

Automates creation of:
- Purchase Orders (POs)
- Contract documents
- Work orders
Using templates and vendor data

Uses: Rule Engine for validation, templates for generation
"""

import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta

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
from formatters import format_currency, format_date
from logger import get_logger
from constants import FINANCIAL_THRESHOLDS


class DocumentAutomationStage(BaseStage):
    """
    Generates purchase orders and contracts automatically.
    """
    
    def __init__(self):
        """
        Initializes Stage 2 with document templates.
        """
        super().__init__(stage_number=2)
        self.rule_engine = RuleEngine()
        self.logger = get_logger(__name__)
        
        # Document templates
        self.po_template = self._get_po_template()
        self.contract_template = self._get_contract_template()
    
    def _get_required_columns(self):
        """
        Returns required columns for document generation.
        """
        return ['vendor_id', 'vendor_name', 'amount']
    
    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Processes document generation workflow.
        
        Args:
            data: DataFrame with vendor and order information
            **kwargs: doc_type ('po' or 'contract')
            
        Returns:
            Dictionary containing generated documents
        """
        doc_type = kwargs.get('doc_type', 'po')
        
        # Generate documents
        if doc_type == 'po':
            documents = self._generate_purchase_orders(data)
        elif doc_type == 'contract':
            documents = self._generate_contracts(data)
        else:
            documents = self._generate_both(data)
        
        # Validate documents
        validated = self._validate_documents(documents, doc_type)
        
        # Generate statistics
        stats = self._generate_statistics(validated)
        
        results = {
            'total_documents': len(validated),
            'valid_documents': sum(1 for d in validated if d['is_valid']),
            'invalid_documents': sum(1 for d in validated if not d['is_valid']),
            'document_type': doc_type,
            'documents': validated,
            'statistics': stats
        }
        
        self.logger.info(f"   Generated: {results['valid_documents']} valid documents")
        self.logger.info(f"   Invalid: {results['invalid_documents']} documents")
        
        return results
    
    def _generate_purchase_orders(self, data: pd.DataFrame) -> list:
        """
        Generates purchase orders from vendor data.
        
        Args:
            data: Vendor order DataFrame
            
        Returns:
            List of generated PO documents
        """
        documents = []
        
        for idx, row in data.iterrows():
            po_number = f"PO-{datetime.now().strftime('%Y%m%d')}-{idx+1:04d}"
            
            # Fill template
            po_doc = self.po_template.copy()
            po_doc['po_number'] = po_number
            po_doc['vendor_id'] = row['vendor_id']
            po_doc['vendor_name'] = row['vendor_name']
            po_doc['amount'] = row['amount']
            po_doc['date'] = datetime.now()
            po_doc['delivery_date'] = datetime.now() + timedelta(days=30)
            
            # Add payment terms based on amount
            if row['amount'] < FINANCIAL_THRESHOLDS['small_invoice']:
                po_doc['payment_terms'] = '15 days'
            elif row['amount'] < FINANCIAL_THRESHOLDS['medium_invoice']:
                po_doc['payment_terms'] = '30 days'
            else:
                po_doc['payment_terms'] = '45 days'
            
            # Add items if present
            if 'items' in row:
                po_doc['items'] = row['items']
            else:
                po_doc['items'] = f"Supply of materials as per RFQ"
            
            # Add approval requirements
            po_doc['requires_approval'] = row['amount'] > FINANCIAL_THRESHOLDS['critical_invoice']
            
            documents.append(po_doc)
        
        return documents
    
    def _generate_contracts(self, data: pd.DataFrame) -> list:
        """
        Generates contract documents from vendor data.
        
        Args:
            data: Vendor contract DataFrame
            
        Returns:
            List of generated contract documents
        """
        documents = []
        
        for idx, row in data.iterrows():
            contract_number = f"CONT-{datetime.now().strftime('%Y%m%d')}-{idx+1:04d}"
            
            # Fill template
            contract_doc = self.contract_template.copy()
            contract_doc['contract_number'] = contract_number
            contract_doc['vendor_id'] = row['vendor_id']
            contract_doc['vendor_name'] = row['vendor_name']
            contract_doc['contract_value'] = row['amount']
            contract_doc['start_date'] = datetime.now()
            contract_doc['end_date'] = datetime.now() + timedelta(days=365)
            
            # Add mandatory clauses
            contract_doc['payment_clause'] = f"Payment within {self._get_payment_days(row['amount'])} days"
            contract_doc['penalty_clause'] = "Penalty @ 0.5% per week delay"
            contract_doc['termination_clause'] = "Termination with 30 days notice"
            contract_doc['warranty_clause'] = "12 months warranty"
            contract_doc['liability_clause'] = "Limited to contract value"
            
            # GCC/SCC compliance
            contract_doc['gcc_compliant'] = True
            contract_doc['scc_included'] = True
            
            # Approval workflow
            if row['amount'] > FINANCIAL_THRESHOLDS['critical_invoice']:
                contract_doc['approval_level'] = 'Board'
            elif row['amount'] > FINANCIAL_THRESHOLDS['large_invoice']:
                contract_doc['approval_level'] = 'Director'
            else:
                contract_doc['approval_level'] = 'Manager'
            
            documents.append(contract_doc)
        
        return documents
    
    def _generate_both(self, data: pd.DataFrame) -> list:
        """
        Generates both POs and contracts.
        
        Args:
            data: Vendor data DataFrame
            
        Returns:
            Combined list of documents
        """
        pos = self._generate_purchase_orders(data)
        contracts = self._generate_contracts(data)
        
        return pos + contracts
    
    def _validate_documents(self, documents: list, doc_type: str) -> list:
        """
        Validates generated documents.
        
        Args:
            documents: List of generated documents
            doc_type: Document type
            
        Returns:
            Documents with validation flags
        """
        validated = []
        
        for doc in documents:
            is_valid = True
            errors = []
            
            # Check required fields
            if doc_type == 'po':
                required = ['po_number', 'vendor_id', 'vendor_name', 'amount', 'payment_terms']
            else:
                required = ['contract_number', 'vendor_id', 'vendor_name', 'contract_value']
            
            for field in required:
                if field not in doc or doc[field] is None:
                    is_valid = False
                    errors.append(f"Missing {field}")
            
            # Check amount validity
            amount_field = 'amount' if doc_type == 'po' else 'contract_value'
            if amount_field in doc and doc[amount_field] <= 0:
                is_valid = False
                errors.append("Invalid amount")
            
            doc['is_valid'] = is_valid
            doc['validation_errors'] = errors
            validated.append(doc)
        
        return validated
    
    def _generate_statistics(self, documents: list) -> Dict[str, Any]:
        """
        Generates document generation statistics.
        
        Args:
            documents: List of validated documents
            
        Returns:
            Statistics dictionary
        """
        amounts = [d.get('amount', d.get('contract_value', 0)) for d in documents]
        
        stats = {
            'total_value': sum(amounts),
            'average_value': sum(amounts) / len(amounts) if amounts else 0,
            'max_value': max(amounts) if amounts else 0,
            'min_value': min(amounts) if amounts else 0,
            'high_value_count': sum(1 for a in amounts if a > FINANCIAL_THRESHOLDS['critical_invoice'])
        }
        
        return stats
    
    def _get_po_template(self) -> Dict[str, Any]:
        """
        Returns PO template structure.
        """
        return {
            'po_number': None,
            'vendor_id': None,
            'vendor_name': None,
            'amount': None,
            'date': None,
            'delivery_date': None,
            'payment_terms': None,
            'items': None,
            'requires_approval': False
        }
    
    def _get_contract_template(self) -> Dict[str, Any]:
        """
        Returns contract template structure.
        """
        return {
            'contract_number': None,
            'vendor_id': None,
            'vendor_name': None,
            'contract_value': None,
            'start_date': None,
            'end_date': None,
            'payment_clause': None,
            'penalty_clause': None,
            'termination_clause': None,
            'warranty_clause': None,
            'liability_clause': None,
            'gcc_compliant': None,
            'scc_included': None,
            'approval_level': None
        }
    
    def _get_payment_days(self, amount: float) -> int:
        """
        Determines payment days based on amount.
        
        Args:
            amount: Contract/PO amount
            
        Returns:
            Payment days
        """
        if amount < FINANCIAL_THRESHOLDS['small_invoice']:
            return 15
        elif amount < FINANCIAL_THRESHOLDS['medium_invoice']:
            return 30
        else:
            return 45


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING STAGE 2: DOCUMENT AUTOMATION")
    print("="*60)
    
    # Create sample vendor data
    sample_data = pd.DataFrame({
        'vendor_id': ['V001', 'V002', 'V003', 'V004', 'V005'],
        'vendor_name': ['ABC Construction', 'XYZ Suppliers', 'PQR Materials', 
                        'LMN Contractors', 'DEF Industries'],
        'amount': [250000, 5500000, 1200000, 450000, 8000000],
        'items': ['Steel & Cement', 'Electrical Equipment', 'Plumbing Materials',
                  'HVAC Systems', 'Complete Civil Work']
    })
    
    # Test PO Generation
    print("\n📄 Testing Purchase Order Generation...")
    stage = DocumentAutomationStage()
    po_result = stage.execute(sample_data, doc_type='po')
    
    print("\n📊 PO RESULTS:")
    print(f"   Status: {po_result['status']}")
    print(f"   Total POs: {po_result['results']['total_documents']}")
    print(f"   Valid: {po_result['results']['valid_documents']}")
    print(f"   Total Value: {format_currency(po_result['results']['statistics']['total_value'])}")
    print(f"   Duration: {po_result['duration_seconds']}s")
    
    print("\n📋 Sample PO:")
    sample_po = po_result['results']['documents'][0]
    print(f"   PO Number: {sample_po['po_number']}")
    print(f"   Vendor: {sample_po['vendor_name']}")
    print(f"   Amount: {format_currency(sample_po['amount'])}")
    print(f"   Payment Terms: {sample_po['payment_terms']}")
    print(f"   Requires Approval: {sample_po['requires_approval']}")
    
    # Test Contract Generation
    print("\n📑 Testing Contract Generation...")
    contract_result = stage.execute(sample_data, doc_type='contract')
    
    print("\n📊 CONTRACT RESULTS:")
    print(f"   Status: {contract_result['status']}")
    print(f"   Total Contracts: {contract_result['results']['total_documents']}")
    print(f"   Valid: {contract_result['results']['valid_documents']}")
    print(f"   Total Value: {format_currency(contract_result['results']['statistics']['total_value'])}")
    
    print("\n📋 Sample Contract:")
    sample_contract = contract_result['results']['documents'][0]
    print(f"   Contract Number: {sample_contract['contract_number']}")
    print(f"   Vendor: {sample_contract['vendor_name']}")
    print(f"   Value: {format_currency(sample_contract['contract_value'])}")
    print(f"   Approval Level: {sample_contract['approval_level']}")
    print(f"   GCC Compliant: {sample_contract['gcc_compliant']}")
    
    print("\n" + "="*60)
    print("✅ STAGE 2 TEST COMPLETE")
    print("="*60)
