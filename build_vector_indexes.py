#!/usr/bin/env python3
"""
🚀 FIXED Vector Index Builder
"""
import sys
import os
sys.path.insert(0, '.')  

from pathlib import Path
import numpy as np
from core.ai_engine.rag_engine import VectorDB, embed_text

KNOWLEDGE_PATH = Path("knoweldgebase")
VECTOR_DB_PATH = Path("vector_db")

def main():
    print("🚀 BUILDING KNOWLEDGE BASE")
    print("="*60)
    
    folders = {
        "company_policies": ["vendor_selection_policy.txt", "contract_approval_policy.txt", "procurement_policy.txt"],
        "gcc_clauses": ["liability_clauses.txt", "payment_terms.txt", "penalty_clauses.txt", "termination_clauses.txt"],
        "historical_cases": ["past_disputes.txt", "risk_incidents.txt", "successful_negotiations.txt"],
        "scc_clauses": ["custom_terms.txt", "project_specific_clauses.txt"]
    }
    
    documents, metadata = [], []
    total_chars = 0
    
    for folder, files in folders.items():
        folder_path = KNOWLEDGE_PATH / folder
        print(f"\n📁 {folder}/")
        for filename in files:
            file_path = folder_path / filename
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    metadata.append({
                        'filename': filename,
                        'folder': folder,
                        'chars': len(content)
                    })
                    total_chars += len(content)
                    print(f"   ✅ {filename} ({len(content):,} chars)")
            except FileNotFoundError:
                print(f"   ⚠️  MISSING: {filename}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Files loaded: {len(documents)}/13")
    print(f"   Total chars: {total_chars:,}")
    
    if not documents:
        print("❌ NO FILES FOUND! Check knoweldgebase/ folder path")
        return
    
    # Generate embeddings
    print("\n🔄 Creating embeddings...")
    embeddings = np.array([embed_text(doc) for doc in documents])
    print(f"   ✅ Embeddings: {embeddings.shape}")
    
    # Build FAISS
    print("\n🏗️  Building FAISS index...")
    db = VectorDB(dimension=384)
    db.build_index(documents, embeddings, metadata)
    
    # Save
    VECTOR_DB_PATH.mkdir(exist_ok=True)
    db.save(str(VECTOR_DB_PATH), 'contracts_kb')
    
    print("\n🎉 VECTOR DATABASE READY!")
    print(f"📁 {VECTOR_DB_PATH}/contracts_kb.faiss")
    print("✅ RAG Chatbot ready to use!")

if __name__ == "__main__":
    main()
