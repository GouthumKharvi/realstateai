#!/usr/bin/env python3
"""
🚀 Build RAG Index - Run ONCE
"""
import sys
import os
sys.path.insert(0, '.')  

from pathlib import Path
from core.ai_engine.rag_engine import RAGEngine

KNOWLEDGE_PATH = Path("knoweldgebase")  # My  folder

if __name__ == "__main__":
    rag = RAGEngine()
    rag.build_index(KNOWLEDGE_PATH)
    print("\n🎉 INDEX BUILT! Ready for chat!")
