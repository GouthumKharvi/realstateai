import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ai_engine.llm_engine import LLMEngine

llm = LLMEngine(provider="mock")
print("🧪 1️⃣ MOCK TEST (No API)")
print("="*50)
print(llm.answer_question("What are contract payment terms?"))
print("\n✅ MOCK PASS ✓")

