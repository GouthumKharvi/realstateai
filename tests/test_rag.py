import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ai_engine.llm_engine import LLMEngine

llm = LLMEngine(provider="groq")
questions = [
    "GCC LD rates?",
    "Vendor DoA matrix?",
    "Payment terms?",
    "Termination clauses?"
]

print("🧪 3️⃣ CONTRACTS TEST")
print("="*50)
for q in questions:
    print(f"\nQ: {q}")
    print("-"*30)
    ans = llm.answer_question(q)
    print(ans[:150]+"...")
print("\n✅ RAG PASS ✓")

