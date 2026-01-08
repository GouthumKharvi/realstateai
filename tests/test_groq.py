import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ai_engine.llm_engine import LLMEngine

llm = LLMEngine(provider="groq")
context = "Mobilization advance: 10% with BG. LD: 0.5% per week, max 10%."
print("🧪 2️⃣ GROQ API TEST")
print("="*50)
print(llm.answer_question("Explain advance & LD", context=context))
print("\n✅ GROQ PASS ✓")
