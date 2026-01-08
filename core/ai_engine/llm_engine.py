"""
LLM Engine - FIXED FOR llama-3.3-70b-versatile
============================================
Your API: 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
Your Model: llama-3.3-70b-versatile
"""

from typing import Optional
import textwrap
import random

# ============================================================
# PROMPT TEMPLATE
# ============================================================
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

# ============================================================
# MOCK LLM (OFFLINE FALLBACK)
# ============================================================
class MockLLM:
    def __init__(self, name: str = "mock-llm"):
        self.name = name
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        seed = abs(hash(prompt)) % (2**32)
        random.seed(seed)
        boilerplate = [
            "Based on the provided context, the key points are:",
            "From the available information, the following can be inferred:",
            "Considering the context and data, the analysis is as follows:"
        ]
        choice = random.choice(boilerplate)
        return (
            choice + "\n\n"
            + f"- Mock response from {self.name}.\n"
            + "- Replace with real API for production.\n"
        )

# ============================================================
# MAIN LLM ENGINE - YOUR MODEL: llama-3.3-70b-versatile
# ============================================================
class LLMEngine:
    """
    Production LLM Engine for llama-3.3-70b-versatile via Groq
    """
    def __init__(
        self, 
        provider: str = "mock", 
        model: str = "llama-3.3-70b-versatile",  # ← YOUR MODEL
        api_key: Optional[str] = None
    ):
        """
        Args:
            provider: 'mock' (offline) or 'groq' (your API)
            model: YOUR MODEL - llama-3.3-70b-versatile
            api_key: YOUR API KEY
        """
        self.provider = provider
        self.model = model
        
        # ✅ YOUR API KEY (UPDATED)
        self.api_key = api_key or "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"
        
        if provider == "mock":
            self.client = MockLLM(name=model)
            print("✅ MockLLM initialized (NO API - Offline mode)")
        
        elif provider == "groq":
            try:
                from openai import OpenAI
                
                # ✅ GROQ CLIENT WITH YOUR API KEY
                self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key
                )
                
                print(f"✅ Groq API initialized")
                print(f"   📋 Model: {self.model}")
                print(f"   🔑 API Key: {self.api_key[:20]}...")
                
            except ImportError:
                print("❌ Install: pip install openai")
                self.client = MockLLM(name=model)
                self.provider = "mock"
        
        else:
            raise ValueError(f"Provider must be 'mock' or 'groq'. Got: {provider}")
    
    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Calls YOUR MODEL: llama-3.3-70b-versatile via Groq
        """
        if self.provider == "mock":
            return self.client.generate(prompt, max_tokens=max_tokens)
        
        # ✅ GROQ API CALL WITH YOUR MODEL
        response = self.client.chat.completions.create(
            model=self.model,  # ← llama-3.3-70b-versatile
            messages=[
                {
                    "role": "system", 
                    "content": "You are a procurement and contracts expert. Provide accurate, grounded answers using ONLY the provided context. Cite sources [filename]. Be concise."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def generate_explanation(
        self, 
        context: str, 
        question: str, 
        max_tokens: int = 512
    ) -> str:
        """Generates explanation using context"""
        template = PromptTemplate(
            textwrap.dedent("""
                You are an AI assistant that explains procurement, contracts, and risk in clear simple language.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Instructions:
                - Use only the context above.
                - Explain step-by-step but keep it concise.
                - If something is unknown from context, say that explicitly.
                
                Answer:
            """).strip()
        )
        
        prompt = template.format(context=context, question=question)
        return self._call_llm(prompt, max_tokens=max_tokens)
    
    def summarize(
        self, 
        text: str, 
        style: str = "bullet", 
        max_tokens: int = 512
    ) -> str:
        """Summarizes text in requested style"""
        template = PromptTemplate(
            textwrap.dedent("""
                You are summarizing content for a procurement and contracts dashboard.
                
                Text to summarize:
                {text}
                
                Style: {style}
                
                Instructions:
                - Preserve key numbers, risk indicators, and decisions.
                - For 'bullet', return 3-7 bullets.
                - For 'short', return 2-3 concise sentences.
                - For 'detailed', return a structured explanation with headings.
                
                Summary:
            """).strip()
        )
        
        prompt = template.format(text=text[:6000], style=style)
        return self._call_llm(prompt, max_tokens=max_tokens)
    
    def answer_question(
        self, 
        question: str, 
        context: Optional[str] = None, 
        max_tokens: int = 1024
    ) -> str:
        """Answers question with optional context (RAG-style)"""
        if context:
            template = PromptTemplate(
                textwrap.dedent("""
                    You are an AI assistant helping with public procurement and contracts.
                    
                    Context:
                    {context}
                    
                    Question:
                    {question}
                    
                    Instructions:
                    - Use the context as primary source of truth.
                    - If context does not contain the answer, say "Not found in knowledge base."
                    - Keep the answer short and actionable.
                    - Cite sources: [filename]
                    
                    Answer:
                """).strip()
            )
            prompt = template.format(context=context, question=question)
        else:
            template = PromptTemplate(
                textwrap.dedent("""
                    You are an AI assistant for procurement and contract management.
                    
                    Question:
                    {question}
                    
                    Instructions:
                    - Answer from general domain knowledge.
                    - If uncertain, say that explicitly.
                    - Keep the answer concise.
                    
                    Answer:
                """).strip()
            )
            prompt = template.format(question=question)
        
        return self._call_llm(prompt, max_tokens=max_tokens)

# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print("🧪 TESTING YOUR LLM ENGINE")
    print("="*70)
    
    # TEST 1: MOCK (NO API)
    print("\n✅ TEST 1: MockLLM (Offline - No API)")
    print("-"*50)
    mock_llm = LLMEngine(provider="mock", model="llama-3.3-70b-versatile")
    mock_answer = mock_llm.answer_question("What are LD rates?")
    print("Mock Answer:", mock_answer[:100] + "...")
    
    # TEST 2: YOUR REAL API
    print("\n✅ TEST 2: Your API + llama-3.3-70b-versatile")
    print("-"*50)
    try:
        real_llm = LLMEngine(provider="groq", model="llama-3.3-70b-versatile")
        
        context = "LD rate is 0.5% per week, max 10% of contract value."
        real_answer = real_llm.answer_question("What is LD rate?", context=context)
        
        print("\n✅ SUCCESS! Your model llama-3.3-70b-versatile is working!")
        print("\nReal Answer:")
        print(real_answer)
        
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        print("💡 Using MockLLM as fallback")
    
    print("\n" + "="*70)
    print("🎉 LLM ENGINE READY!")
    print("📝 Usage: llm = LLMEngine(provider='groq')")
    print("="*70)