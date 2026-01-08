"""
🚀 COMPLETE RAG-POWERED CHATBOT - FIXED UI & ERRORS
================================================================
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.ai_engine.rag_engine import RAGEngine
from core.ai_engine.llm_engine import LLMEngine

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Contracts & Procurement Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - FIXED COLORS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .source-box strong {
        color: #4fc3f7;
    }
    .source-box small {
        color: #b0b0b0;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    /* Fix expander text color */
    .streamlit-expanderHeader {
        color: #1f77b4 !important;
        font-weight: bold;
    }
    /* Make sure all text in source boxes is visible */
    div[data-testid="stExpander"] div {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZE ENGINES
# ============================================================

@st.cache_resource
def load_rag_engine():
    """Loads RAG engine with FAISS index"""
    try:
        rag = RAGEngine()
        rag.load_index()
        return rag
    except Exception as e:
        st.error(f"❌ Error loading RAG engine: {e}")
        st.info("💡 Run `python build_index.py` first")
        return None

@st.cache_resource
def load_llm_engine():
    """Loads LLM engine"""
    try:
        llm = LLMEngine(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key="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"
        )
        return llm
    except Exception as e:
        st.warning(f"⚠️ LLM API not available: {e}")
        st.info("🔄 Using MockLLM for demonstration")
        return LLMEngine(provider="mock")

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">🤖 AI Contracts & Procurement Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by RAG + LangChain + FAISS + Llama 3.3 70B</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📚 Knowledge Base")
    
    # Load engines
    rag_engine = load_rag_engine()
    llm_engine = load_llm_engine()
    
    # Status
    if rag_engine:
        st.success("✅ RAG Engine: **Loaded**")
    else:
        st.error("❌ RAG Engine: **Failed**")
    
    if llm_engine and llm_engine.provider == "groq":
        st.success("✅ LLM API: **Connected**")
        st.caption(f"🔑 Model: llama-3.3-70b-versatile")
    else:
        st.warning("⚠️ LLM API: **Mock Mode**")
    
    st.divider()
    
    # Knowledge base
    st.subheader("📁 Documents Indexed")
    
    with st.expander("🏢 Company Policies (3)", expanded=False):
        st.markdown("""
        - vendor_selection_policy.txt
        - contract_approval_policy.txt
        - procurement_policy.txt
        """)
    
    with st.expander("📋 GCC Clauses (4)", expanded=False):
        st.markdown("""
        - liability_clauses.txt
        - payment_terms.txt
        - penalty_clauses.txt
        - termination_clauses.txt
        """)
    
    with st.expander("📖 Historical Cases (3)", expanded=False):
        st.markdown("""
        - past_disputes.txt
        - risk_incidents.txt
        - successful_negotiations.txt
        """)
    
    with st.expander("📝 SCC Clauses (2)", expanded=False):
        st.markdown("""
        - custom_terms.txt
        - project_specific_clauses.txt
        """)
    
    st.divider()
    
    # Stats
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Docs", "12")
        st.metric("Chunks", "~359")
    with col2:
        st.metric("Embedding", "384D")
        st.metric("Model", "FAISS")
    
    st.divider()
    
    # Sample queries
    st.subheader("💡 Sample Queries")
    st.markdown("""
    **Contract Terms:**
    - "What are standard LD rates?"
    - "Explain payment retention terms"
    
    **Vendor Management:**
    - "What is vendor approval process?"
    - "Show vendor selection criteria"
    
    **Risk & Compliance:**
    - "What are force majeure conditions?"
    - "Show termination requirements"
    
    **Historical Cases:**
    - "Past dispute resolutions?"
    - "Successful negotiation examples?"
    """)
    
    st.divider()
    
    # Settings
    with st.expander("⚙️ Settings"):
        retrieval_k = st.slider("Sources to retrieve", 2, 10, 4)
        show_confidence = st.checkbox("Show confidence", value=True)
        show_sources = st.checkbox("Show sources", value=True)
    
    st.divider()
    st.caption("🚀 AI Enablement for Real Estate")
    st.caption("Built with LangChain + FAISS")

# ============================================================
# CHAT INTERFACE
# ============================================================

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """👋 **Welcome to your AI Contracts & Procurement Assistant!**

I can help you with:
- 📋 **Contract Clauses** (GCC, SCC, LD, payment terms)
- 🏢 **Company Policies** (vendor selection, procurement)
- 📊 **Historical Cases** (disputes, risks, negotiations)
- ⚖️ **Legal Terms** (liability, termination, warranties)

**Just ask your question!** 🎯"""
        }
    ]

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources with fixed styling
        if "sources" in message and show_sources:
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>[{i}] {source['filename']}</strong> ({source['folder']})
                        <br><br>
                        <small>{source['content'][:300]}...</small>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about contracts, GCC/SCC, policies..."):
    
    # Check engines
    if not rag_engine:
        st.error("❌ RAG engine not available")
        st.stop()
    
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                # Query RAG
                result = rag_engine.query(prompt, k=retrieval_k if 'retrieval_k' in locals() else 4)
                
                # Display answer
                st.markdown("**📝 Answer:**")
                st.markdown(result['answer'])
                
                # Confidence
                if show_confidence and 'retrieval_k' in locals() and len(result['sources']) > 0:
                    avg_score = sum(s.get('score', 0) for s in result['sources']) / len(result['sources'])
                    confidence = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.5 else "Low"
                    confidence_class = f"confidence-{confidence.lower()}"
                    st.markdown(f'<p class="{confidence_class}">Confidence: {confidence} ({avg_score:.2f})</p>', unsafe_allow_html=True)
                
                # Sources with fixed styling
                if show_sources if 'show_sources' in locals() else True:
                    with st.expander("📚 **Sources**", expanded=False):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>[{i}] {source['filename']}</strong> ({source['folder']})
                                <br><br>
                                <small>{source['content'][:300]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result['sources']
                })
                
                st.session_state.query_count += 1
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("💡 Try rephrasing or check index")

# ============================================================
# FOOTER
# ============================================================

st.divider()

# Stats
if st.session_state.query_count > 0:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Queries", st.session_state.query_count)
    with col2:
        st.metric("Documents", "12")
    with col3:
        st.metric("Method", "FAISS")

# Clear button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = st.session_state.messages[:1]
    st.session_state.query_count = 0
    st.rerun()

st.caption("🚀 Built with LangChain + FAISS + Groq")
st.caption("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))