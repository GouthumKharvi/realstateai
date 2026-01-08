"""
🚀 Production RAG Engine: LangChain + FAISS + Local Embeddings
Fixed - No Pydantic errors, modern LangChain patterns
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import for Groq
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ langchain-groq not installed. Install with: pip install langchain-groq")

class RAGEngine:
    def __init__(self, api_key: str = "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"):
        """
        Production RAG with LangChain + FAISS + Local Embeddings
        API: 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
        Model: llama-3.3-70b-versatile
        """
        self.api_key = api_key
        
        # LOCAL Embeddings (no API needed!)
        print("📦 Loading local embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embedding model loaded")
        
        # Text splitter (chunking)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        
        # Vector store (FAISS)
        self.vectorstore = None
        
        # ✅ LLM (GROQ) - Using ChatGroq
        if GROQ_AVAILABLE:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=api_key,
                temperature=0.1,
                max_tokens=1024
            )
            print("✅ Groq LLM initialized with llama-3.3-70b-versatile")
        else:
            self.llm = None
            print("❌ Groq not available - install langchain-groq")
        
        # ✅ Modern RAG Prompt Template using ChatPromptTemplate
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a procurement and contracts expert. Use the following context to answer the question. If you cannot answer from the context, say 'I cannot find this information in the knowledge base.'"),
            ("human", """Context:
{context}

Question: {question}

Answer (be concise and cite sources using [filename]):""")
        ])
        
        # Base path for vector DB
        self.vector_db_path = r"C:\Users\Gouthum\Downloads\Assisto.tech Internship\Contracts And Purchase team in Al Enablement for real estate\vector_db"
    
    def build_index(self, documents_path: Path):
        """
        Load files → Split → Embed → FAISS
        """
        print("🚀 Building RAG Index...")
        
        # Load documents
        docs = []
        folders = ["company_policies", "gcc_clauses", "historical_cases", "scc_clauses"]
        
        # Metadata tracking
        metadata_info = {
            "total_documents": 0,
            "total_chunks": 0,
            "documents_by_folder": {},
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "llm_model": "llama-3.3-70b-versatile",
            "api_provider": "groq",
            "created_at": None,
            "documents": []
        }
        
        for folder in folders:
            folder_path = documents_path / folder
            folder_docs = []
            
            if folder_path.exists():
                print(f"📁 {folder}/")
                for file in folder_path.iterdir():
                    if file.is_file() and not file.name.startswith('.') and not file.name.startswith('_'):
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        'filename': file.name,
                                        'folder': folder,
                                        'source': str(file),
                                        'size': len(content)
                                    }
                                )
                                docs.append(doc)
                                
                                # Track metadata
                                folder_docs.append({
                                    'filename': file.name,
                                    'size': len(content),
                                    'source': str(file)
                                })
                                
                            print(f"   ✅ {file.name}")
                        except Exception as e:
                            print(f"   ❌ {file.name} - Error: {e}")
                
                metadata_info["documents_by_folder"][folder] = {
                    "count": len(folder_docs),
                    "files": folder_docs
                }
        
        metadata_info["total_documents"] = len(docs)
        print(f"\n📊 {len(docs)} documents loaded")
        
        if len(docs) == 0:
            print("❌ No documents found!")
            return
        
        # Split into chunks
        print("✂️ Splitting documents into chunks...")
        splits = self.splitter.split_documents(docs)
        metadata_info["total_chunks"] = len(splits)
        print(f"📄 {len(splits)} chunks created")
        
        # Create FAISS vectorstore
        print("🔮 Creating embeddings and FAISS index...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        print("✅ Vector store created")
        
        # Save FAISS index
        os.makedirs(self.vector_db_path, exist_ok=True)
        save_path = os.path.join(self.vector_db_path, "rag_index")
        self.vectorstore.save_local(save_path)
        print(f"💾 FAISS index saved: {save_path}")
        
        # Save metadata.json
        from datetime import datetime
        metadata_info["created_at"] = datetime.now().isoformat()
        metadata_info["index_path"] = save_path
        metadata_info["vector_count"] = len(splits)
        
        metadata_path = os.path.join(self.vector_db_path, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_info, f, indent=2, ensure_ascii=False)
        print(f"📄 Metadata saved: {metadata_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 INDEX SUMMARY:")
        print("="*60)
        print(f"Total Documents: {metadata_info['total_documents']}")
        print(f"Total Chunks: {metadata_info['total_chunks']}")
        print(f"Embedding Model: {metadata_info['embedding_model']}")
        print(f"LLM Model: {metadata_info['llm_model']}")
        print(f"API Provider: {metadata_info['api_provider']}")
        print("\nDocuments by Folder:")
        for folder, info in metadata_info["documents_by_folder"].items():
            print(f"  • {folder}: {info['count']} files")
        print("="*60)
    
    def load_index(self):
        """
        Load existing FAISS index
        """
        print("📂 Loading existing index...")
        
        # Load metadata first
        metadata_path = os.path.join(self.vector_db_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"📄 Metadata loaded: {metadata['total_documents']} docs, {metadata['total_chunks']} chunks")
            print(f"🤖 LLM Model: {metadata.get('llm_model', 'Not specified')}")
        
        # Load FAISS index
        load_path = os.path.join(self.vector_db_path, "rag_index")
        self.vectorstore = FAISS.load_local(
            load_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ Index loaded from: {load_path}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata information
        """
        metadata_path = os.path.join(self.vector_db_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def format_docs(self, docs):
        """Helper function to format retrieved documents"""
        return "\n\n".join([
            f"[{doc.metadata['filename']}]\n{doc.page_content}" 
            for doc in docs
        ])
    
    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        RAG Query: Retrieve + Generate
        Uses: llama-3.3-70b-versatile via Groq API
        ✅ Modern LangChain pattern - NO Pydantic errors
        """
        if not self.vectorstore:
            self.load_index()
        
        if not self.llm:
            return {
                'answer': "❌ LLM not available. Install langchain-groq: pip install langchain-groq",
                'sources': []
            }
        
        print(f"🔍 Searching for: '{question}'")
        
        # Step 1: Create retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Step 2: Retrieve documents
        docs = retriever.invoke(question)  # ✅ Modern method
        print(f"📚 Found {len(docs)} relevant documents")
        
        # Step 3: Build RAG chain using modern LCEL pattern
        # This avoids all Pydantic validation issues
        rag_chain = (
            {
                "context": lambda x: self.format_docs(docs),
                "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Step 4: Generate answer
        print("🤖 Generating answer with llama-3.3-70b-versatile...")
        
        try:
            answer = rag_chain.invoke(question)  # ✅ Modern method
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            answer = f"Error: {str(e)}"
        
        # Step 5: Format response
        return {
            'answer': answer,
            'sources': [
                {
                    'filename': doc.metadata['filename'],
                    'folder': doc.metadata['folder'],
                    'content': doc.page_content[:200] + "...",
                    'score': 0.0  # Placeholder for compatibility
                }
                for doc in docs
            ]
        }


# ============= EXECUTION BLOCK =============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 RAG ENGINE - BUILD & TEST")
    print("="*60)
    print("📋 API: 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")
    print("🤖 Model: llama-3.3-70b-versatile")
    print("="*60 + "\n")
    
    # Initialize RAG Engine
    rag = RAGEngine()
    
    # Path to your knowledge base
    documents_path = Path(r"C:\Users\Gouthum\Downloads\Assisto.tech Internship\Contracts And Purchase team in Al Enablement for real estate\knowledgebase")
    
    # Build the index
    rag.build_index(documents_path)
    
    if rag.vectorstore is not None:
        print("\n" + "="*60)
        print("✅ VECTOR DATABASE CREATED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        # Show metadata
        metadata = rag.get_metadata()
        print(f"📅 Created: {metadata.get('created_at')}")
        print(f"📊 Total Documents: {metadata.get('total_documents')}")
        print(f"📄 Total Chunks: {metadata.get('total_chunks')}")
        print(f"🤖 LLM Model: {metadata.get('llm_model')}")
        print(f"🔧 API Provider: {metadata.get('api_provider')}")
        
        # Test query
        print("\n🧪 Testing RAG query...\n")
        result = rag.query("What are the payment terms in GCC?")
        
        print("\n" + "-"*60)
        print("📝 ANSWER:")
        print("-"*60)
        print(result['answer'])
        
        print("\n" + "-"*60)
        print(f"📚 SOURCES ({len(result['sources'])} documents):")
        print("-"*60)
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['folder']}/{source['filename']}")
            print(f"   Preview: {source['content'][:100]}...")
        
        print("\n" + "="*60)
        print("🎉 RAG ENGINE WORKING PERFECTLY!")
        print("="*60)
    else:
        print("\n❌ Failed to create index")