from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import os

print("🚀 Simple RAG Test...")

# Load one sample document
sample_text = """
Payment Terms - GCC
Payment must be made within 30 days of invoice.
Late payment attracts 1% penalty per month.
Advance payment: 10% upon signing.
"""

doc = Document(page_content=sample_text, metadata={'source': 'test'})
print(f"✅ Document created: {len(sample_text)} chars")

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = splitter.split_documents([doc])
print(f"✅ Splits created: {len(splits)} chunks")

# Use local embeddings (no API needed)
print("📦 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
texts = [s.page_content for s in splits]
embeddings = model.encode(texts)
print(f"✅ Embeddings created: {embeddings.shape}")

# Create FAISS manually
import faiss
import numpy as np

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))
print(f"✅ FAISS index created: {index.ntotal} vectors")

# Save
save_path = r"C:\Users\Gouthum\Downloads\Assisto.tech Internship\Contracts And Purchase team in Al Enablement for real estate\vector_db\test_index"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
faiss.write_index(index, f"{save_path}.faiss")
print(f"✅ Saved: {save_path}.faiss")

# Test load
loaded_index = faiss.read_index(f"{save_path}.faiss")
print(f"✅ Loaded: {loaded_index.ntotal} vectors")

print("\n🎉 SUCCESS! Your system works!")