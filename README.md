# AI Enablement for Contracts & Procurement (Real Estate)
https://realestateaii.streamlit.app/
## Overview

This project implements a **production-ready AI architecture** for enabling the **Contracts and Purchase lifecycle** in real estate projects.

The solution covers **11 end-to-end process stages**, from vendor development to fraud detection, using a **single generic AI pipeline** that is reused across all stages through configuration.

The focus is on:
- Explainable AI
- Minimal code duplication
- Business-aligned logic
- Human-in-the-loop decision making

---

## Key Objectives

- Automate document-heavy procurement workflows
- Detect risks, anomalies, and deviations early
- Support (not replace) procurement and legal teams
- Maintain auditability and compliance
- Enable fast prototyping with mock datasets
- Be production-ready from Day 1

---

## Process Stages Covered (11)

1. Vendor Development  
2. Document Automation  
3. RFQ / RFP – Anomaly Detection  
4. Vendor Shortlisting & Selection Optimization  
5. Negotiation (Strategy Notes & Counter-offers)  
6. Contract Review (GCC / SCC Deviation – RAG Flags)  
7. Variations / Change Orders (Cost & Scope Impact)  
8. Contract Analysis & Risk Assessment  
9. Predictive Analytics  
10. Keeping Records & Benchmarking  
11. Fraud / Irregularity Detection  

All stages use the **same core pipeline**, with only configuration changes.

---

## High-Level Architecture

**Core Principle:**  
> One Generic Pipeline + Multiple Configurations = All Stages

### Architecture Layers

1. **User Interface**
   - Streamlit-based UI
   - File upload, stage selection, results display

2. **Orchestration Layer**
   - Routes execution to the selected process stage
   - Manages session state and configs

3. **Generic AI Pipeline**
   - Data Ingestion
   - OCR (for documents)
   - Data Processing & NLP
   - Rule Engine
   - ML Engine
   - LLM (RAG-based explanations)
   - Output & Visualization

4. **Configuration Layer**
   - One JSON/YAML file per stage
   - Defines rules, thresholds, models, outputs

5. **Knowledge Base (for RAG)**
   - GCC clauses
   - SCC clauses
   - Company policies
   - Historical cases

---

## Generic Pipeline Flow

Input (CSV / PDF / Excel)
↓
OCR (if document-based)
↓
Text Cleaning & NLP
↓
Rule Engine (Business Logic)
↓
ML Models (Optional)
↓
RAG + LLM (Explanation Only)
↓
Dashboards & Reports

yaml
Copy code

---

## AI Techniques Used

### OCR
- Extract text from scanned contracts, invoices, RFQs
- Tools: Tesseract, PyPDF2, pdf2image

### NLP
Used for **information extraction**, not free-text generation:
- Regex-based parsing
- Clause detection
- Keyword tagging
- Section identification

Libraries:
- spaCy
- Python regex
- Custom parsers

---

## Rule-Based Logic

Rules are used where **business clarity is required**:
- Approval thresholds
- Risk flags
- Compliance checks
- Red / Amber / Green classification

Rules are defined in configuration files to keep logic transparent and auditable.

---

## Machine Learning Models

ML is applied only where patterns exist in data:

| Use Case | Model |
|--------|------|
| Vendor scoring | Logistic Regression |
| Risk classification | Random Forest |
| Fraud detection | Isolation Forest |
| Time forecasting | ARIMA / Regression |

Framework:
- scikit-learn

ML outputs are **assistive signals**, not final decisions.

---

## LLM Usage (GenAI + RAG)

LLMs are **not used for decision-making**.

They are used only to:
- Explain why something was flagged
- Summarize risks or deviations
- Generate human-readable insights

### RAG Setup
- FAISS vector database
- Sentence-transformer embeddings
- Knowledge base from policy and contract documents
- LangChain for orchestration

LLM options:
- Mock LLM (for demo)
- OpenAI / Groq / Local LLM (pluggable)

---

## Why Speech-to-Text / TTS / Translation Are Not Used

This system is:
- Document-centric
- Data-heavy
- Compliance-focused

Procurement workflows rely on:
- PDFs
- Emails
- Structured records

Voice and translation services are more suitable for **contact centers**, not contracts and procurement analytics.

---

## Folder Structure

ai_enablement_project/
│
├── app.py
├── config/
│ ├── stage_1_vendor_dev.json
│ ├── ...
│ └── stage_11_fraud.json
│
├── core/
│ ├── data_ingestion.py
│ ├── ocr_layer.py
│ ├── data_processing.py
│ ├── rule_engine.py
│ ├── ml_engine.py
│ ├── rag_engine.py
│ ├── llm_engine.py
│ └── output_generator.py
│
├── stages/
│ └── base_stage.py
│
├── knowledge_base/
│ ├── gcc_clauses/
│ ├── scc_clauses/
│ ├── company_policies/
│ └── historical_cases/
│
├── vector_db/
├── mock_data/
├── models/
└── requirements.txt

yaml
Copy code

---

## Mock Data Strategy

- All demos use **synthetic mock datasets**
- Data is generated programmatically
- Same structure as real ERP / DMS data
- Easy migration to production systems later

---

## What Is Implemented vs Planned

### Implemented (Prototype Level)
- Generic architecture
- RAG vector database
- OCR pipeline
- Rule engine
- ML stubs
- Multiple stages wired to same pipeline

### Planned (Next Phase)
- ERP integration
- Real-time workflows
- Approval automation
- Model retraining pipelines
- Access control and audit logs

---

## Key Design Philosophy

- Human-in-the-loop
- Explainable AI
- Minimal code
- Maximum reuse
- Business-first AI

---

## Summary

This project demonstrates how **one well-designed AI architecture** can power an entire procurement and contract lifecycle with:
- Transparency
- Scalability
- Governance
- Real-world practicality

It is designed to **work today with mock data** and **scale tomorrow into production*
