# RAG System for 10-K Documents (Apple + Tesla)

This project is a **Retrieval-Augmented Generation (RAG)** based Question Answering system for **Apple and Tesla 10-K annual reports**.

You ask a question → the system retrieves the most relevant chunks from the PDFs → re-ranks them → and generates a final answer using a small LLM.

---

## What This Project Does

Pipeline (high-level):

**User Question → Vector Search → Re-ranking → Answer Generation**

Key components:
- **PDF parsing** using 'pdfplumber'
- **Chunking** text into overlapping segments
- **Embeddings** using 'BAAI/bge-small-en-v1.5'
- **Vector database** using 'FAISS'
- **Re-ranking** using 'cross-encoder/ms-marco-MiniLM-L-6-v2'
- **Answer generation** using 'TinyLlama-1.1B-Chat'

---

## Project Structure

Recommended folder structure:

rag-10k-rag-qa/
├── README.md
├── requirements.txt
├── design_report.md
├── rag.ipynb
└── content/
    ├── apple_10k.pdf
    └── tesla_10k.pdf
