#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG pipeline for answering questions from 10-K PDFs (Apple + Tesla).
"""


import os
import re
import json
import numpy as np
import torch
import pdfplumber
import faiss
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# DATA_DIR = "/Users/sjain/Documents/Projects/ABB_Assignment/RAG/content"
DATA_DIR = "put your data directory here"

APPLE_PDF = os.path.join(DATA_DIR, "10-Q4-2024-As-Filed.pdf")
TESLA_PDF = os.path.join(DATA_DIR, "tsla-20231231-gen.pdf")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# Chunk data structure
@dataclass
class Chunk:
    text: str
    doc_name: str
    page: int
    section: Optional[str] = None
    chunk_id: int = 0
    
    def citation(self):
        parts = [self.doc_name]
        if self.section:
            parts.append(self.section)
        parts.append(f"p. {self.page}")
        return str(parts)


# Section detection patterns
SECTION_PATTERNS = [
    (r"(?i)\bITEM\s*1[A-B]?\b", "Item 1"), (r"(?i)\bITEM\s*1A\b", "Item 1A"),
    (r"(?i)\bITEM\s*1B\b", "Item 1B"), (r"(?i)\bITEM\s*1C\b", "Item 1C"),
    (r"(?i)\bITEM\s*2\b", "Item 2"), (r"(?i)\bITEM\s*3\b", "Item 3"),
    (r"(?i)\bITEM\s*4\b", "Item 4"), (r"(?i)\bITEM\s*5\b", "Item 5"),
    (r"(?i)\bITEM\s*6\b", "Item 6"), (r"(?i)\bITEM\s*7[A]?\b", "Item 7"),
    (r"(?i)\bITEM\s*7A\b", "Item 7A"), (r"(?i)\bITEM\s*8\b", "Item 8"),
    (r"(?i)\bITEM\s*9[A-C]?\b", "Item 9"), (r"(?i)\bITEM\s*10\b", "Item 10"),
    (r"(?i)\bITEM\s*11\b", "Item 11"), (r"(?i)\bITEM\s*12\b", "Item 12"),
    (r"(?i)\bITEM\s*13\b", "Item 13"), (r"(?i)\bITEM\s*14\b", "Item 14"),
    (r"(?i)\bITEM\s*15\b", "Item 15"), (r"(?i)\bITEM\s*16\b", "Item 16"),
]

def detect_section(text):
    header = text[:500]
    for pattern, name in SECTION_PATTERNS:
        if re.search(pattern, header):
            return name
    return None


# PDF parsing functions
def format_table(table):
    if not table or len(table) < 2:
        return ""
    headers = [str(h).strip() if h else "" for h in (table[0] or [])]
    rows = []
    for row in table[1:]:
        if not row:
            continue
        parts = []
        for i, cell in enumerate(row):
            val = str(cell).strip() if cell else ""
            if val:
                if i < len(headers) and headers[i]:
                    parts.append(f"{headers[i]}: {val}")
                else:
                    parts.append(val)
        if parts:
            rows.append(" | ".join(parts))
    return "\n".join(rows)

def parse_pdf(pdf_path, doc_name):
    pages = []
    current_section = None
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            if tables:
                table_text = "\n\n".join(format_table(t) for t in tables if t)
                text = text + "\n\n" + table_text
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text).strip()
            section = detect_section(text)
            if section:
                current_section = section
            pages.append({"text": text, "page": i + 1, "section": current_section})
    return pages


# Text chunking
def split_text(text, chunk_size=1000, overlap=200):
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    for sep in ["\n\n", "\n", ". ", " "]:
        if sep in text:
            break
    else:
        sep = ""
    
    parts = text.split(sep) if sep else list(text)
    chunks = []
    current = []
    current_len = 0
    
    for part in parts:
        part_len = len(part) + len(sep)
        if current_len + part_len > chunk_size and current:
            chunks.append(sep.join(current))
            overlap_text = sep.join(current)
            if len(overlap_text) > overlap:
                overlap_text = overlap_text[-overlap:]
                dot = overlap_text.find(". ")
                if dot != -1 and dot < len(overlap_text) - 10:
                    overlap_text = overlap_text[dot + 2:]
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)
        current.append(part)
        current_len += part_len
    
    if current:
        chunks.append(sep.join(current))
    return chunks

def chunk_document(pages, doc_name, chunk_size=1000, overlap=200):
    chunks = []
    chunk_id = 0
    for page in pages:
        for text in split_text(page["text"], chunk_size, overlap):
            if text.strip():
                chunks.append(Chunk(text=text, doc_name=doc_name, page=page["page"], section=page["section"], chunk_id=chunk_id))
                chunk_id += 1
    return chunks


# Load and chunk both PDFs
print("Parsing Apple 10-K...")
apple_pages = parse_pdf(APPLE_PDF, "Apple 10-K")
apple_chunks = chunk_document(apple_pages, "Apple 10-K")
print(f"  -> {len(apple_chunks)} chunks")

print("Parsing Tesla 10-K...")
tesla_pages = parse_pdf(TESLA_PDF, "Tesla 10-K")
tesla_chunks = chunk_document(tesla_pages, "Tesla 10-K")
print(f"  -> {len(tesla_chunks)} chunks")

all_chunks = apple_chunks + tesla_chunks
for i, chunk in enumerate(all_chunks):
    chunk.chunk_id = i
print(f"\nTotal: {len(all_chunks)} chunks")


# Vector store
class VectorStore:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.dim = embed_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []
    
    def add(self, chunks, batch_size=32):
        print(f"Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        embeddings = self.embed_model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"  -> Done! Total: {len(self.chunks)} chunks")
    
    def search(self, query, k=5):
        q_embed = self.embed_model.encode([query], normalize_embeddings=True).astype('float32')
        scores, indices = self.index.search(q_embed, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results


# Retriever with re-ranking
class Retriever:
    def __init__(self, store, reranker_model=None):
        self.store = store
        self.reranker = None
        if reranker_model:
            print(f"Loading re-ranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
    
    def retrieve(self, query, top_k=5, initial_k=20):
        candidates = self.store.search(query, k=initial_k if self.reranker else top_k)
        if self.reranker and candidates:
            pairs = [(query, c.text) for c, _ in candidates]
            scores = self.reranker.predict(pairs)
            reranked = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked[:top_k]
        return candidates
    
    def get_context(self, query, top_k=5):
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "", [], []
        context_parts = []
        sources = []
        chunks = []
        for chunk, score in results:
            src = f"[Source: {chunk.doc_name}"
            if chunk.section:
                src += f", {chunk.section}"
            src += f", p. {chunk.page}]"
            context_parts.append(f"{src}\n{chunk.text}")
            sources.append(chunk.citation())
            chunks.append(chunk)
        return "\n\n---\n\n".join(context_parts), sources, chunks


# Build vector store and retriever
print(f"Loading embedding model: {EMBEDDING_MODEL}")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

store = VectorStore(embed_model)
store.add(all_chunks)

retriever = Retriever(store, RERANKER_MODEL)


# Load LLM
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading LLM on {device}...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)
print("Done!")


# Generation function
def generate(prompt, max_tokens=200):
    max_input = 2048 - max_tokens - 50
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# Prompt template and helpers
PROMPT_TEMPLATE = """<|system|>
Answer using ONLY the context. Cite as ["Doc", "Section", "p. X"].
Not found: "Not specified in the document."
Unanswerable: "This question cannot be answered based on the provided documents."</s>
<|user|>
Context:
{context}

Question: {question}</s>
<|assistant|>
"""

OUT_OF_SCOPE = [
    r"(?i)\b(forecast|predict|projection|future|will be|going to)\b.*\b(stock|price|revenue|earnings)\b",
    r"(?i)\bstock\s*price\s*(forecast|prediction|for|in)\s*\d{4}\b",
    r"(?i)\b(2025|2026|2027)\b",
    r"(?i)\bwhat\s+color\b",
    r"(?i)\b(cfo|ceo|executive).*\b(2025|current|today|now)\b",
]

def is_out_of_scope(question):
    for pattern in OUT_OF_SCOPE:
        if re.search(pattern, question):
            return True
    return False

def clean_answer(text):
    text = re.sub(r'<\|.*?\|>', '', text)
    text = text.replace('</s>', '').replace('<s>', '')
    return ' '.join(text.split()).strip()

def answer_not_found(answer):
    phrases = ["not specified", "not mentioned", "not found", "no information", "does not mention", "doesn't mention", "not available", "cannot be determined", "cannot answer"]
    return any(p in answer.lower() for p in phrases)


# Main answer function
def answer_question(query):
    # Check if out of scope
    if is_out_of_scope(query):
        return {"answer": "This question cannot be answered based on the provided documents.", "sources": []}
    
    # Get context
    context, sources, chunks = retriever.get_context(query, top_k=3)
    if not context.strip():
        return {"answer": "Not specified in the document.", "sources": []}
    
    # Truncate context
    tokens = tokenizer.encode(context)
    if len(tokens) > 1200:
        context = tokenizer.decode(tokens[:1200], skip_special_tokens=True)
    
    # Generate
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    try:
        answer = clean_answer(generate(prompt))
        if answer_not_found(answer):
            return {"answer": "Not specified in the document.", "sources": []}
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": "Error generating response.", "sources": []}

