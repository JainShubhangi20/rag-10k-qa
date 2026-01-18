#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluation.py
Evaluation script to test the RAG system on a small set of QA pairs.

Usage:
  python evaluation.py 
"""

import argparse
import os
from rag_simple_converted_cli import answer_question

# DATA_DIR = "/Users/sjain/Documents/Projects/ABB_Assignment/RAG/content"
DATA_DIR = "put your data directory here"

# Evaluation questions
EVAL_QUESTIONS = [
    {"question_id": 1, "question": "What was Apple's total revenue for the fiscal year ended September 28, 2024?"},
    {"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
    {"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
    {"question_id": 4, "question": "On what date was Apple's 10-K report for 2024 signed and filed with the SEC?"},
    {"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
    {"question_id": 6, "question": "What was Tesla's total revenue for the year ended December 31, 2023?"},
    {"question_id": 7, "question": "What percentage of Tesla's total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
    {"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
    {"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
    {"question_id": 10, "question": "What is the purpose of Tesla's 'lease pass-through fund arrangements'?"},
    {"question_id": 11, "question": "What is Tesla's stock price forecast for 2025?"},
    {"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
    {"question_id": 13, "question": "What color is Tesla's headquarters painted?"},
]


# Run evaluation
print("=" * 60)
print("RUNNING EVALUATION")
print("=" * 60)

results = []
for q in EVAL_QUESTIONS:
    qid = q["question_id"]
    question = q["question"]
    
    print(f"\nQ{qid}: {question}")
    print("-" * 50)
    
    response = answer_question(question)
    results.append({"question_id": qid, "answer": response["answer"], "sources": response["sources"]})
    
    ans = response["answer"]
    print(f"Answer: {ans[:150]}..." if len(ans) > 150 else f"Answer: {ans}")
    print(f"Sources: {response['sources']}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)


# Save and display results
output_file = os.path.join(DATA_DIR, "evaluation_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to: {output_file}")

print("\nFinal Results:")
print(json.dumps(results, indent=2))
