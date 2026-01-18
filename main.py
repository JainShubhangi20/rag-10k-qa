#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py
CLI entrypoint to build the RAG index and ask questions.

Defaults:
- PDFs are expected under ./content 
- Index artifacts are stored under ./artifacts

Usage:
  # Build (or rebuild) index
  python main.py --build

  # Ask a single question
  python main.py --question "What was Apple's revenue in 2024?"

  # Batch questions from a JSON file and write answers to output JSON
  python main.py --batch questions.json -o output.json

Input format for batch:
  questions.json can be either:
    1) A JSON list of strings:
       ["Q1", "Q2", "Q3"]

    2) A JSON list of objects with optional ids:
       [{"id": "q1", "question": "..."}, {"id": "q2", "question": "..."}]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from rag_simple_converted_cli import answer_question


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG QA over 10-K PDFs (Apple + Tesla)")
    parser.add_argument("--question", "-q", type=str, required=True, help="Question to ask")
    args = parser.parse_args()

    result = answer_question(args.question)
    print("\nAnswer:\n", result["answer"])
    if result.get("sources"):
        print("\nSources:")
        for s in result["sources"]:
            print("-", s)

if __name__ == "__main__":
    main()
