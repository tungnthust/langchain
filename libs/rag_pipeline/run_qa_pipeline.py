#!/usr/bin/env python3
"""
Standalone script to run the RAG pipeline with questions from CSV.

This script can be run directly without package installation.
Usage: python run_qa_pipeline.py [questions.csv]
"""

import csv
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from config import create_custom_config
from pipeline import DocumentQAPipeline


def load_questions(csv_path: str) -> list:
    """Load questions from CSV file."""
    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    return questions


def find_correct_answers(question_text: str, choices: dict, answer_data: dict) -> list:
    """Determine correct answer choices based on the QA engine's response."""
    answer = answer_data.get("answer", "").lower()

    # Simple heuristic: check which choices are mentioned in the answer
    correct = []
    for letter, choice in choices.items():
        if choice and choice.lower() in answer:
            correct.append(letter)

    # If no choices found, default to first choice
    if not correct:
        correct = ["A"]

    return sorted(correct)


def format_answer_output(question_num: int, correct_answers: list) -> str:
    """Format the answer output according to requirements."""
    if len(correct_answers) == 1:
        return f"{question_num},{correct_answers[0]}"
    else:
        return f'{question_num},"{",".join(correct_answers)}"'


def main():
    """Main function to process questions."""
    # Parse arguments
    question_csv_path = sys.argv[1] if len(sys.argv) > 1 else "questions.csv"
    output_path = "answers.txt"

    # Configuration
    document_storage_dir = "/home/public/hoangnguyen/TechnicalDocument/Solution/extraction/submission"

    # Check if document directory exists
    if not os.path.exists(document_storage_dir):
        print(f"Warning: Document directory not found: {document_storage_dir}")
        print("Please update document_storage_dir in the script or ensure the directory exists.")
        return

    # Create pipeline configuration
    config = create_custom_config(
        llm_model_name="Qwen/Qwen2.5-3B-Instruct",
        embedding_model_name="BAAI/bge-m3",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        document_storage_dir=document_storage_dir,
        vector_store_path="./vector_store",
        document_store_path="./document_store",
    )

    print("=" * 80)
    print("Document Q&A Pipeline - Question Processor")
    print("=" * 80)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = DocumentQAPipeline(config)

    # Load or ingest documents
    try:
        # Get all markdown files
        md_files = [
            os.path.join(document_storage_dir, f)
            for f in os.listdir(document_storage_dir)
            if f.endswith(".md")
        ]

        if not md_files:
            print("Error: No markdown files found in document directory!")
            return

        print(f"Found {len(md_files)} markdown files")
        pipeline.load_or_ingest(file_paths=md_files, force_reingest=False)
    except Exception as e:
        print(f"Error during pipeline initialization: {e}")
        return

    # Load questions
    if not os.path.exists(question_csv_path):
        print(f"\nError: Question file not found: {question_csv_path}")
        return

    print(f"\nLoading questions from {question_csv_path}...")
    questions = load_questions(question_csv_path)
    print(f"Loaded {len(questions)} questions")

    # Process questions
    print("\nProcessing questions...")
    print("=" * 80)

    answers = []
    for idx, question_data in enumerate(questions, 1):
        question_text = question_data.get("Question", "")
        choices = {
            "A": question_data.get("A", ""),
            "B": question_data.get("B", ""),
            "C": question_data.get("C", ""),
            "D": question_data.get("D", ""),
        }

        print(f"\nQuestion {idx}: {question_text}")
        print("Choices:")
        for letter, choice in choices.items():
            print(f"  {letter}: {choice}")

        # Get answer from pipeline
        try:
            answer_data = pipeline.ask(question_text)
            print(f"\nAnswer: {answer_data['answer']}")

            if answer_data.get("citations"):
                print("\nCitations:")
                for citation in answer_data["citations"][:3]:  # Show top 3 citations
                    print(f"  - {citation['document_name']}: {citation['section_header']}")

            # Determine correct answers
            correct_answers = find_correct_answers(question_text, choices, answer_data)
            answer_line = format_answer_output(idx, correct_answers)
            answers.append(answer_line)

            print(f"\nSelected Answer(s): {', '.join(correct_answers)}")

        except Exception as e:
            print(f"Error processing question {idx}: {e}")
            answers.append(format_answer_output(idx, ["A"]))  # Default answer

        print("-" * 80)

    # Save answers
    print(f"\nSaving answers to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(answers))

    print("\nDone! Answers saved to", output_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
