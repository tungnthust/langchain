"""Test script for processing questions from CSV file."""

import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

from rag_pipeline.config import PipelineConfig, create_custom_config
from rag_pipeline.pipeline import DocumentQAPipeline


def load_questions(csv_path: str) -> List[dict]:
    """
    Load questions from CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of question dictionaries.
    """
    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    return questions


def find_correct_answers(question_text: str, choices: dict, answer_data: dict) -> List[str]:
    """
    Determine correct answer choices based on the QA engine's response.

    Args:
        question_text: The question text.
        choices: Dict with choices A, B, C, D.
        answer_data: Answer data from QA engine.

    Returns:
        List of correct answer letters.
    """
    answer = answer_data.get("answer", "").lower()

    # Simple heuristic: check which choices are mentioned in the answer
    correct = []
    for letter, choice in choices.items():
        if choice and choice.lower() in answer:
            correct.append(letter)

    # If no choices found, try to match semantically (basic approach)
    if not correct:
        # Default to first choice if nothing matches
        correct = ["A"]

    return sorted(correct)


def format_answer_output(question_num: int, correct_answers: List[str]) -> str:
    """
    Format the answer output according to requirements.

    Args:
        question_num: Question number (1-indexed).
        correct_answers: List of correct answer letters.

    Returns:
        Formatted answer string.
    """
    if len(correct_answers) == 1:
        return f"{question_num},{correct_answers[0]}"
    else:
        return f'{question_num},"{",".join(correct_answers)}"'


def main():
    """Main function to process questions."""
    # Configuration
    document_storage_dir = "/home/public/hoangnguyen/TechnicalDocument/Solution/extraction/submission"
    question_csv_path = "questions.csv"  # Adjust path as needed
    output_path = "answers.txt"

    # Check if document directory exists
    if not os.path.exists(document_storage_dir):
        print(f"Warning: Document directory not found: {document_storage_dir}")
        print("Using test configuration with local directory...")
        document_storage_dir = "./test_documents"
        os.makedirs(document_storage_dir, exist_ok=True)

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
    print("Document Q&A Pipeline - Test Runner")
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
            print("No markdown files found. Creating sample documents for testing...")
            create_sample_documents(document_storage_dir)
            md_files = [
                os.path.join(document_storage_dir, f)
                for f in os.listdir(document_storage_dir)
                if f.endswith(".md")
            ]

        print(f"Found {len(md_files)} markdown files")
        pipeline.load_or_ingest(file_paths=md_files, force_reingest=False)
    except Exception as e:
        print(f"Error during pipeline initialization: {e}")
        print("Creating sample documents for testing...")
        create_sample_documents(document_storage_dir)
        md_files = [
            os.path.join(document_storage_dir, f)
            for f in os.listdir(document_storage_dir)
            if f.endswith(".md")
        ]
        pipeline.load_or_ingest(file_paths=md_files, force_reingest=True)

    # Load questions
    if not os.path.exists(question_csv_path):
        print(f"\nWarning: Question file not found: {question_csv_path}")
        print("Creating sample questions for testing...")
        create_sample_questions(question_csv_path)

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


def create_sample_documents(directory: str):
    """Create sample documents for testing."""
    os.makedirs(directory, exist_ok=True)

    sample_doc1 = """# Điện Tử Cơ Bản

## Linh Kiện Điện Tử

### Điện Trở
Điện trở là linh kiện có chức năng hạn chế dòng điện chạy qua mạch. Điện trở được đo bằng đơn vị Ohm (Ω).

### Tụ Điện
Tụ điện là linh kiện có khả năng lưu trữ năng lượng dưới dạng điện trường.

### Transistor
Transistor là linh kiện bán dẫn được sử dụng để khuếch đại hoặc chuyển mạch tín hiệu điện tử.

### Diode
Diode là linh kiện bán dẫn cho phép dòng điện chạy theo một chiều.

## Công Suất Điện

Công suất điện được tính bằng công thức P = V × I, trong đó:
- P là công suất (Watt)
- V là điện áp (Volt)
- I là dòng điện (Ampere)

Với nguồn 230 VAC và dòng điện 0.25 A, công suất cực đại là:
P = 230 × 0.25 = 57.5 W
"""

    sample_doc2 = """# Hướng Dẫn PowerPoint

## Các Thành Phần Slide

### Chèn Văn Bản
Bạn có thể chèn văn bản vào slide bằng cách sử dụng Text Box.

### Chèn Hình Ảnh
PowerPoint cho phép chèn hình ảnh từ file hoặc từ internet.

### Chèn Video và Âm Thanh
Bạn có thể chèn video và âm thanh để làm phong phú bài trình bày.

## Tùy Chỉnh Giao Diện

### Thay Đổi Màu Nền
Có thể thay đổi màu nền của slide hoặc áp dụng template có sẵn.

### Tạo Hiệu Ứng
PowerPoint cung cấp nhiều hiệu ứng cho chữ, hình ảnh và chuyển slide.

## Tóm Tắt
Có thể chèn văn bản, hình ảnh, video, âm thanh; thay đổi màu nền, Template; tạo hiệu ứng chữ, hình, chuyển Slide.
"""

    with open(os.path.join(directory, "electronics.md"), "w", encoding="utf-8") as f:
        f.write(sample_doc1)

    with open(os.path.join(directory, "powerpoint.md"), "w", encoding="utf-8") as f:
        f.write(sample_doc2)

    print(f"Created sample documents in {directory}")


def create_sample_questions(csv_path: str):
    """Create sample questions CSV file."""
    questions = [
        {
            "Question": "Linh kiện nào có chức năng hạn chế dòng điện chạy qua mạch?",
            "A": "Điện trở",
            "B": "Tụ điện",
            "C": "Transistor",
            "D": "Diode",
        },
        {
            "Question": "Tóm tắt các thành phần có thể chèn vào Slide và cách tùy chỉnh giao diện/hiệu ứng trình chiếu.",
            "A": "Chỉ chèn được văn bản",
            "B": "Chỉ chèn được hình ảnh và video",
            "C": "Có thể chèn văn bản, hình ảnh, âm thanh, video nhưng không đổi nền",
            "D": "Có thể chèn văn bản, hình, video, âm thanh; thay đổi màu nền, Template; tạo hiệu ứng chữ, hình, chuyển Slide",
        },
        {
            "Question": "Nếu dòng điện cực đại của nguồn 230 VAC là 0.25 A, công suất tiêu thụ cực đại của RCE khoảng bao nhiêu?",
            "A": "25 W",
            "B": "57,5 W",
            "C": "110 W",
            "D": "250 W",
        },
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["Question", "A", "B", "C", "D"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for question in questions:
            writer.writerow(question)

    print(f"Created sample questions in {csv_path}")


if __name__ == "__main__":
    main()
