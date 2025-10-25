"""Simple example of using the RAG pipeline."""

import os

from rag_pipeline import DocumentQAPipeline, create_custom_config


def main():
    """Run a simple example."""
    # Configuration
    config = create_custom_config(
        llm_model_name="Qwen/Qwen2.5-3B-Instruct",
        embedding_model_name="BAAI/bge-m3",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        document_storage_dir="./test_documents",
        vector_store_path="./vector_store",
        document_store_path="./document_store",
    )

    print("Initializing RAG Pipeline...")
    pipeline = DocumentQAPipeline(config)

    # Create test documents if they don't exist
    if not os.path.exists("./test_documents"):
        print("Creating test documents...")
        os.makedirs("./test_documents", exist_ok=True)
        create_test_documents()

    # Load or ingest documents
    print("Loading documents...")
    pipeline.load_or_ingest()

    # Example questions
    questions = [
        "What is a resistor?",
        "How do you insert images in PowerPoint?",
        "Calculate the power consumption with 230V and 0.25A",
    ]

    print("\n" + "=" * 80)
    print("Asking Questions")
    print("=" * 80)

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 80)

        result = pipeline.ask(question)

        print(f"Answer: {result['answer']}\n")

        if result.get("citations"):
            print("Citations:")
            for citation in result["citations"][:3]:
                print(f"  - {citation['document_name']}: {citation['section_header']}")
                print(f"    Snippet: {citation['snippet'][:100]}...")

        print()


def create_test_documents():
    """Create sample test documents."""
    doc1 = """# Electronics Basics

## Electronic Components

### Resistors
A resistor is an electronic component that limits or regulates the flow of electrical current in an electronic circuit. Resistors are measured in Ohms (Ω).

### Capacitors
Capacitors store electrical energy in an electric field. They are used in various applications including filtering and energy storage.

### Transistors
Transistors are semiconductor devices used to amplify or switch electronic signals and electrical power.

## Electrical Power

Electrical power is calculated using the formula P = V × I, where:
- P is power (Watts)
- V is voltage (Volts)
- I is current (Amperes)

With a 230 VAC source and 0.25 A current, the maximum power is:
P = 230 × 0.25 = 57.5 W
"""

    doc2 = """# PowerPoint Guide

## Slide Components

### Inserting Text
You can insert text into slides using Text Boxes. Simply click on the Text Box tool and start typing.

### Inserting Images
PowerPoint allows you to insert images from files or from the internet. Go to Insert > Pictures to add images.

### Inserting Videos and Audio
You can insert videos and audio files to enrich your presentations. Use Insert > Video or Insert > Audio.

## Customizing Appearance

### Changing Background
You can change the background color of slides or apply pre-made templates.

### Creating Effects
PowerPoint provides many effects for text, images, and slide transitions to make presentations more engaging.

## Summary
You can insert text, images, videos, and audio; change backgrounds and templates; create effects for text, images, and slide transitions.
"""

    with open("./test_documents/electronics.md", "w", encoding="utf-8") as f:
        f.write(doc1)

    with open("./test_documents/powerpoint.md", "w", encoding="utf-8") as f:
        f.write(doc2)

    print("Test documents created successfully!")


if __name__ == "__main__":
    main()
