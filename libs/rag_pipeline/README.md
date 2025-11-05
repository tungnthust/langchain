# Advanced RAG Pipeline for Document Q&A

This is a comprehensive Retrieval-Augmented Generation (RAG) pipeline for accurate question answering over a collection of markdown documents.

## Features

### Advanced RAG Techniques

1. **Hierarchical Chunking**
   - Parent Chunks: Larger sections with full context (~2000 tokens)
   - Child Chunks: Smaller paragraphs for precise retrieval (~500 tokens)
   - Automatic header metadata extraction

2. **Query Preprocessing**
   - Query Decomposition: Break complex questions into simpler sub-queries
   - HyDE (Hypothetical Document Embeddings): Generate hypothetical answers for better retrieval

3. **Hybrid Search**
   - BM25: Lexical/keyword-based search
   - BGE-M3: Multilingual semantic embeddings
   - Weighted combination for optimal results

4. **ColBERT Re-ranking**
   - Cross-encoder re-ranking for top-k refinement
   - Improves precision by re-scoring candidates

5. **Small-to-Big Context**
   - Retrieve with child chunks (fast, precise)
   - Fetch parent chunks for generation (full context)

6. **Structured Output**
   - JSON format with answer and citations
   - Document name, section headers, and snippets

## Architecture

```
Question → Query Preprocessing → Hybrid Search → Re-ranking → Context Fetching → Answer Generation
           (Decomposition, HyDE)  (BM25 + BGE)   (ColBERT)   (Small-to-Big)   (Structured Output)
```

## Installation

### Option 1: Install from source

```bash
# Navigate to the rag_pipeline directory
cd libs/rag_pipeline

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Option 2: Install dependencies only

```bash
cd libs/rag_pipeline
pip install -r requirements.txt

# For GPU support (CUDA 11.8 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Running the Question Processor

The easiest way to use the pipeline is with the standalone script:

```bash
cd libs/rag_pipeline

# Make sure your markdown files are in the correct directory
# Update document_storage_dir in run_qa_pipeline.py if needed

# Run with questions.csv in the current directory
python run_qa_pipeline.py

# Or specify a different CSV file
python run_qa_pipeline.py path/to/your/questions.csv
```

### Using as a Python Package

```python
import sys
sys.path.insert(0, 'libs/rag_pipeline')

from config import create_custom_config
from pipeline import DocumentQAPipeline

# Create configuration
config = create_custom_config(
    llm_model_name="Qwen/Qwen2.5-3B-Instruct",
    embedding_model_name="BAAI/bge-m3",
    reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    document_storage_dir="/path/to/your/markdown/files",
)

# Initialize pipeline
pipeline = DocumentQAPipeline(config)

# Load or ingest documents
pipeline.load_or_ingest()

# Ask questions
result = pipeline.ask("What is the function of a resistor?")

print(result["answer"])
for citation in result["citations"]:
    print(f"- {citation['document_name']}: {citation['section_header']}")
```

## Question CSV Format

The input CSV file should have the following format:

```csv
Question,A,B,C,D
"What component limits current in a circuit?","Resistor","Capacitor","Transistor","Diode"
"What can you insert in PowerPoint?","Only text","Only images","Text and images","Text, images, audio, video, backgrounds, effects"
```

## Output Format

The pipeline generates an `answers.txt` file with the following format:

```
1,A
2,D
3,B
```

For questions with multiple correct answers:

```
1,"A,B"
```

## Usage

### Basic Usage

```python
from langchain_rag_pipeline import DocumentQAPipeline, create_custom_config

# Create configuration
config = create_custom_config(
    llm_model_name="Qwen/Qwen2.5-3B-Instruct",
    embedding_model_name="BAAI/bge-m3",
    reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    document_storage_dir="/path/to/your/markdown/files",
)

# Initialize pipeline
pipeline = DocumentQAPipeline(config)

# Load or ingest documents
pipeline.load_or_ingest()

# Ask questions
result = pipeline.ask("What is the function of a resistor?")

print(result["answer"])
for citation in result["citations"]:
    print(f"- {citation['document_name']}: {citation['section_header']}")
```

### Custom Model Configuration

```python
from langchain_rag_pipeline import create_custom_config

# Use different models
config = create_custom_config(
    llm_model_name="meta-llama/Llama-2-7b-chat-hf",  # Any HuggingFace model
    embedding_model_name="BAAI/bge-large-en-v1.5",
    reranker_model_name="BAAI/bge-reranker-large",
    document_storage_dir="/path/to/documents",
    vector_store_path="./my_vector_store",
    document_store_path="./my_document_store",
)
```

## Configuration Options

### Model Configuration

- `llm_model_name`: LLM for query preprocessing and answer generation (default: "Qwen/Qwen2.5-3B-Instruct")
- `embedding_model_name`: Embedding model for vector search (default: "BAAI/bge-m3")
- `reranker_model_name`: Re-ranker model (default: "cross-encoder/ms-marco-MiniLM-L-6-v2")
- `llm_device`: Device for LLM ("cuda" or "cpu")
- `embedding_device`: Device for embeddings
- `reranker_device`: Device for re-ranker

### Retrieval Configuration

- `hybrid_search_top_k`: Number of candidates from hybrid search (default: 50)
- `reranker_top_k`: Final number of results after re-ranking (default: 5)
- `bm25_weight`: Weight for BM25 in hybrid search (default: 0.5)
- `vector_weight`: Weight for vector search (default: 0.5)

### Chunking Configuration

- `parent_chunk_size`: Size of parent chunks (default: 2000)
- `parent_chunk_overlap`: Overlap for parent chunks (default: 200)
- `child_chunk_size`: Size of child chunks (default: 500)
- `child_chunk_overlap`: Overlap for child chunks (default: 50)

## Components

### DocumentProcessor
Parses markdown files and creates hierarchical chunks with header metadata.

### DocumentIngestionPipeline
Ingests documents and creates document store and vector store.

### HybridRetriever
Combines BM25 and vector search, with ColBERT re-ranking.

### QueryPreprocessor
Preprocesses queries with decomposition and HyDE.

### QAEngine
Generates answers with structured output and citations.

### DocumentQAPipeline
Main orchestrator that ties all components together.

## File Structure

```
libs/rag_pipeline/
├── __init__.py
├── config.py                 # Configuration management
├── document_processor.py     # Markdown parsing and chunking
├── ingestion.py              # Document ingestion and stores
├── bm25.py                   # BM25 implementation
├── retrieval.py              # Hybrid retrieval and re-ranking
├── query_processor.py        # Query preprocessing
├── qa_engine.py              # Answer generation
├── pipeline.py               # Main pipeline orchestrator
├── run_qa_pipeline.py        # Standalone script for CSV questions
├── run_questions.py          # Module version of question processor
├── example.py                # Example usage
├── tests/                    # Unit tests
├── requirements.txt          # Dependencies
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Models

All models are open-source from HuggingFace:

- **LLM**: Qwen/Qwen2.5-3B-Instruct (recommended, easily swappable)
- **Embeddings**: BAAI/bge-m3 (multilingual support)
- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-6-v2 (efficient and accurate)

You can easily switch to other models:
- LLM: Any instruction-tuned model (Llama, Mistral, Phi, etc.)
- Embeddings: bge-large, e5-large, multilingual-e5, etc.
- Re-ranker: bge-reranker-large, colbert-v2, etc.

## Performance Tips

1. **GPU Usage**: Enable CUDA for significant speedup
2. **Batch Processing**: Process multiple questions in batches
3. **Caching**: Embeddings and indices are cached on disk
4. **Model Size**: Use smaller models for faster inference, larger for better accuracy

## Troubleshooting

### Out of Memory
- Reduce batch size in configuration
- Use smaller models
- Reduce chunk sizes

### Slow Performance
- Enable GPU if available
- Reduce `hybrid_search_top_k`
- Use smaller models

### Poor Accuracy
- Increase `hybrid_search_top_k` and `reranker_top_k`
- Adjust chunk sizes
- Try different embedding models

## Example Workflow

1. **Prepare your documents**: Place all markdown files in a directory
2. **Create questions.csv**: Format your questions as shown above
3. **Update configuration**: Edit `run_qa_pipeline.py` to point to your document directory
4. **Run the pipeline**: `python run_qa_pipeline.py`
5. **Check results**: Review `answers.txt` for the answers

## License

MIT License - See repository LICENSE file for details.
