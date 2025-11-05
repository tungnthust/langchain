# RAG Pipeline Usage Guide

This guide provides step-by-step instructions for using the Advanced RAG Pipeline for document Q&A.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Configuration](#detailed-configuration)
5. [Running with CSV Questions](#running-with-csv-questions)
6. [API Usage](#api-usage)
7. [Model Configuration](#model-configuration)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- Python 3.10 or higher
- 16GB+ RAM recommended
- GPU with CUDA support (optional but highly recommended for performance)
- 10GB+ free disk space for models and indices

### Required Software

- Python 3.10+
- pip or uv package manager
- Git (for cloning the repository)

## Installation

### Step 1: Install Dependencies

```bash
cd libs/rag_pipeline

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Install PyTorch with CUDA Support (Optional but Recommended)

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### Example 1: Simple Question Answering

```python
cd libs/rag_pipeline

# Run the example script
python example.py
```

This will:
1. Create sample documents
2. Ingest them into the pipeline
3. Ask example questions
4. Display answers with citations

### Example 2: Processing CSV Questions

```bash
# Prepare your questions.csv file
# Format: Question,A,B,C,D

# Update the document path in run_qa_pipeline.py
# Change document_storage_dir to point to your markdown files

# Run the pipeline
python run_qa_pipeline.py questions.csv
```

## Detailed Configuration

### Document Directory Setup

Your markdown documents should be organized in a directory:

```
/path/to/documents/
├── document1.md
├── document2.md
├── document3.md
└── ...
```

### Configuration Options

Create a custom configuration:

```python
from config import create_custom_config

config = create_custom_config(
    # LLM Configuration
    llm_model_name="Qwen/Qwen2.5-3B-Instruct",
    llm_device="cuda",  # or "cpu"
    llm_max_length=2048,
    llm_temperature=0.1,
    
    # Embedding Configuration
    embedding_model_name="BAAI/bge-m3",
    embedding_device="cuda",
    embedding_batch_size=32,
    
    # Re-ranker Configuration
    reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_device="cuda",
    reranker_top_k=5,
    
    # Search Configuration
    hybrid_search_top_k=50,
    bm25_weight=0.5,
    vector_weight=0.5,
    
    # Chunking Configuration
    parent_chunk_size=2000,
    parent_chunk_overlap=200,
    child_chunk_size=500,
    child_chunk_overlap=50,
    
    # Storage Paths
    document_storage_dir="/path/to/your/documents",
    vector_store_path="./vector_store",
    document_store_path="./document_store",
    
    # Pipeline Options
    use_query_decomposition=True,
    use_hyde=True,
)
```

## Running with CSV Questions

### Step 1: Prepare Questions CSV

Create a `questions.csv` file:

```csv
Question,A,B,C,D
"What is the function of a resistor?","Limits current","Stores charge","Amplifies signal","Switches current"
"How to insert images in PowerPoint?","File > Insert","Edit > Image","View > Pictures","Insert > Pictures"
```

### Step 2: Update Script Configuration

Edit `run_qa_pipeline.py`:

```python
# Update this line with your document directory
document_storage_dir = "/path/to/your/markdown/files"
```

### Step 3: Run the Pipeline

```bash
python run_qa_pipeline.py questions.csv
```

### Step 4: Review Results

The pipeline will generate `answers.txt`:

```
1,A
2,D
3,B
```

## API Usage

### Basic Usage

```python
import sys
sys.path.insert(0, 'libs/rag_pipeline')

from config import create_custom_config
from pipeline import DocumentQAPipeline

# Create configuration
config = create_custom_config(
    document_storage_dir="/path/to/documents",
)

# Initialize pipeline
pipeline = DocumentQAPipeline(config)

# Load or ingest documents
pipeline.load_or_ingest()

# Ask a question
result = pipeline.ask("What is the function of a resistor?")

# Access results
print("Answer:", result["answer"])
print("\nCitations:")
for citation in result["citations"]:
    print(f"- Document: {citation['document_name']}")
    print(f"  Section: {citation['section_header']}")
    print(f"  Snippet: {citation['snippet']}")
```

### Advanced Usage

```python
# Force re-ingestion of documents
pipeline.load_or_ingest(force_reingest=True)

# Customize retrieval parameters
result = pipeline.ask(
    question="Your question here",
    use_decomposition=True,  # Enable query decomposition
    use_hyde=True,           # Enable HyDE
    hybrid_top_k=100,        # More candidates
    rerank_top_k=10,         # More final results
)
```

## Model Configuration

### Recommended Models by Use Case

#### For Vietnamese/Multilingual Documents

```python
config = create_custom_config(
    llm_model_name="Qwen/Qwen2.5-3B-Instruct",
    embedding_model_name="BAAI/bge-m3",  # Multilingual
    reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
```

#### For English Only (Better Performance)

```python
config = create_custom_config(
    llm_model_name="microsoft/Phi-3-mini-4k-instruct",
    embedding_model_name="BAAI/bge-large-en-v1.5",
    reranker_model_name="BAAI/bge-reranker-large",
)
```

#### For Maximum Accuracy (Requires More Resources)

```python
config = create_custom_config(
    llm_model_name="meta-llama/Llama-2-7b-chat-hf",
    embedding_model_name="BAAI/bge-large-en-v1.5",
    reranker_model_name="BAAI/bge-reranker-large",
    hybrid_search_top_k=100,
    reranker_top_k=10,
)
```

#### For CPU-Only Environments

```python
config = create_custom_config(
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",  # Smaller model
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster
    reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    llm_device="cpu",
    embedding_device="cpu",
    reranker_device="cpu",
    embedding_batch_size=8,  # Smaller batch
)
```

### Switching Models

Models can be easily swapped without changing the pipeline code:

```python
# Just change the model names in configuration
config.model_config.llm_model_name = "new-model/name"
config.model_config.embedding_model_name = "new-embedding/name"
```

## Troubleshooting

### Issue: Out of Memory

**Solution 1:** Use smaller models
```python
config = create_custom_config(
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    embedding_batch_size=8,
)
```

**Solution 2:** Reduce batch sizes
```python
config = create_custom_config(
    embedding_batch_size=8,
    hybrid_search_top_k=20,
    reranker_top_k=3,
)
```

**Solution 3:** Use CPU for some components
```python
config = create_custom_config(
    llm_device="cuda",      # Keep LLM on GPU
    embedding_device="cpu",  # Move embeddings to CPU
    reranker_device="cpu",   # Move reranker to CPU
)
```

### Issue: Slow Performance

**Solution 1:** Enable GPU
```python
config = create_custom_config(
    llm_device="cuda",
    embedding_device="cuda",
    reranker_device="cuda",
)
```

**Solution 2:** Reduce search space
```python
config = create_custom_config(
    hybrid_search_top_k=20,  # Reduced from 50
    reranker_top_k=3,        # Reduced from 5
)
```

**Solution 3:** Disable query preprocessing
```python
config = create_custom_config(
    use_query_decomposition=False,
    use_hyde=False,
)
```

### Issue: Poor Answer Quality

**Solution 1:** Increase retrieval candidates
```python
config = create_custom_config(
    hybrid_search_top_k=100,  # More candidates
    reranker_top_k=10,        # More final results
)
```

**Solution 2:** Adjust chunk sizes
```python
config = create_custom_config(
    parent_chunk_size=3000,   # Larger context
    child_chunk_size=700,     # Larger search units
)
```

**Solution 3:** Try better models
```python
config = create_custom_config(
    llm_model_name="meta-llama/Llama-2-7b-chat-hf",
    embedding_model_name="BAAI/bge-large-en-v1.5",
    reranker_model_name="BAAI/bge-reranker-large",
)
```

### Issue: Models Not Downloading

**Solution:** Download models manually

```python
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Download LLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModel.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Download embedding model
embedding_model = SentenceTransformer("BAAI/bge-m3")
```

### Issue: Documents Not Found

**Solution:** Check paths and permissions

```bash
# Verify document directory exists
ls -la /path/to/documents/

# Check file extensions
ls /path/to/documents/*.md

# Verify read permissions
chmod -R +r /path/to/documents/
```

## Performance Benchmarks

### Expected Processing Times (on A100 GPU)

- **Document Ingestion:** ~1-2 seconds per document
- **Embedding Generation:** ~0.1-0.5 seconds per chunk
- **Query Processing:** ~2-5 seconds per question (with decomposition + HyDE)
- **Answer Generation:** ~1-3 seconds per answer

### Memory Requirements

- **Minimum:** 8GB RAM (CPU only, small models)
- **Recommended:** 16GB RAM + 8GB VRAM (GPU, standard models)
- **Optimal:** 32GB RAM + 24GB VRAM (GPU, large models)

## Best Practices

1. **Document Preparation:**
   - Use clear, hierarchical headers in markdown
   - Keep sections focused and coherent
   - Include relevant metadata in headers

2. **Configuration:**
   - Start with default settings
   - Tune parameters based on specific needs
   - Monitor memory and performance

3. **Question Formulation:**
   - Be specific in questions
   - Include relevant context
   - Use clear, concise language

4. **Evaluation:**
   - Review citations to verify answers
   - Check multiple questions to assess consistency
   - Adjust configuration based on results

## Additional Resources

- [README.md](README.md) - Overview and features
- [example.py](example.py) - Working example code
- [run_qa_pipeline.py](run_qa_pipeline.py) - CSV question processor
- [config.py](config.py) - Configuration options

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example scripts
3. Examine the configuration options
4. Open an issue on GitHub with detailed information

## License

MIT License - See repository LICENSE file for details.
