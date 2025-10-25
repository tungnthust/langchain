"""Configuration module for RAG pipeline models and settings."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for models used in the RAG pipeline."""

    # LLM Configuration
    llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_device: str = "cuda"
    llm_max_length: int = 2048
    llm_temperature: float = 0.1

    # Embedding Model Configuration
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # Re-ranker Configuration
    reranker_model_name: str = "colbert-ir/colbertv2.0"
    reranker_device: str = "cuda"
    reranker_top_k: int = 5

    # Retrieval Configuration
    hybrid_search_top_k: int = 50
    bm25_weight: float = 0.5
    vector_weight: float = 0.5

    # Chunking Configuration
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 500
    child_chunk_overlap: int = 50


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline."""

    model_config: ModelConfig = field(default_factory=ModelConfig)
    document_storage_dir: str = "/home/public/hoangnguyen/TechnicalDocument/Solution/extraction/submission"
    vector_store_path: str = "./vector_store"
    document_store_path: str = "./document_store"
    use_query_decomposition: bool = True
    use_hyde: bool = True
    max_retries: int = 3
    cache_embeddings: bool = True


def get_default_config() -> PipelineConfig:
    """Get the default pipeline configuration."""
    return PipelineConfig()


def create_custom_config(
    llm_model_name: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    reranker_model_name: Optional[str] = None,
    **kwargs
) -> PipelineConfig:
    """
    Create a custom pipeline configuration.

    Args:
        llm_model_name: Custom LLM model name from HuggingFace.
        embedding_model_name: Custom embedding model name from HuggingFace.
        reranker_model_name: Custom re-ranker model name from HuggingFace.
        **kwargs: Additional configuration parameters.

    Returns:
        PipelineConfig with custom settings.
    """
    config = PipelineConfig()

    if llm_model_name:
        config.model_config.llm_model_name = llm_model_name
    if embedding_model_name:
        config.model_config.embedding_model_name = embedding_model_name
    if reranker_model_name:
        config.model_config.reranker_model_name = reranker_model_name

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model_config, key):
            setattr(config.model_config, key, value)

    return config
