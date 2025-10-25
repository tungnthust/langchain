"""
Advanced RAG Pipeline for Document Q&A.

This module provides a comprehensive RAG (Retrieval-Augmented Generation) pipeline
with advanced techniques including hierarchical chunking, hybrid search, query
preprocessing, and ColBERT re-ranking.
"""

from .config import ModelConfig, PipelineConfig, create_custom_config, get_default_config
from .document_processor import DocumentProcessor
from .ingestion import DocumentIngestionPipeline
from .pipeline import DocumentQAPipeline
from .qa_engine import QAEngine
from .query_processor import QueryPreprocessor
from .retrieval import HybridRetriever

__all__ = [
    "DocumentProcessor",
    "DocumentIngestionPipeline",
    "HybridRetriever",
    "QueryPreprocessor",
    "QAEngine",
    "DocumentQAPipeline",
    "ModelConfig",
    "PipelineConfig",
    "create_custom_config",
    "get_default_config",
]
