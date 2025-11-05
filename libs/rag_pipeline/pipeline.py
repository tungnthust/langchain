"""Main RAG pipeline orchestrator."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from .config import PipelineConfig
from .ingestion import DocumentIngestionPipeline, DocumentStore, VectorStore
from .qa_engine import QAEngine
from .query_processor import QueryPreprocessor
from .retrieval import HybridRetriever


class DocumentQAPipeline:
    """Complete RAG pipeline for document Q&A."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the RAG pipeline.

        Args:
            config: Pipeline configuration. If None, uses default config.
        """
        self.config = config if config else PipelineConfig()
        self.document_store: Optional[DocumentStore] = None
        self.vector_store: Optional[VectorStore] = None
        self.retriever: Optional[HybridRetriever] = None
        self.query_preprocessor: Optional[QueryPreprocessor] = None
        self.qa_engine: Optional[QAEngine] = None

    def ingest_documents(self, file_paths: Optional[List[str]] = None) -> None:
        """
        Ingest documents into the pipeline.

        Args:
            file_paths: List of markdown file paths. If None, scans document_storage_dir.
        """
        if file_paths is None:
            # Scan directory for markdown files
            doc_dir = Path(self.config.document_storage_dir)
            if not doc_dir.exists():
                raise ValueError(f"Document directory does not exist: {doc_dir}")

            file_paths = [
                str(f) for f in doc_dir.glob("*.md") if f.is_file()
            ]

        if not file_paths:
            raise ValueError("No markdown files found to ingest")

        print(f"Found {len(file_paths)} markdown files to ingest")

        # Create ingestion pipeline
        ingestion_pipeline = DocumentIngestionPipeline(
            document_store_path=self.config.document_store_path,
            vector_store_path=self.config.vector_store_path,
            embedding_model_name=self.config.model_config.embedding_model_name,
            device=self.config.model_config.embedding_device,
            parent_chunk_size=self.config.model_config.parent_chunk_size,
            child_chunk_size=self.config.model_config.child_chunk_size,
        )

        # Ingest documents
        ingestion_pipeline.ingest_documents(file_paths)

        print("Document ingestion complete!")

    def load_or_ingest(self, file_paths: Optional[List[str]] = None, force_reingest: bool = False) -> None:
        """
        Load existing stores or ingest documents.

        Args:
            file_paths: List of markdown file paths.
            force_reingest: If True, force re-ingestion even if stores exist.
        """
        doc_store_path = Path(self.config.document_store_path)
        vector_store_path = Path(self.config.vector_store_path)

        stores_exist = (
            (doc_store_path / "document_store.pkl").exists()
            and (vector_store_path / "vector_store.pkl").exists()
        )

        if stores_exist and not force_reingest:
            print("Loading existing document stores...")
            self.load_pipeline()
        else:
            print("Ingesting documents...")
            self.ingest_documents(file_paths)
            self.load_pipeline()

    def load_pipeline(self) -> None:
        """Load all components of the pipeline."""
        print("Loading pipeline components...")

        # Load stores
        self.document_store = DocumentStore(self.config.document_store_path)
        self.document_store.load()

        self.vector_store = VectorStore(
            self.config.vector_store_path,
            embedding_model_name=self.config.model_config.embedding_model_name,
            device=self.config.model_config.embedding_device,
        )
        self.vector_store.load()

        # Initialize retriever
        bm25_index_path = Path(self.config.vector_store_path) / "bm25_index.pkl"
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index_path=str(bm25_index_path),
            bm25_weight=self.config.model_config.bm25_weight,
            vector_weight=self.config.model_config.vector_weight,
            reranker_model_name=self.config.model_config.reranker_model_name,
            device=self.config.model_config.reranker_device,
        )

        # Initialize query preprocessor
        self.query_preprocessor = QueryPreprocessor(
            llm_model_name=self.config.model_config.llm_model_name,
            device=self.config.model_config.llm_device,
            max_length=self.config.model_config.llm_max_length,
            temperature=self.config.model_config.llm_temperature,
        )

        # Initialize QA engine
        self.qa_engine = QAEngine(
            document_store=self.document_store,
            vector_store=self.vector_store,
            retriever=self.retriever,
            query_preprocessor=self.query_preprocessor,
            llm_model_name=self.config.model_config.llm_model_name,
            device=self.config.model_config.llm_device,
            max_length=self.config.model_config.llm_max_length,
            temperature=self.config.model_config.llm_temperature,
        )

        print("Pipeline loaded successfully!")

    def ask(self, question: str) -> Dict:
        """
        Ask a question and get an answer with citations.

        Args:
            question: The question to answer.

        Returns:
            Dict with answer, citations, and metadata.
        """
        if self.qa_engine is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

        return self.qa_engine.answer_question(
            question,
            use_decomposition=self.config.use_query_decomposition,
            use_hyde=self.config.use_hyde,
            hybrid_top_k=self.config.model_config.hybrid_search_top_k,
            rerank_top_k=self.config.model_config.reranker_top_k,
        )
