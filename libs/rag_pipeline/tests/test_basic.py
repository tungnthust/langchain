"""Basic unit tests for RAG pipeline components."""

import os
from pathlib import Path

import pytest


class TestDocumentProcessor:
    """Test document processor functionality."""

    def test_chunk_creation(self):
        """Test that chunks can be created."""
        from langchain_rag_pipeline.document_processor import Chunk

        chunk = Chunk(
            content="Test content",
            metadata={"doc": "test"},
            chunk_id="test_1",
            parent_id=None,
        )

        assert chunk.content == "Test content"
        assert chunk.chunk_id == "test_1"
        assert chunk.metadata["doc"] == "test"


class TestConfig:
    """Test configuration functionality."""

    def test_default_config(self):
        """Test default configuration."""
        from langchain_rag_pipeline.config import get_default_config

        config = get_default_config()
        assert config is not None
        assert config.model_config.llm_model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert config.model_config.embedding_model_name == "BAAI/bge-m3"

    def test_custom_config(self):
        """Test custom configuration."""
        from langchain_rag_pipeline.config import create_custom_config

        config = create_custom_config(
            llm_model_name="custom/model",
            embedding_model_name="custom/embeddings",
        )

        assert config.model_config.llm_model_name == "custom/model"
        assert config.model_config.embedding_model_name == "custom/embeddings"


class TestMarkdownParser:
    """Test markdown parser functionality."""

    def test_parse_simple_markdown(self):
        """Test parsing simple markdown."""
        from langchain_rag_pipeline.document_processor import MarkdownParser

        parser = MarkdownParser()
        content = """# Header 1

Some content here.

## Header 2

More content."""

        result = parser.parse_markdown(content, "test.md")

        assert "sections" in result
        assert len(result["sections"]) > 0
        assert result["file_path"] == "test.md"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
