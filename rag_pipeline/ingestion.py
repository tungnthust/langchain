"""Document ingestion pipeline for storing documents and creating embeddings."""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .document_processor import Chunk, Document, DocumentProcessor


class DocumentStore:
    """Store for parent chunks (full sections) with metadata."""

    def __init__(self, storage_path: str):
        """
        Initialize document store.

        Args:
            storage_path: Path to store the documents.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, Dict] = {}
        self.chunk_index: Dict[str, Chunk] = {}

    def add_document(self, document: Document) -> None:
        """
        Add a document to the store.

        Args:
            document: Document to add.
        """
        doc_data = {
            "document_name": document.document_name,
            "file_path": document.file_path,
            "metadata": document.metadata,
            "parent_chunks": [],
        }

        for chunk in document.parent_chunks:
            chunk_data = {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
            }
            doc_data["parent_chunks"].append(chunk_data)
            self.chunk_index[chunk.chunk_id] = chunk

        self.documents[document.document_name] = doc_data

    def get_chunk_by_id(self, chunk_id: str) -> Chunk:
        """
        Retrieve a chunk by its ID.

        Args:
            chunk_id: ID of the chunk to retrieve.

        Returns:
            The chunk object.
        """
        if chunk_id in self.chunk_index:
            chunk_data = self.chunk_index[chunk_id]
            return chunk_data

        # Reconstruct from documents if not in index
        for doc_name, doc_data in self.documents.items():
            for chunk_data in doc_data["parent_chunks"]:
                if chunk_data["chunk_id"] == chunk_id:
                    return Chunk(
                        content=chunk_data["content"],
                        metadata=chunk_data["metadata"],
                        chunk_id=chunk_data["chunk_id"],
                        parent_id=None,
                    )
        return None

    def save(self) -> None:
        """Save the document store to disk."""
        save_path = self.storage_path / "document_store.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(
                {"documents": self.documents, "chunk_index": self.chunk_index}, f
            )

    def load(self) -> None:
        """Load the document store from disk."""
        load_path = self.storage_path / "document_store.pkl"
        if load_path.exists():
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                # Reconstruct chunk_index with Chunk objects
                self.chunk_index = {}
                for doc_name, doc_data in self.documents.items():
                    for chunk_data in doc_data["parent_chunks"]:
                        self.chunk_index[chunk_data["chunk_id"]] = Chunk(
                            content=chunk_data["content"],
                            metadata=chunk_data["metadata"],
                            chunk_id=chunk_data["chunk_id"],
                            parent_id=None,
                        )


class VectorStore:
    """Vector store for child chunks with embeddings."""

    def __init__(
        self,
        storage_path: str,
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """
        Initialize vector store.

        Args:
            storage_path: Path to store the vectors.
            embedding_model_name: Name of the embedding model.
            device: Device to use for embeddings.
            batch_size: Batch size for embedding generation.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        print(f"Loading embedding model: {embedding_model_name} on {self.device}")
        self.model = SentenceTransformer(embedding_model_name, device=self.device)

        self.embeddings: np.ndarray = None
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the vector store and generate embeddings.

        Args:
            chunks: List of chunks to add.
        """
        self.chunks.extend(chunks)

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")

        chunk_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        if self.embeddings is None:
            self.embeddings = chunk_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, chunk_embeddings])

        # Update index
        start_idx = len(self.chunks) - len(chunks)
        for i, chunk in enumerate(chunks):
            self.chunk_id_to_idx[chunk.chunk_id] = start_idx + i

    def search(
        self, query: str, top_k: int = 50
    ) -> List[tuple[Chunk, float]]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query: Query text.
            top_k: Number of top results to return.

        Returns:
            List of (chunk, score) tuples.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )[0]

        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))

        return results

    def save(self) -> None:
        """Save the vector store to disk."""
        save_path = self.storage_path / "vector_store.pkl"

        # Convert chunks to serializable format
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append(
                {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id,
                }
            )

        data = {
            "embeddings": self.embeddings,
            "chunks": chunks_data,
            "chunk_id_to_idx": self.chunk_id_to_idx,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> None:
        """Load the vector store from disk."""
        load_path = self.storage_path / "vector_store.pkl"
        if load_path.exists():
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                self.embeddings = data["embeddings"]
                self.chunk_id_to_idx = data["chunk_id_to_idx"]

                # Reconstruct chunks
                self.chunks = []
                for chunk_data in data["chunks"]:
                    self.chunks.append(
                        Chunk(
                            content=chunk_data["content"],
                            metadata=chunk_data["metadata"],
                            chunk_id=chunk_data["chunk_id"],
                            parent_id=chunk_data.get("parent_id"),
                        )
                    )


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents and creating stores."""

    def __init__(
        self,
        document_store_path: str,
        vector_store_path: str,
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            document_store_path: Path for document store.
            vector_store_path: Path for vector store.
            embedding_model_name: Name of embedding model.
            device: Device for embeddings.
            parent_chunk_size: Size of parent chunks.
            child_chunk_size: Size of child chunks.
        """
        self.document_processor = DocumentProcessor(
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
        )

        self.document_store = DocumentStore(document_store_path)
        self.vector_store = VectorStore(
            vector_store_path, embedding_model_name, device
        )

    def ingest_documents(self, file_paths: List[str]) -> None:
        """
        Ingest documents into the stores.

        Args:
            file_paths: List of paths to markdown files.
        """
        print(f"Processing {len(file_paths)} documents...")
        documents = self.document_processor.process_documents(file_paths)

        print(f"Storing {len(documents)} documents...")
        all_child_chunks = []

        for doc in tqdm(documents, desc="Ingesting documents"):
            self.document_store.add_document(doc)
            all_child_chunks.extend(doc.child_chunks)

        print(f"Creating embeddings for {len(all_child_chunks)} child chunks...")
        self.vector_store.add_chunks(all_child_chunks)

        print("Saving stores...")
        self.document_store.save()
        self.vector_store.save()

        print("Ingestion complete!")

    def load_stores(self) -> None:
        """Load existing stores from disk."""
        self.document_store.load()
        self.vector_store.load()
        print("Stores loaded successfully!")
