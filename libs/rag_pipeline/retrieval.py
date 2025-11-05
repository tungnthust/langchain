"""Hybrid retrieval combining BM25 and vector search with ColBERT re-ranking."""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sentence_transformers import CrossEncoder

from .bm25 import BM25
from .document_processor import Chunk
from .ingestion import VectorStore


class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index_path: str,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Vector store for semantic search.
            bm25_index_path: Path to BM25 index.
            bm25_weight: Weight for BM25 scores.
            vector_weight: Weight for vector scores.
            reranker_model_name: Name of the re-ranker model.
            device: Device to use for re-ranking.
        """
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Initialize BM25
        self.bm25 = BM25()
        bm25_path = Path(bm25_index_path)
        if bm25_path.exists():
            print("Loading BM25 index...")
            self.bm25.load(str(bm25_path))
        else:
            print("Creating BM25 index...")
            self.bm25.fit(vector_store.chunks)
            self.bm25.save(str(bm25_path))
            print("BM25 index created and saved.")

        # Initialize re-ranker
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading re-ranker model: {reranker_model_name} on {self.device}")
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)

    def hybrid_search(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[Chunk, float]]:
        """
        Perform hybrid search combining BM25 and vector search.

        Args:
            query: Query string.
            top_k: Number of top results to return.

        Returns:
            List of (chunk, score) tuples.
        """
        # Get results from both retrievers
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        vector_results = self.vector_store.search(query, top_k=top_k * 2)

        # Normalize scores
        bm25_scores = self._normalize_scores([score for _, score in bm25_results])
        vector_scores = self._normalize_scores([score for _, score in vector_results])

        # Create score dictionaries
        bm25_dict = {chunk.chunk_id: score for (chunk, _), score in zip(bm25_results, bm25_scores)}
        vector_dict = {chunk.chunk_id: score for (chunk, _), score in zip(vector_results, vector_scores)}

        # Combine results
        combined_chunks: Dict[str, Tuple[Chunk, float]] = {}

        for chunk, _ in bm25_results:
            chunk_id = chunk.chunk_id
            bm25_score = bm25_dict.get(chunk_id, 0.0)
            vector_score = vector_dict.get(chunk_id, 0.0)
            combined_score = (
                self.bm25_weight * bm25_score + self.vector_weight * vector_score
            )
            combined_chunks[chunk_id] = (chunk, combined_score)

        for chunk, _ in vector_results:
            chunk_id = chunk.chunk_id
            if chunk_id not in combined_chunks:
                bm25_score = bm25_dict.get(chunk_id, 0.0)
                vector_score = vector_dict.get(chunk_id, 0.0)
                combined_score = (
                    self.bm25_weight * bm25_score + self.vector_weight * vector_score
                )
                combined_chunks[chunk_id] = (chunk, combined_score)

        # Sort by combined score
        results = sorted(combined_chunks.values(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def rerank(
        self, query: str, chunks: List[Tuple[Chunk, float]], top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Re-rank chunks using ColBERT-style cross-encoder.

        Args:
            query: Query string.
            chunks: List of (chunk, score) tuples to re-rank.
            top_k: Number of top results to return after re-ranking.

        Returns:
            List of re-ranked (chunk, score) tuples.
        """
        if not chunks:
            return []

        # Prepare pairs for re-ranking
        pairs = [[query, chunk.content] for chunk, _ in chunks]

        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

        # Combine with original chunks
        reranked = [(chunk, float(score)) for (chunk, _), score in zip(chunks, rerank_scores)]

        # Sort by re-ranking score
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]

    def retrieve(
        self, query: str, hybrid_top_k: int = 50, rerank_top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Complete retrieval pipeline: hybrid search + re-ranking.

        Args:
            query: Query string.
            hybrid_top_k: Number of candidates from hybrid search.
            rerank_top_k: Number of final results after re-ranking.

        Returns:
            List of (chunk, score) tuples.
        """
        # Step 1: Hybrid search
        hybrid_results = self.hybrid_search(query, top_k=hybrid_top_k)

        # Step 2: Re-rank
        reranked_results = self.rerank(query, hybrid_results, top_k=rerank_top_k)

        return reranked_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: List of scores to normalize.

        Returns:
            List of normalized scores.
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(score - min_score) / (max_score - min_score) for score in scores]
