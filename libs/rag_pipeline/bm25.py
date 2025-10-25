"""BM25 retrieval implementation for hybrid search."""

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .document_processor import Chunk


class BM25:
    """BM25 retrieval algorithm implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with parameters.

        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.chunks: List[Chunk] = []
        self.doc_freqs: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        # Simple tokenization - split on whitespace and punctuation
        return text.lower().split()

    def fit(self, chunks: List[Chunk]) -> None:
        """
        Fit BM25 on a corpus of chunks.

        Args:
            chunks: List of chunks to index.
        """
        self.chunks = chunks
        self.corpus = [self._tokenize(chunk.content) for chunk in chunks]

        # Calculate document frequencies
        self.doc_freqs = Counter()
        for doc in self.corpus:
            self.doc_freqs.update(set(doc))

        # Calculate IDF values
        num_docs = len(self.corpus)
        self.idf = {}
        for word, freq in self.doc_freqs.items():
            self.idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        # Calculate document lengths
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float]]:
        """
        Search for relevant chunks using BM25.

        Args:
            query: Query string.
            top_k: Number of top results to return.

        Returns:
            List of (chunk, score) tuples.
        """
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_len[i]
            doc_freqs = Counter(doc)

            for term in query_tokens:
                if term not in self.idf:
                    continue

                tf = doc_freqs[term]
                idf = self.idf[term]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avgdl)
                )
                score += idf * (numerator / denominator)

            scores.append((self.chunks[i], score))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str) -> None:
        """Save BM25 index to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

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
            "k1": self.k1,
            "b": self.b,
            "corpus": self.corpus,
            "chunks": chunks_data,
            "doc_freqs": dict(self.doc_freqs),
            "idf": self.idf,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load BM25 index from disk."""
        load_path = Path(path)
        if load_path.exists():
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                self.k1 = data["k1"]
                self.b = data["b"]
                self.corpus = data["corpus"]
                self.doc_freqs = Counter(data["doc_freqs"])
                self.idf = data["idf"]
                self.doc_len = data["doc_len"]
                self.avgdl = data["avgdl"]

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
