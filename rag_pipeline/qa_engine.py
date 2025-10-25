"""QA Engine with structured output and citations."""

import json
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .document_processor import Chunk
from .ingestion import DocumentStore, VectorStore
from .query_processor import QueryPreprocessor
from .retrieval import HybridRetriever


class QAEngine:
    """Question Answering engine with structured output."""

    def __init__(
        self,
        document_store: DocumentStore,
        vector_store: VectorStore,
        retriever: HybridRetriever,
        query_preprocessor: QueryPreprocessor,
        llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 0.1,
    ):
        """
        Initialize QA engine.

        Args:
            document_store: Store for parent chunks.
            vector_store: Store for child chunks with embeddings.
            retriever: Hybrid retriever with re-ranking.
            query_preprocessor: Query preprocessor.
            llm_model_name: Name of the LLM model.
            device: Device to use for inference.
            max_length: Maximum length for generation.
            temperature: Temperature for generation.
        """
        self.document_store = document_store
        self.vector_store = vector_store
        self.retriever = retriever
        self.query_preprocessor = query_preprocessor

        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.temperature = temperature

        # Use the same model from query preprocessor if available
        if query_preprocessor and query_preprocessor.model:
            print("Reusing LLM model from query preprocessor")
            self.tokenizer = query_preprocessor.tokenizer
            self.model = query_preprocessor.model
        else:
            print(f"Loading LLM model: {llm_model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
            )

    def answer_question(
        self,
        question: str,
        use_decomposition: bool = True,
        use_hyde: bool = True,
        hybrid_top_k: int = 50,
        rerank_top_k: int = 5,
    ) -> Dict:
        """
        Answer a question with structured output and citations.

        Args:
            question: The question to answer.
            use_decomposition: Whether to use query decomposition.
            use_hyde: Whether to use HyDE.
            hybrid_top_k: Number of candidates from hybrid search.
            rerank_top_k: Number of final results after re-ranking.

        Returns:
            Dict with answer, citations, and metadata.
        """
        # Step 1: Preprocess query
        preprocessed = self.query_preprocessor.preprocess(
            question, use_decomposition=use_decomposition, use_hyde=use_hyde
        )

        # Step 2: Retrieve for each sub-query
        all_chunks = []
        chunk_scores = {}

        for sub_query in preprocessed["sub_queries"]:
            retrieved = self.retriever.retrieve(
                sub_query, hybrid_top_k=hybrid_top_k, rerank_top_k=rerank_top_k
            )
            for chunk, score in retrieved:
                if chunk.chunk_id not in chunk_scores:
                    all_chunks.append(chunk)
                    chunk_scores[chunk.chunk_id] = score
                else:
                    # Keep the highest score
                    chunk_scores[chunk.chunk_id] = max(chunk_scores[chunk.chunk_id], score)

        # Also retrieve using HyDE if available
        if preprocessed["hyde"]:
            hyde_retrieved = self.retriever.retrieve(
                preprocessed["hyde"], hybrid_top_k=hybrid_top_k, rerank_top_k=rerank_top_k
            )
            for chunk, score in hyde_retrieved:
                if chunk.chunk_id not in chunk_scores:
                    all_chunks.append(chunk)
                    chunk_scores[chunk.chunk_id] = score
                else:
                    chunk_scores[chunk.chunk_id] = max(chunk_scores[chunk.chunk_id], score)

        # Sort by score and take top chunks
        all_chunks.sort(key=lambda c: chunk_scores[c.chunk_id], reverse=True)
        top_chunks = all_chunks[:rerank_top_k]

        # Step 3: Fetch parent chunks (Small-to-Big)
        parent_chunks = []
        for chunk in top_chunks:
            if chunk.parent_id:
                parent_chunk = self.document_store.get_chunk_by_id(chunk.parent_id)
                if parent_chunk:
                    parent_chunks.append(parent_chunk)

        # Step 4: Generate answer with citations
        answer_data = self._generate_answer(question, parent_chunks)

        return answer_data

    def _generate_answer(
        self, question: str, context_chunks: List[Chunk]
    ) -> Dict:
        """
        Generate answer with structured output.

        Args:
            question: The question to answer.
            context_chunks: List of context chunks.

        Returns:
            Dict with answer and citations.
        """
        # Build context
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            doc_name = chunk.metadata.get("document_name", "Unknown")
            headers = []
            for key in sorted(chunk.metadata.keys()):
                if key.startswith("header_level_"):
                    headers.append(chunk.metadata[key])
            section_header = " > ".join(headers) if headers else "No section"

            context_parts.append(
                f"[Document {i+1}: {doc_name} - {section_header}]\n{chunk.content}\n"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant that answers questions based on the provided context. You must provide accurate answers with citations.

Context:
{context}

Question: {question}

Provide your answer in the following JSON format:
{{
    "answer": "Your detailed answer here",
    "citations": [
        {{
            "document_name": "name of the document",
            "section_header": "section header",
            "snippet": "relevant snippet from the text"
        }}
    ]
}}

Response:"""

        try:
            response = self._generate(prompt)
            # Extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                answer_data = json.loads(json_str)

                # Validate structure
                if "answer" not in answer_data:
                    answer_data["answer"] = response

                if "citations" not in answer_data or not isinstance(answer_data["citations"], list):
                    # Create citations from context chunks
                    answer_data["citations"] = []
                    for chunk in context_chunks:
                        doc_name = chunk.metadata.get("document_name", "Unknown")
                        headers = []
                        for key in sorted(chunk.metadata.keys()):
                            if key.startswith("header_level_"):
                                headers.append(chunk.metadata[key])
                        section_header = " > ".join(headers) if headers else "No section"

                        answer_data["citations"].append({
                            "document_name": doc_name,
                            "section_header": section_header,
                            "snippet": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        })

                return answer_data
        except Exception as e:
            print(f"Error in answer generation: {e}")

        # Fallback response
        return {
            "answer": "Unable to generate answer. Please try again.",
            "citations": [],
        }

    def _generate(self, prompt: str) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text.
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the generated part (after the prompt)
        if prompt in response:
            response = response[len(prompt):].strip()

        return response
