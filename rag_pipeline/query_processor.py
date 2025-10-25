"""Query preprocessing with query decomposition and HyDE."""

import json
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QueryPreprocessor:
    """Preprocessor for queries with decomposition and HyDE."""

    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 0.1,
    ):
        """
        Initialize query preprocessor.

        Args:
            llm_model_name: Name of the LLM model.
            device: Device to use for inference.
            max_length: Maximum length for generation.
            temperature: Temperature for generation.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.temperature = temperature

        print(f"Loading LLM model: {llm_model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map=self.device
        )

    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.

        Args:
            query: Original query.

        Returns:
            List of sub-queries.
        """
        prompt = f"""You are a helpful assistant that breaks down complex questions into simpler sub-questions.

Given the following question, break it down into 1-3 simpler sub-questions that would help answer the original question. If the question is already simple, just return the original question.

Question: {query}

Return your response as a JSON list of sub-questions. For example:
["sub-question 1", "sub-question 2", "sub-question 3"]

Sub-questions:"""

        try:
            response = self._generate(prompt)
            # Extract JSON from response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                sub_queries = json.loads(json_str)
                return sub_queries if isinstance(sub_queries, list) else [query]
        except Exception as e:
            print(f"Error in query decomposition: {e}")

        return [query]

    def generate_hyde(self, query: str) -> str:
        """
        Generate a hypothetical document (HyDE) for the query.

        Args:
            query: Original query.

        Returns:
            Hypothetical document text.
        """
        prompt = f"""You are a helpful assistant that generates hypothetical answers to questions.

Given the following question, write a detailed hypothetical answer that would be relevant to the question. The answer should be informative and specific.

Question: {query}

Hypothetical Answer:"""

        try:
            response = self._generate(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error in HyDE generation: {e}")
            return query

    def preprocess(
        self, query: str, use_decomposition: bool = True, use_hyde: bool = True
    ) -> dict:
        """
        Preprocess a query with decomposition and HyDE.

        Args:
            query: Original query.
            use_decomposition: Whether to use query decomposition.
            use_hyde: Whether to use HyDE.

        Returns:
            Dict with preprocessed queries.
        """
        result = {"original": query, "sub_queries": [query], "hyde": None}

        # Query decomposition
        if use_decomposition:
            sub_queries = self.decompose_query(query)
            if sub_queries:
                result["sub_queries"] = sub_queries

        # HyDE generation
        if use_hyde:
            hyde = self.generate_hyde(query)
            result["hyde"] = hyde

        return result

    def _generate(self, prompt: str) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
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
