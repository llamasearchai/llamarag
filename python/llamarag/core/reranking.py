"""
Reranking strategies for refining retrieval results.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llamarag.core.retrieval import SearchResult


class Reranker(ABC):
    """Base class for reranking strategies."""

    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank search results based on relevance to the query.

        Args:
            query: The search query
            results: List of initial search results

        Returns:
            Reranked search results
        """
        pass


class CrossEncoderReranker(Reranker):
    """Reranker using cross-encoder models for relevance scoring."""

    def __init__(
        self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize a CrossEncoderReranker.

        Args:
            model_name_or_path: Name or path of the cross-encoder model
        """
        try:
            from sentence_transformers.cross_encoder import CrossEncoder

            self.model = CrossEncoder(model_name_or_path)
        except ImportError:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using cross-encoder relevance scores."""
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [(query, result.document.content) for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Create new results with updated scores
        reranked_results = []
        for i, (score, result) in enumerate(zip(scores, results)):
            reranked_results.append(SearchResult(result.document, float(score), i + 1))

        # Sort by new scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i + 1

        return reranked_results


class LLMReranker(Reranker):
    """Reranker using an LLM to score relevance."""

    def __init__(self, llm_client, prompt_template: str = None, batch_size: int = 10):
        """
        Initialize an LLMReranker.

        Args:
            llm_client: Client for accessing LLM API
            prompt_template: Template for reranking prompt
            batch_size: Number of documents to rerank in one batch
        """
        self.llm_client = llm_client
        self.batch_size = batch_size

        # Default prompt template if none provided
        if prompt_template is None:
            self.prompt_template = (
                "You are an expert at determining if a document is relevant to a query.\n"
                "Query: {query}\n"
                "Document: {document}\n"
                "Rate the relevance of this document to the query on a scale of 0 to 10, "
                "where 0 means completely irrelevant and 10 means perfectly relevant.\n"
                "Output only the numerical score without explanation."
            )
        else:
            self.prompt_template = prompt_template

    def _get_relevance_score(self, query: str, document: str) -> float:
        """Get relevance score from LLM."""
        prompt = self.prompt_template.format(query=query, document=document)
        response = self.llm_client.generate(prompt)

        # Extract numerical score from response
        try:
            score = float(response.strip())
            # Normalize to [0, 1] range
            score = min(max(score / 10.0, 0.0), 1.0)
            return score
        except ValueError:
            # Fall back to 0 if response is not a number
            return 0.0

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using LLM relevance judgments."""
        if not results:
            return []

        # Process in batches to avoid overwhelming the LLM API
        reranked_results = []

        for i in range(0, len(results), self.batch_size):
            batch = results[i : i + self.batch_size]

            # Get scores for batch
            for result in batch:
                score = self._get_relevance_score(query, result.document.content)
                reranked_results.append(SearchResult(result.document, score, 0))

        # Sort by new scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i + 1

        return reranked_results
