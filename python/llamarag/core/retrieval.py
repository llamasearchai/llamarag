"""
Retrieval strategies for RAG applications.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class Document:
    """Representation of a document or chunk for retrieval."""
    
    def __init__(self, 
                 content: str, 
                 metadata: Optional[Dict[str, Any]] = None,
                 embedding: Optional[np.ndarray] = None,
                 id: Optional[str] = None):
        """
        Initialize a Document.
        
        Args:
            content: The text content of the document
            metadata: Optional metadata associated with the document
            embedding: Optional pre-computed embedding
            id: Optional document identifier
        """
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding
        self.id = id
        
    def __repr__(self) -> str:
        return f"Document(id={self.id}, content={self.content[:50]}...)"

class SearchResult:
    """Result from a retrieval operation."""
    
    def __init__(self, 
                 document: Document, 
                 score: float,
                 rank: int = 0):
        """
        Initialize a SearchResult.
        
        Args:
            document: The retrieved document
            score: Relevance score
            rank: Position in the result list
        """
        self.document = document
        self.score = score
        self.rank = rank
        
    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, doc={self.document.content[:30]}...)"

class Retriever(ABC):
    """Base class for retrieval strategies."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[SearchResult]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of search results
        """
        pass

class VectorRetriever(Retriever):
    """Vector-based retrieval using embeddings."""
    
    def __init__(self, 
                 documents: List[Document],
                 embedding_model,
                 distance_strategy: str = "cosine"):
        """
        Initialize a VectorRetriever.
        
        Args:
            documents: List of documents to search
            embedding_model: Model to generate embeddings
            distance_strategy: Strategy for computing distance ("cosine", "euclidean", "dot")
        """
        self.documents = documents
        self.embedding_model = embedding_model
        self.distance_strategy = distance_strategy
        
        # Pre-compute embeddings for documents if not already present
        for i, doc in enumerate(self.documents):
            if doc.embedding is None:
                doc.embedding = self.embedding_model.embed(doc.content)
                
    def _compute_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """Compute similarity between query and document embedding."""
        if self.distance_strategy == "cosine":
            # Cosine similarity
            return np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
        elif self.distance_strategy == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(query_embedding - doc_embedding)
            return 1.0 / (1.0 + distance)
        elif self.distance_strategy == "dot":
            # Dot product
            return np.dot(query_embedding, doc_embedding)
        else:
            raise ValueError(f"Unknown distance strategy: {self.distance_strategy}")
        
    def retrieve(self, query: str, k: int = 3) -> List[SearchResult]:
        """Retrieve documents using vector similarity."""
        # Embed the query
        query_embedding = self.embedding_model.embed(query)
        
        # Compute similarities
        similarities = []
        for doc in self.documents:
            score = self._compute_similarity(query_embedding, doc.embedding)
            similarities.append((doc, score))
            
        # Sort by score and convert to SearchResult objects
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, (doc, score) in enumerate(similarities[:k]):
            results.append(SearchResult(doc, score, i+1))
            
        return results

class HybridRetriever(Retriever):
    """Combines multiple retrieval strategies."""
    
    def __init__(self, 
                 retrievers: List[Tuple[Retriever, float]]):
        """
        Initialize a HybridRetriever.
        
        Args:
            retrievers: List of (retriever, weight) tuples
        """
        self.retrievers = retrievers
        
    def retrieve(self, query: str, k: int = 3) -> List[SearchResult]:
        """Retrieve documents using multiple strategies."""
        all_results = {}
        
        # Collect results from all retrievers
        for retriever, weight in self.retrievers:
            results = retriever.retrieve(query, k=k)
            for result in results:
                doc_id = result.document.id
                if doc_id in all_results:
                    all_results[doc_id].score += result.score * weight
                else:
                    # Create a new result with weighted score
                    weighted_result = SearchResult(
                        result.document, 
                        result.score * weight,
                        result.rank
                    )
                    all_results[doc_id] = weighted_result
                    
        # Convert to list and sort
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results[:k]):
            result.rank = i + 1
            
        return combined_results[:k]

class RerankingRetriever(Retriever):
    """Retriever that first fetches candidates then reranks them."""
    
    def __init__(self, 
                 base_retriever: Retriever,
                 reranker,
                 candidate_multiplier: int = 3):
        """
        Initialize a RerankingRetriever.
        
        Args:
            base_retriever: Retriever to get initial candidates
            reranker: Reranker to score candidates
            candidate_multiplier: How many candidates to fetch vs. final results
        """
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier
        
    def retrieve(self, query: str, k: int = 3) -> List[SearchResult]:
        """Retrieve and rerank documents."""
        # Get initial candidates
        candidates = self.base_retriever.retrieve(
            query, 
            k=k * self.candidate_multiplier
        )
        
        # Rerank candidates
        reranked = self.reranker.rerank(query, candidates)
        
        # Return top k
        return reranked[:k] 