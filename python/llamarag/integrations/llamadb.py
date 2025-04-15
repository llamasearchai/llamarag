"""
Integration with llamadb for vector storage and retrieval.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from llamarag.core.retrieval import Document, Retriever, SearchResult


class LlamaDBVectorStore:
    """Interface to llamadb vector storage."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        collection_name: str = "llamarag_docs",
    ):
        """
        Initialize LlamaDBVectorStore.

        Args:
            connection_string: Connection string for llamadb
            collection_name: Name of the collection to use
        """
        try:
            import llamadb

            if connection_string:
                self.db = llamadb.connect(connection_string)
            else:
                self.db = llamadb.connect()
        except ImportError:
            raise ImportError(
                "LlamaDBVectorStore requires the llamadb package. "
                "Please install llamadb first."
            )

        self.collection_name = collection_name

        # Initialize collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize vector collection if it doesn't exist."""
        collections = self.db.list_collections()

        if self.collection_name not in collections:
            self.db.create_collection(
                name=self.collection_name,
                vector_dimension=1536,  # Default dimension, will be updated on first add
            )

    def add_documents(
        self, documents: List[Document], embedding_model=None
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: Documents to add
            embedding_model: Optional model to generate embeddings if not present

        Returns:
            List of document IDs
        """
        # Generate embeddings if needed
        for doc in documents:
            if doc.embedding is None and embedding_model is not None:
                doc.embedding = embedding_model.embed(doc.content)

        # Get collection
        collection = self.db.get_collection(self.collection_name)

        # Add documents
        ids = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(
                    "Document missing embedding and no embedding model provided"
                )

            doc_id = doc.id or os.urandom(16).hex()
            collection.add(
                id=doc_id,
                vector=doc.embedding.tolist(),
                metadata={"content": doc.content, **doc.metadata},
            )
            ids.append(doc_id)

        return ids

    def similarity_search(
        self, query_vector: np.ndarray, k: int = 3
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return

        Returns:
            List of (id, metadata, score) tuples
        """
        collection = self.db.get_collection(self.collection_name)
        results = collection.query(vector=query_vector.tolist(), limit=k)

        return [(r.id, r.metadata, r.score) for r in results]


class LlamaDBRetriever(Retriever):
    """Retriever that uses llamadb for vector storage and search."""

    def __init__(self, vector_store: LlamaDBVectorStore, embedding_model):
        """
        Initialize LlamaDBRetriever.

        Args:
            vector_store: LlamaDBVectorStore instance
            embedding_model: Model to generate query embeddings
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, k: int = 3) -> List[SearchResult]:
        """Retrieve documents from llamadb."""
        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)

        # Search
        results = self.vector_store.similarity_search(query_embedding, k=k)

        # Convert to SearchResult objects
        search_results = []
        for i, (doc_id, metadata, score) in enumerate(results):
            content = metadata.pop("content")
            doc = Document(content=content, metadata=metadata, id=doc_id)
            search_results.append(SearchResult(doc, score, i + 1))

        return search_results

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the retriever's vector store."""
        return self.vector_store.add_documents(documents, self.embedding_model)
