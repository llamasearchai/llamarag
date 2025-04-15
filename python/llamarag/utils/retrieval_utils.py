"""
Utility functions for working with RAG retrieval processes.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from llamarag.core.embeddings import EmbeddingModel
from llamarag.core.retrieval import Document, SearchResult


def load_documents_from_json(filepath: str) -> List[Document]:
    """
    Load documents from a JSON file.

    Args:
        filepath: Path to the JSON file containing documents

    Returns:
        List of Document objects

    The JSON file should contain a list of objects with at least a 'content' field.
    Optional fields include 'metadata', 'embedding', and 'id'.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        doc = Document(
            content=item["content"],
            metadata=item.get("metadata", {}),
            embedding=np.array(item.get("embedding")) if "embedding" in item else None,
            id=item.get("id"),
        )
        documents.append(doc)

    return documents


def save_documents_to_json(documents: List[Document], filepath: str) -> None:
    """
    Save documents to a JSON file.

    Args:
        documents: List of Document objects to save
        filepath: Path to save the JSON file
    """
    data = []
    for doc in documents:
        item = {"content": doc.content, "metadata": doc.metadata}

        if doc.id is not None:
            item["id"] = doc.id

        if doc.embedding is not None:
            item["embedding"] = doc.embedding.tolist()

        data.append(item)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_document_id(document: Document) -> str:
    """
    Generate a deterministic ID for a document based on its content and metadata.

    Args:
        document: Document object

    Returns:
        A hash string that can be used as an ID
    """
    content = document.content
    metadata_str = (
        json.dumps(document.metadata, sort_keys=True) if document.metadata else ""
    )

    combined = f"{content}|{metadata_str}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def ensure_document_ids(documents: List[Document]) -> List[Document]:
    """
    Ensure all documents have an ID. Generate IDs for documents that don't have one.

    Args:
        documents: List of Document objects

    Returns:
        List of Document objects with IDs
    """
    for i, doc in enumerate(documents):
        if doc.id is None:
            doc.id = generate_document_id(doc)

    return documents


def batch_embed_documents(
    documents: List[Document],
    embedding_model: EmbeddingModel,
    batch_size: int = 32,
    force_reembed: bool = False,
) -> List[Document]:
    """
    Embed a list of documents in batches.

    Args:
        documents: List of Document objects to embed
        embedding_model: EmbeddingModel instance to use for embedding
        batch_size: Number of documents to embed in each batch
        force_reembed: Whether to reembed documents that already have embeddings

    Returns:
        List of Document objects with embeddings
    """
    docs_to_embed = []
    indices = []

    for i, doc in enumerate(documents):
        if doc.embedding is None or force_reembed:
            docs_to_embed.append(doc)
            indices.append(i)

    if not docs_to_embed:
        return documents

    # Process in batches
    for i in range(0, len(docs_to_embed), batch_size):
        batch = docs_to_embed[i : i + batch_size]
        batch_indices = indices[i : i + batch_size]

        texts = [doc.content for doc in batch]
        embeddings = embedding_model.embed_batch(texts)

        for j, embedding in enumerate(embeddings):
            doc_idx = batch_indices[j]
            documents[doc_idx].embedding = embedding

    return documents


def format_retrieval_results(
    results: List[SearchResult], include_metadata: bool = True
) -> str:
    """
    Format retrieval results into a readable string.

    Args:
        results: List of SearchResult objects
        include_metadata: Whether to include metadata in the output

    Returns:
        Formatted string representation of the results
    """
    output = []

    for i, result in enumerate(results):
        output.append(f"Result {i+1} [Score: {result.score:.4f}]")
        output.append(f"Content: {result.document.content}")

        if include_metadata and result.document.metadata:
            metadata_str = ", ".join(
                f"{k}: {v}" for k, v in result.document.metadata.items()
            )
            output.append(f"Metadata: {metadata_str}")

        output.append("")  # Empty line between results

    return "\n".join(output)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (between -1 and 1)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(vec1 - vec2)
