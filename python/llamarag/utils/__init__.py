"""
Utility functions for LlamaRAG.
"""

from llamarag.utils.retrieval_utils import (
    load_documents_from_json,
    save_documents_to_json,
    generate_document_id,
    ensure_document_ids,
    batch_embed_documents,
    format_retrieval_results,
    cosine_similarity,
    euclidean_distance
)

__all__ = [
    "load_documents_from_json",
    "save_documents_to_json",
    "generate_document_id",
    "ensure_document_ids",
    "batch_embed_documents",
    "format_retrieval_results",
    "cosine_similarity",
    "euclidean_distance"
] 