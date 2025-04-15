"""
llamarag - A Python package for Retrieval-Augmented Generation with LLMs

llamarag provides tools and utilities for building retrieval-augmented generation
systems that combine the power of large language models with information retrieval.
"""

__version__ = "0.1.0"

from llamarag.core.chunking import (
    Chunker,
    FixedSizeChunker,
    OverlappingChunker,
    ParagraphChunker,
    SemanticChunker,
    SentenceChunker,
)
from llamarag.core.embeddings import (
    AnthropicEmbeddings,
    EmbeddingModel,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)
from llamarag.core.reranking import (
    CrossEncoderReranker,
    LLMReranker,
    Reranker,
)
from llamarag.core.retrieval import (
    HybridRetriever,
    RerankingRetriever,
    Retriever,
    VectorRetriever,
)
