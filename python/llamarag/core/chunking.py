"""
Document chunking strategies for RAG applications.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize


class Chunker(ABC):
    """Base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split the text into chunks according to the chunking strategy."""
        pass


class SentenceChunker(Chunker):
    """Chunk documents by sentences."""

    def __init__(self, max_sentences_per_chunk: int = 5, language: str = "english"):
        """
        Initialize a SentenceChunker.

        Args:
            max_sentences_per_chunk: Maximum number of sentences per chunk
            language: Language for sentence tokenization
        """
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.language = language

    def chunk(self, text: str) -> List[str]:
        """Split the text into chunks by sentences."""
        sentences = sent_tokenize(text, language=self.language)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class ParagraphChunker(Chunker):
    """Chunk documents by paragraphs."""

    def __init__(self, max_paragraphs_per_chunk: int = 1):
        """
        Initialize a ParagraphChunker.

        Args:
            max_paragraphs_per_chunk: Maximum number of paragraphs per chunk
        """
        self.max_paragraphs_per_chunk = max_paragraphs_per_chunk

    def chunk(self, text: str) -> List[str]:
        """Split the text into chunks by paragraphs."""
        # Split by double newlines or more
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []

        for paragraph in paragraphs:
            current_chunk.append(paragraph)
            if len(current_chunk) >= self.max_paragraphs_per_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


class FixedSizeChunker(Chunker):
    """Chunk documents by fixed size with optional overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize a FixedSizeChunker.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        """Split the text into chunks of fixed size with overlap."""
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # If we're not at the beginning, try to find a space to break at
            if start > 0 and end < len(text):
                # Look for a space to break at
                space_pos = text.rfind(" ", start, end)
                if space_pos != -1:
                    end = space_pos

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap if end < len(text) else len(text)

        return [c for c in chunks if c]  # Filter out any empty chunks


class OverlappingChunker(Chunker):
    """Chunk documents with controlled overlap, ensuring clean breaks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 200,
        split_tokens: List[str] = ["\n", ".", " "],
    ):
        """
        Initialize an OverlappingChunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap_size: Target overlap size in characters
            split_tokens: Tokens to consider for clean breaks, in preference order
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.split_tokens = split_tokens

    def _find_split_position(self, text: str, target_pos: int) -> int:
        """Find the best position to split text, favoring natural breaks."""
        # Check each split token in order of preference
        for token in self.split_tokens:
            # Look for the token before and after the target position
            before = text.rfind(token, 0, target_pos)
            after = text.find(token, target_pos)

            # If found within a reasonable distance, use it
            if before != -1 and target_pos - before <= 100:
                return before + len(token)
            if after != -1 and after - target_pos <= 100:
                return after + len(token)

        # Fall back to the exact position if no good breaks found
        return target_pos

    def chunk(self, text: str) -> List[str]:
        """Split the text into overlapping chunks with clean breaks."""
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            # Calculate the end position
            end_pos = min(start + self.chunk_size, len(text))

            # Find a good split position
            if end_pos < len(text):
                end_pos = self._find_split_position(text, end_pos)

            # Add the chunk
            chunks.append(text[start:end_pos].strip())

            # Calculate the next start position with overlap
            if end_pos >= len(text):
                break

            start = end_pos - self.overlap_size
            # Find a good starting position
            if start > 0:
                start = self._find_split_position(text, start)

        return [c for c in chunks if c]


class SemanticChunker(Chunker):
    """Chunk documents based on semantic meaning using embeddings."""

    def __init__(self, embedding_model, similarity_threshold: float = 0.7):
        """
        Initialize a SemanticChunker.

        Args:
            embedding_model: Model to generate embeddings
            similarity_threshold: Threshold for semantic similarity
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.sentence_chunker = SentenceChunker(max_sentences_per_chunk=1)

    def _calculate_similarity(self, embedding1, embedding2) -> float:
        """Calculate cosine similarity between two embeddings."""
        import numpy as np

        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def chunk(self, text: str) -> List[str]:
        """Split the text into semantically coherent chunks."""
        # First split into sentences
        sentences = self.sentence_chunker.chunk(text)
        if not sentences:
            return []

        # Generate embeddings for each sentence
        embeddings = [self.embedding_model.embed(s) for s in sentences]

        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            similarity = self._calculate_similarity(current_embedding, embeddings[i])

            if similarity >= self.similarity_threshold:
                # Similar enough, add to current chunk
                current_chunk.append(sentences[i])
                # Update embedding as average
                import numpy as np

                current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
            else:
                # Start a new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
