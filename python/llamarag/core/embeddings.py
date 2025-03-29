"""
Embedding models for converting text to vector representations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np

class EmbeddingModel(ABC):
    """Base class for embedding models."""
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to vector representation.
        
        Args:
            text: The text to embed
            
        Returns:
            Vector representation of the text
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Convert multiple texts to vector representations.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector representations
        """
        pass

class AnthropicEmbeddings(EmbeddingModel):
    """Embeddings from Anthropic's Claude API."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "claude-3-haiku-20240307"):
        """
        Initialize AnthropicEmbeddings.
        
        Args:
            api_key: Anthropic API key
            model: Model name to use for embeddings
        """
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "AnthropicEmbeddings requires the anthropic package. "
                "Install with: pip install anthropic"
            )
        self.model = model
        
    def embed(self, text: str) -> np.ndarray:
        """Get embeddings for a single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.embeddings[0].embedding)
        
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [np.array(data.embedding) for data in response.embeddings]

class OpenAIEmbeddings(EmbeddingModel):
    """Embeddings from OpenAI's API."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "text-embedding-ada-002"):
        """
        Initialize OpenAIEmbeddings.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use for embeddings
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAIEmbeddings requires the openai package. "
                "Install with: pip install openai"
            )
        self.model = model
        
    def embed(self, text: str) -> np.ndarray:
        """Get embeddings for a single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding)
        
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [np.array(data.embedding) for data in response.data]

class HuggingFaceEmbeddings(EmbeddingModel):
    """Embeddings from HuggingFace models."""
    
    def __init__(self, 
                 model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        """
        Initialize HuggingFaceEmbeddings.
        
        Args:
            model_name_or_path: Model name or path
            device: Device to run model on ("cpu", "cuda", etc.)
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name_or_path, device=device)
        except ImportError:
            raise ImportError(
                "HuggingFaceEmbeddings requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
        
    def embed(self, text: str) -> np.ndarray:
        """Get embeddings for a single text."""
        return self.model.encode(text, convert_to_numpy=True)
        
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embedding for embedding in embeddings] 