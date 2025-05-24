"""Embedding providers for the RAG system.

This module provides interfaces and implementations for generating embeddings
to be used with vector storage in the RAG system.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

# Try to import popular embedding libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector as a list of floats
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass


class DummyEmbeddingProvider(EmbeddingProvider):
    """Dummy embedding provider for testing that returns random vectors."""
    
    def __init__(self, vector_size: int = 384):
        """
        Initialize the dummy embedding provider.
        
        Args:
            vector_size: Size of the embedding vectors to generate
        """
        self.vector_size = vector_size
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate a random embedding vector.
        
        Args:
            text: The text to embed (ignored)
            
        Returns:
            Random embedding vector
        """
        # Return a random unit vector for testing
        vector = np.random.rand(self.vector_size)
        # Normalize to unit length for cosine similarity
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate random embedding vectors for a batch of texts.
        
        Args:
            texts: List of texts to embed (ignored)
            
        Returns:
            List of random embedding vectors
        """
        return [await self.embed(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using the OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with 'pip install openai'")
        
        self.model = model
        openai.api_key = api_key
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding using OpenAI API.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        try:
            # Truncate text if too long
            if len(text) > 8191:
                text = text[:8191]
            
            # Get embedding from OpenAI
            response = await openai.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding from OpenAI: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Truncate texts if too long
            truncated_texts = [text[:8191] if len(text) > 8191 else text for text in texts]
            
            # Get embeddings from OpenAI
            response = await openai.embeddings.create(
                model=self.model,
                input=truncated_texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error getting batch embeddings from OpenAI: {e}")
            raise


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using Sentence Transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Sentence Transformer embedding provider.
        
        Args:
            model_name: Name of the Sentence Transformers model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence Transformers not available. Install with 'pip install sentence-transformers'")
        
        self.model_name = model_name
        self.model = sentence_transformers.SentenceTransformer(model_name)
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding using Sentence Transformers.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        try:
            # Use Sentence Transformers to generate embedding
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error getting embedding from Sentence Transformers: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Sentence Transformers.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Use Sentence Transformers to generate embeddings
            embeddings = self.model.encode(texts)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error getting batch embeddings from Sentence Transformers: {e}")
            raise


def create_embedding_provider(config: Dict[str, Any] = None) -> EmbeddingProvider:
    """
    Create an embedding provider based on configuration.
    
    Args:
        config: Configuration dictionary
            - type: Type of provider ('openai', 'sentence-transformer', 'dummy')
            - api_key: API key (for services like OpenAI)
            - model: Model name
            - vector_size: Vector size (for dummy provider)
            
    Returns:
        An embedding provider instance
    """
    config = config or {}
    provider_type = config.get("type", "dummy").lower()
    
    if provider_type == "openai":
        api_key = config.get("api_key")
        model = config.get("model", "text-embedding-ada-002")
        if not api_key:
            raise ValueError("API key is required for OpenAI embedding provider")
        return OpenAIEmbeddingProvider(api_key=api_key, model=model)
    
    elif provider_type == "sentence-transformer":
        model = config.get("model", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbeddingProvider(model_name=model)
    
    elif provider_type == "dummy":
        vector_size = config.get("vector_size", 384)
        return DummyEmbeddingProvider(vector_size=vector_size)
    
    else:
        raise ValueError(f"Unknown embedding provider type: {provider_type}") 