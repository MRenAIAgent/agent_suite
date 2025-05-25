"""Vector storage adaptor interface for the RAG system.

This module provides the abstract base class for vector storage adaptors in the RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

from agents.rag.storage.base import StorageAdaptor
from agents.rag.models.document import Document, DocumentChunk


class VectorStorageAdaptor(StorageAdaptor, ABC):
    """Base interface for vector storage adaptors."""
    
    @abstractmethod
    async def embed_and_store(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Embed text and store it in the vector database.
        
        Args:
            text: The text to embed and store
            metadata: Optional metadata
            
        Returns:
            The ID of the stored vector
        """
        pass
    
    @abstractmethod
    async def store_document(self, document: Document, chunk_size: int = 0) -> str:
        """
        Store a document, potentially chunking it.
        
        Args:
            document: The document to store
            chunk_size: Optional chunk size (0 means no chunking)
            
        Returns:
            The ID of the stored document
        """
        pass
    
    @abstractmethod
    async def semantic_search(self, query: str, top_k: int = 5, 
                            filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search.
        
        Args:
            query: The query text
            top_k: Maximum number of results
            filter_criteria: Optional filters
            
        Returns:
            List of matches with scores
        """
        pass
    
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        pass 