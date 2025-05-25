"""RagRetrieval interface for the RAG system.

This module provides the abstract base class for the retrieval interface
of the RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, TypeVar

from agents.rag.models.document import Document
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.context import ContextItem

# Generic type for retrieval results
T = TypeVar('T', Document, Entity, Relationship, ContextItem)


class RetrievalResult:
    """Container for retrieval results with relevance scoring."""
    
    def __init__(self, item: T, score: float = 1.0, source: str = None):
        self.item = item
        self.score = score
        self.source = source
    
    def __repr__(self):
        return f"RetrievalResult(item={self.item}, score={self.score}, source={self.source})"


class RagRetrieval(ABC):
    """Interface for retrieving data from the RAG system."""
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5, 
                     filter_criteria: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant information based on a query.
        
        Args:
            query: The query string
            top_k: Maximum number of results to return
            filter_criteria: Optional filters to apply
            
        Returns:
            List of retrieval results with relevance scores
        """
        pass
    
    @abstractmethod
    async def retrieve_by_id(self, id: str) -> Optional[Any]:
        """
        Retrieve a specific item by ID.
        
        Args:
            id: The ID of the item to retrieve
            
        Returns:
            The retrieved item or None if not found
        """
        pass
    
    @abstractmethod
    async def retrieve_documents(self, query: str, top_k: int = 5,
                               filter_criteria: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Retrieve documents based on a query.
        
        Args:
            query: The query string
            top_k: Maximum number of results to return
            filter_criteria: Optional filters to apply
            
        Returns:
            List of document retrieval results with relevance scores
        """
        pass
    
    @abstractmethod
    async def retrieve_knowledge(self, query: str, top_k: int = 5,
                               filter_criteria: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Retrieve knowledge based on a query.
        
        Args:
            query: The query string
            top_k: Maximum number of results to return
            filter_criteria: Optional filters to apply
            
        Returns:
            List of knowledge retrieval results with relevance scores
        """
        pass
    
    @abstractmethod
    async def retrieve_context(self, key: Optional[str] = None) -> List[ContextItem]:
        """
        Retrieve context items, optionally filtered by key.
        
        Args:
            key: Optional key to filter by
            
        Returns:
            List of context items
        """
        pass 