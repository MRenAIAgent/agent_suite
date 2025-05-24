"""RagStore interface for the RAG system.

This module provides the abstract base class for the storage interface
of the RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

from agent_suite.rag.models.document import Document
from agent_suite.rag.models.knowledge import Entity, Relationship, KnowledgeGraph
from agent_suite.rag.models.context import ContextItem


class RagStore(ABC):
    """Interface for storing data in the RAG system."""
    
    @abstractmethod
    async def store_document(self, document: Document) -> str:
        """
        Store a document in the system.
        
        Args:
            document: The document to store
            
        Returns:
            The ID of the stored document
        """
        pass
    
    @abstractmethod
    async def store_knowledge(self, knowledge: Union[Entity, Relationship, KnowledgeGraph]) -> str:
        """
        Store knowledge (entity, relationship, or graph) in the system.
        
        Args:
            knowledge: The knowledge to store
            
        Returns:
            The ID of the stored knowledge
        """
        pass
    
    @abstractmethod
    async def store_context(self, context: ContextItem) -> str:
        """
        Store contextual information in the system.
        
        Args:
            context: The context item to store
            
        Returns:
            The ID of the stored context item
        """
        pass
    
    @abstractmethod
    async def update(self, id: str, data: Any) -> bool:
        """
        Update an existing item in the store.
        
        Args:
            id: The ID of the item to update
            data: The new data
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete an item from the store.
        
        Args:
            id: The ID of the item to delete
            
        Returns:
            Success status
        """
        pass 