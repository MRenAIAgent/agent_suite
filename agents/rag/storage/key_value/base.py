"""Key-value storage adaptor interface for the RAG system.

This module provides the abstract base class for key-value storage adaptors in the RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

from agents.rag.storage.base import StorageAdaptor
from agents.rag.models.context import ContextItem


class KeyValueStorageAdaptor(StorageAdaptor, ABC):
    """Base interface for key-value storage adaptors."""
    
    @abstractmethod
    async def set(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> str:
        """
        Set a key-value pair.
        
        Args:
            key: The key
            value: The value
            metadata: Optional metadata
            
        Returns:
            The ID of the stored item
        """
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value by key.
        
        Args:
            key: The key
            
        Returns:
            The value or None if not found
        """
        pass
    
    @abstractmethod
    async def get_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """
        Get all key-value pairs with keys starting with the given prefix.
        
        Args:
            prefix: The key prefix
            
        Returns:
            Dictionary of matching key-value pairs
        """
        pass
    
    @abstractmethod
    async def store_context_item(self, item: ContextItem) -> str:
        """
        Store a context item.
        
        Args:
            item: The context item to store
            
        Returns:
            The ID of the stored item
        """
        pass
    
    @abstractmethod
    async def get_all_context(self) -> List[ContextItem]:
        """
        Get all stored context items.
        
        Returns:
            List of all context items
        """
        pass 