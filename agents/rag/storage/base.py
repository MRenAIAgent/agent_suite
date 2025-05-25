"""Base storage adaptor interface for the RAG system.

This module provides the abstract base class for all storage adaptors in the RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class StorageAdaptor(ABC):
    """Base interface for all storage adaptors."""
    
    @abstractmethod
    async def store(self, data: Any, metadata: Dict[str, Any] = None) -> str:
        """
        Store data in the backend.
        
        Args:
            data: The data to store
            metadata: Optional metadata
            
        Returns:
            The ID of the stored data
        """
        pass
    
    @abstractmethod
    async def retrieve(self, id: str) -> Optional[Any]:
        """
        Retrieve data by ID.
        
        Args:
            id: The ID of the data to retrieve
            
        Returns:
            The retrieved data or None if not found
        """
        pass
    
    @abstractmethod
    async def update(self, id: str, data: Any) -> bool:
        """
        Update existing data.
        
        Args:
            id: The ID of the data to update
            data: The new data
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete data by ID.
        
        Args:
            id: The ID of the data to delete
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def search(self, query: Any, **kwargs) -> List[Any]:
        """
        Search for data matching the query.
        
        Args:
            query: The search query
            **kwargs: Additional search parameters
            
        Returns:
            List of matching data items
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass 