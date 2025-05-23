"""Base interface for storage service.

This module defines the interface that all storage implementations
must adhere to, providing a consistent API for different backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar

T = TypeVar('T')


class BaseStorage(Generic[T], ABC):
    """Base interface for storage operations.
    
    This abstract class defines the contract that all storage implementations must follow,
    providing a consistent API regardless of the underlying storage technology.
    """
    
    @abstractmethod
    async def store(self, key: str, data: T, **kwargs) -> str:
        """Store data with the given key.
        
        Args:
            key: Unique identifier for the data
            data: The data to store
            **kwargs: Additional storage-specific parameters
            
        Returns:
            Identifier for the stored data
        """
        pass
    
    @abstractmethod
    async def retrieve(self, key: str, **kwargs) -> Optional[T]:
        """Retrieve data by key.
        
        Args:
            key: Unique identifier for the data
            **kwargs: Additional retrieval-specific parameters
            
        Returns:
            The stored data or None if not found
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete data by key.
        
        Args:
            key: Unique identifier for the data to delete
            **kwargs: Additional deletion-specific parameters
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def update(self, key: str, data: T, **kwargs) -> bool:
        """Update data associated with the key.
        
        Args:
            key: Unique identifier for the data to update
            data: New data to store
            **kwargs: Additional update-specific parameters
            
        Returns:
            True if update was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def list(self, pattern: str = "*", **kwargs) -> List[str]:
        """List keys matching a pattern.
        
        Args:
            pattern: Pattern to match keys against
            **kwargs: Additional parameters for the listing operation
            
        Returns:
            List of matching keys
        """
        pass
    
    @abstractmethod
    async def clear(self, **kwargs) -> bool:
        """Clear all data from the storage.
        
        Args:
            **kwargs: Additional parameters for the clear operation
            
        Returns:
            True if clear was successful, False otherwise
        """
        pass 