"""
Base class for long-term memory implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union


class LongTermMemory(ABC):
    """
    Base class for long-term memory implementations.
    
    This defines the interface that all long-term memory implementations
    must adhere to, providing methods for storing, retrieving, and
    querying persistent memory.
    """
    
    @abstractmethod
    def store(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store data in long-term memory.
        
        Args:
            data: The data to store (text, structured data, etc.)
            metadata: Optional metadata about this data (e.g. source, timestamp, user ID)
            
        Returns:
            memory_id: A unique identifier for the stored memory
        """
        pass
    
    @abstractmethod
    def query(self, query: Union[str, Dict], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the long-term memory.
        
        Args:
            query: Either a natural language query or a structured query dict
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories with their metadata and relevance scores
        """
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory item.
        
        Args:
            memory_id: The unique identifier of the memory
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all stored memories.
        """
        pass 