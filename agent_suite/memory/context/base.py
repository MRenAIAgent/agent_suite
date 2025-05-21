"""
Base class for context memory implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class ContextMemory(ABC):
    """
    Base class for context memory implementations.
    
    This defines the interface for storing, retrieving, and formatting
    context memory during agent execution.
    """
    
    @abstractmethod
    def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new memory item.
        
        Args:
            key: The key/type of the memory item
            value: The value to store
            metadata: Optional metadata about this item (e.g. timestamps, user ID)
        """
        pass
    
    @abstractmethod
    def get(self, key: Optional[str] = None) -> Any:
        """
        Retrieve memory items with optional filtering.
        
        Args:
            key: Optional key to filter items (if None, return all items)
            
        Returns:
            Memory items matching the key, or all items if key is None
        """
        pass
    
    @abstractmethod
    def get_formatted(self, format_type: str = "default") -> str:
        """
        Get formatted memory for inclusion in LLM prompts.
        
        Args:
            format_type: Type of formatting to use (e.g. 'default', 'compact', 'detailed')
            
        Returns:
            Formatted string representation of context memory
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory items."""
        pass 