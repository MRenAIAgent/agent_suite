"""Base memory manager for agent memory management.

This module provides a base class for memory managers that extends
the basic memory functionality with more specialized features.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class MemoryManagerBase(ABC):
    """Base class for memory managers with common functionality.
    
    Provides a standard interface for memory operations that can be
    implemented by various types of memory managers.
    """
    
    def __init__(self, max_history: int = 20):
        """Initialize the memory manager.
        
        Args:
            max_history: Maximum number of history items to keep
        """
        self.max_history = max_history
    
    @abstractmethod
    def save_memory(self, message: Dict) -> None:
        """Save a message to memory.
        
        Args:
            message: Message to save
        """
        pass
        
    @abstractmethod
    def load_memory(self) -> List[Dict]:
        """Load memory from storage.
        
        Returns:
            List of message items
        """
        pass
    
    @abstractmethod
    def add(self, message: Dict) -> None:
        """Add a message to history.
        
        Args:
            message: Message to add
        """
        pass
            
    @abstractmethod
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get current conversation history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of messages
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear conversation history."""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory for relevant messages.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching messages
        """
        pass 