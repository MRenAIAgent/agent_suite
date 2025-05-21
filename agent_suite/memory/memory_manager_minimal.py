"""
Simplified memory manager for agent_suite.

This module provides a MemoryManagerMinimal class that works with
only context (short-term) memory, without requiring long-term memory.
"""

from typing import Dict, List, Any, Optional, Union

from agent_suite.memory.context.base import ContextMemory


class MemoryManagerMinimal:
    """
    Simplified memory manager that handles context memory only.
    
    This class provides an interface for context memory operations,
    without requiring long-term memory implementations.
    """
    
    def __init__(
        self,
        context_memory: ContextMemory,
        user_id: Optional[str] = None
    ):
        """
        Initialize the memory manager with context memory implementation.
        
        Args:
            context_memory: Implementation of context memory
            user_id: Optional user ID for user-specific memory
        """
        self.context_memory = context_memory
        self.user_id = user_id
    
    # Context memory methods
    
    def add_to_context(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new item to context memory.
        
        Args:
            key: The key/type of the memory item
            value: The value to store
            metadata: Optional metadata about this item
        """
        # Add user_id to metadata if available
        if self.user_id:
            if metadata is None:
                metadata = {}
            metadata["user_id"] = self.user_id
        
        self.context_memory.add(key, value, metadata)
    
    def get_from_context(self, key: Optional[str] = None) -> Any:
        """
        Retrieve items from context memory.
        
        Args:
            key: Optional key to filter items
            
        Returns:
            Memory items matching the key, or all items if key is None
        """
        return self.context_memory.get(key)
    
    def get_formatted_context(self, format_type: str = "default") -> str:
        """
        Get formatted context memory for prompts.
        
        Args:
            format_type: Type of formatting to use
            
        Returns:
            Formatted string representation of context memory
        """
        return self.context_memory.get_formatted(format_type)
    
    def clear_context(self) -> None:
        """Clear all context memory."""
        self.context_memory.clear() 