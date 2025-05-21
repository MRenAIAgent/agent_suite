"""
Unified memory manager for agent_suite.

This module provides a MemoryManager class that brings together
context (short-term) and long-term memory capabilities.
"""

from typing import Dict, List, Any, Optional, Union

from agent_suite.memory.context.base import ContextMemory
from agent_suite.memory.long_term.base import LongTermMemory


class MemoryManager:
    """
    Unified memory manager that handles context and long-term memory.
    
    This class provides a unified interface for memory operations,
    supporting both context (short-term) and long-term memory,
    as well as user-specific and global memory.
    """
    
    def __init__(
        self,
        context_memory: ContextMemory,
        long_term_memory: LongTermMemory,
        user_id: Optional[str] = None
    ):
        """
        Initialize the memory manager with specific memory implementations.
        
        Args:
            context_memory: Implementation of context memory
            long_term_memory: Implementation of long-term memory
            user_id: Optional user ID for user-specific memory
        """
        self.context_memory = context_memory
        self.long_term_memory = long_term_memory
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
    
    # Long-term memory methods
    
    def store(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store data in long-term memory.
        
        Args:
            data: The data to store
            metadata: Optional metadata
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        # Add user_id to metadata if available
        if self.user_id:
            if metadata is None:
                metadata = {}
            metadata["user_id"] = self.user_id
        
        return self.long_term_memory.store(data, metadata)
    
    def query(self, query: Union[str, Dict], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the long-term memory.
        
        Args:
            query: Natural language or structured query
            limit: Maximum number of results
            
        Returns:
            List of matching memories with metadata
        """
        # Add user_id filter for structured queries
        if isinstance(query, dict) and self.user_id:
            query = query.copy()
            query["user_id"] = self.user_id
        
        return self.long_term_memory.query(query, limit)
    
    def remember(self, query: str, context_key: str = "retrieved_memories", limit: int = 3) -> None:
        """
        Query long-term memory and add results to context.
        
        This is a convenience method that performs a query against long-term
        memory and automatically adds the results to context memory.
        
        Args:
            query: The query string
            context_key: Key to use for storing in context
            limit: Maximum number of results to retrieve
        """
        results = self.query(query, limit=limit)
        
        # Add each result to context
        for result in results:
            self.add_to_context(
                context_key,
                result.get("text", result),
                {"source": "long_term_memory", "score": result.get("score")}
            )
    
    def memorize(self, data: Any, context_key: Optional[str] = None) -> str:
        """
        Store context memory item in long-term memory.
        
        This is a convenience method that takes either a specific context
        memory item or the latest item for a key and stores it in long-term memory.
        
        Args:
            data: Data to memorize, or None to use context
            context_key: Optional key to retrieve from context
            
        Returns:
            memory_id: ID of the stored memory
        """
        if context_key:
            # Get the latest item for this key from context
            items = self.get_from_context(context_key)
            if items and isinstance(items, list) and items:
                data = items[-1]["value"]
        
        # Store in long-term memory
        return self.store(data)
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory from long-term storage.
        
        Args:
            memory_id: The unique identifier of the memory
            
        Returns:
            True if deleted successfully, False otherwise
        """
        return self.long_term_memory.delete(memory_id)
    
    def clear_long_term(self) -> None:
        """Clear all long-term memories."""
        self.long_term_memory.clear()
    
    def clear_all(self) -> None:
        """Clear both context and long-term memory."""
        self.clear_context()
        self.clear_long_term() 