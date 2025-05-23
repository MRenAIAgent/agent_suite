"""Unified memory manager that combines context and longterm memory.

This module provides a unified memory manager that integrates
both short-term context memory and long-term persistent memory.
"""

from typing import List, Dict, Any, Optional
from agents.memory.base_memory_manager import MemoryManagerBase
from agents.memory.context_memory import ContextMemory
from agents.memory.longterm_memory import LongtermMemory
from agents.memory.stores.store import Store


class MemoryManager(MemoryManagerBase):
    """Unified memory manager that integrates context and longterm memory.
    
    This manager provides a simplified interface for working with both
    context (short-term) memory and longterm (persistent) memory, 
    handling synchronization between them.
    """
    
    def __init__(
        self,
        max_history: int = 20,
        use_longterm: bool = False,
        storage_dir: str = "./memory_storage",
        session_id: Optional[str] = None,
        context_store: Optional[Store] = None
    ):
        """Initialize the unified memory manager.
        
        Args:
            max_history: Maximum number of history items to keep in context
            use_longterm: Whether to use longterm storage
            storage_dir: Directory for longterm storage
            session_id: Optional session ID for organizing memories
            context_store: Optional custom store for context memory (e.g., RedisStore)
        """
        super().__init__(max_history)
        
        # Initialize memory components
        self.context_memory = ContextMemory(max_history=max_history, store=context_store)
        self.use_longterm = use_longterm
        
        if use_longterm:
            self.longterm_memory = LongtermMemory(
                max_history=max_history * 5,  # Keep more in longterm
                storage_dir=storage_dir,
                session_id=session_id
            )
    
    def save_memory(self, message: Dict) -> None:
        """Save a message to both context and longterm memory.
        
        Args:
            message: Message to save
        """
        self.context_memory.save_memory(message)
        
        if self.use_longterm:
            self.longterm_memory.save_memory(message)
        
    def load_memory(self) -> List[Dict]:
        """Load memory from storage.
        
        If longterm memory is enabled, tries to load from there first.
        Otherwise, loads from context memory.
        
        Returns:
            List of message items
        """
        if self.use_longterm:
            # Load from longterm storage
            memories = self.longterm_memory.load_memory()
            
            # Sync to context memory (just most recent ones)
            recent = memories[-self.max_history:] if len(memories) > self.max_history else memories
            self.context_memory.store.clear(history=True)
            for msg in recent:
                self.context_memory.store.add_history(msg)
                
            return memories
        else:
            return self.context_memory.load_memory()
    
    def add(self, message: Dict) -> None:
        """Add a message to both context and longterm memory.
        
        Args:
            message: Message to add
        """
        self.context_memory.add(message)
        
        if self.use_longterm:
            self.longterm_memory.add(message)
            
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get current conversation history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of messages
        """
        return self.context_memory.get_history(limit=limit)
    
    def clear(self) -> None:
        """Clear conversation history from context memory.
        
        Note: This only clears context memory, not longterm storage.
        """
        self.context_memory.clear()
    
    def purge(self) -> bool:
        """Permanently delete longterm storage.
        
        Returns:
            True if successful, False if not using longterm or failed
        """
        if self.use_longterm:
            return self.longterm_memory.purge()
        return False
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory for relevant messages.
        
        Searches context memory first, then longterm if available.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching messages
        """
        # Search context memory first
        results = self.context_memory.search(query, limit)
        
        # If we need more results and longterm is available, search there too
        if self.use_longterm and len(results) < limit:
            remaining = limit - len(results)
            longterm_results = self.longterm_memory.search(query, remaining)
            
            # Merge results, avoiding duplicates
            existing_ids = set(msg.get('id', '') for msg in results)
            for msg in longterm_results:
                if msg.get('id', '') not in existing_ids:
                    results.append(msg)
        
        return results
    
    # Pass-through methods for context memory features
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a value in the working context."""
        self.context_memory.set_context(key, value)
    
    def get_context(self, key: Optional[str] = None) -> Any:
        """Get value(s) from the working context."""
        return self.context_memory.get_context(key)
    
    def clear_context(self, key: Optional[str] = None) -> None:
        """Clear working context."""
        self.context_memory.clear_context(key)
    
    def get_formatted_history(self, format_type: str = "default") -> str:
        """Get formatted conversation history."""
        return self.context_memory.get_formatted_history(format_type) 