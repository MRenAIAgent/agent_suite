"""Context memory manager for short-term memory management.

This module provides a context memory manager for handling short-term
conversation history and working memory.
"""

from typing import Dict, List, Any, Optional
from agents.memory.base_memory_manager import MemoryManagerBase
from agents.memory.stores.in_memory_store import InMemoryStore
from agents.memory.stores.store import Store


class ContextMemory(MemoryManagerBase):
    """Memory manager for short-term context and conversation history.
    
    This manager handles working memory for the agent, including the
    immediate conversation context, recent interactions, and temporary
    information needed during the current session.
    """
    
    def __init__(self, max_history: int = 20, store: Optional[Store] = None):
        """Initialize the context memory manager.
        
        Args:
            max_history: Maximum number of history items to keep
            store: Store implementation to use (defaults to InMemoryStore)
        """
        super().__init__(max_history)
        self.store = store if store is not None else InMemoryStore()
        self.context = {}  # Additional context beyond conversation history
    
    def save_memory(self, message: Dict) -> None:
        """Save a message to memory.
        
        Args:
            message: Message to save
        """
        self.store.add_history(message)
        
    def load_memory(self) -> List[Dict]:
        """Load memory from storage.
        
        Returns:
            List of conversation messages
        """
        return self.store.load_history()
    
    def add(self, message: Dict) -> None:
        """Add a message to history.
        
        Args:
            message: Message to add
        """
        self.store.add_history(message)
        history = self.store.get_history()
        if len(history) > self.max_history:
            # Create new truncated history
            new_history = history[1:]
            self.store.clear(history=True)
            for msg in new_history:
                self.store.add_history(msg)
            
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get current conversation history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of conversation messages
        """
        return self.store.get_history(limit=limit)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.store.clear(history=True)
        self.context = {}
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search context memory for relevant messages.
        
        In this basic implementation, performs a simple string matching search.
        More sophisticated implementations could use embeddings or other techniques.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching messages
        """
        results = []
        history = self.store.get_history()
        
        # Simple string matching search
        for message in history:
            if 'content' in message and query.lower() in message['content'].lower():
                results.append(message)
                if len(results) >= limit:
                    break
        
        return results
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a value in the working context.
        
        Args:
            key: Context key
            value: Value to store
        """
        self.context[key] = value
    
    def get_context(self, key: Optional[str] = None) -> Any:
        """Get value(s) from the working context.
        
        Args:
            key: Optional key to retrieve a specific value
            
        Returns:
            The requested context value, or all context if key is None
        """
        if key is None:
            return self.context
        return self.context.get(key)
    
    def clear_context(self, key: Optional[str] = None) -> None:
        """Clear working context.
        
        Args:
            key: Optional key to clear a specific value
        """
        if key is None:
            self.context = {}
        elif key in self.context:
            del self.context[key]
    
    def get_formatted_history(self, format_type: str = "default") -> str:
        """Get formatted conversation history for inclusion in prompts.
        
        Args:
            format_type: Type of formatting ('default', 'compact')
            
        Returns:
            Formatted string representation of history
        """
        history = self.get_history()
        
        if format_type == "compact":
            return self._format_compact_history(history)
        else:  # default
            return self._format_default_history(history)
    
    def _format_default_history(self, history: List[Dict]) -> str:
        """Default formatting for conversation history."""
        result = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            result.append(f"{role.upper()}: {content}")
        return "\n".join(result)
    
    def _format_compact_history(self, history: List[Dict]) -> str:
        """Compact formatting for conversation history."""
        result = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            role_short = role[0].upper()  # First letter of role
            result.append(f"{role_short}: {content}")
        return "\n".join(result) 