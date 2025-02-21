from typing import Any, Optional, Dict, List
from agents.memory_stores.store import Store


class MemoryStore(Store):
    """In-memory implementation of Store."""

    def add_history(self, message: Dict[str, str]) -> None:
        """Add a message to conversation history.
        
        Args:
            message: Dict with 'role' and 'content' keys for LLM conversation
        """
        if "role" in message and "content" in message:
            self.history.append(message)

    def set_cache(self, key: str, value: Any) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        self.cache[key] = value

    def get_history(self, limit: int = 20) -> list:
        """Get conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.history[-limit:]

    def fetch_cache(self, key: Optional[str] = None) -> Any:
        """Get value(s) from cache.
        
        Args:
            key: Optional key to get specific value. If None, returns all values.
            
        Returns:
            Cached value(s)
        """
        if key is None:
            return self.cache
        return self.cache.get(key)

    def clear(self, history: bool = True, cache: bool = True) -> None:
        """Clear all items from the store."""
        if history:
            self.history = []
        if cache:
            self.cache = {}

    async def async_add_history(self, message: Dict[str, str]) -> None:
        """Asynchronously add a message to conversation history.
        
        Args:
            message: Dict with 'role' and 'content' keys for LLM conversation
        """
        if "role" in message and "content" in message:
            self.history.append(message)

    async def async_set_cache(self, key: str, value: Any) -> None:
        """Asynchronously set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        self.cache[key] = value

    async def async_get_history(self) -> list:
        """Asynchronously get conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.history

    async def async_fetch_cache(self, key: Optional[str] = None) -> Any:
        """Asynchronously get value(s) from cache.
        
        Args:
            key: Optional key to get specific value. If None, returns all values.
            
        Returns:
            Cached value(s)
        """
        if key is None:
            return self.cache
        return self.cache.get(key)

    async def async_clear_history(self) -> None:
        """Asynchronously clear conversation history."""
        self.history = []

    async def async_clear_cache(self) -> None:
        """Asynchronously clear the cache."""
        self.cache = {}

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []

    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}