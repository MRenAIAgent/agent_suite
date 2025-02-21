from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List


class Store(ABC):
    """Abstract base class for memory and cache storage."""
    history: List[Dict[str, Any]]
    cache: Dict[str, Any]

    def __init__(self):
        self.history = []
        self.cache = {}

    @abstractmethod
    def add_history(self, message: dict) -> None:
        """Add a message to conversation history.
        
        Args:
            message: Dict with 'role' and 'content' keys for LLM conversation
        """

    @abstractmethod 
    async def async_add_history(self, message: dict) -> None:
        """Asynchronously add a message to conversation history.
        
        Args:
            message: Dict with 'role' and 'content' keys for LLM conversation
        """

    @abstractmethod
    def set_cache(self, key: str, value: Any) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """

    @abstractmethod
    async def async_set_cache(self, key: str, value: Any) -> None:
        """Asynchronously set a value in the cache.
        
        Args:
            key: Cache key 
            value: Value to store
        """

    @abstractmethod
    def get_history(self, limit: int = -1) -> list:
        """Get conversation history.
        
        Returns:
            List of conversation messages
        """

    @abstractmethod
    async def async_get_history(self) -> list:
        """Asynchronously get conversation history.
        
        Returns:
            List of conversation messages
        """

    @abstractmethod
    def fetch_cache(self, key: Optional[str] = None) -> Any:
        """Get value(s) from cache.
        
        Args:
            key: Optional key to get specific value. If None, returns all values.
            
        Returns:
            Cached value(s)
        """

    @abstractmethod
    async def async_fetch_cache(self, key: Optional[str] = None) -> Any:
        """Asynchronously get value(s) from cache.
        
        Args:
            key: Optional key to get specific value. If None, returns all values.
            
        Returns:
            Cached value(s)
        """

    @abstractmethod
    def clear_history(self) -> None:
        """Clear conversation history."""

    @abstractmethod
    async def async_clear_history(self) -> None:
        """Asynchronously clear conversation history."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the cache."""

    @abstractmethod
    async def async_clear_cache(self) -> None:
        """Asynchronously clear the cache."""