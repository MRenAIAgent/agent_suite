"""Abstract base class for memory storage backends.

This module provides the base Store class that defines the interface
all store implementations must implement.
"""

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
        """Add a message to conversation history."""

    @abstractmethod
    def get_history(self, limit: int = -1) -> list:
        """Get conversation history."""

    @abstractmethod
    def set_cache(self, key: str, value: Any) -> None:
        """Set a value in the cache."""

    @abstractmethod
    def fetch_cache(self, key: Optional[str] = None) -> Any:
        """Get value(s) from cache."""

    @abstractmethod
    def clear_history(self) -> None:
        """Clear conversation history."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the cache."""
        
    @abstractmethod
    def clear(self, history: bool = True, cache: bool = True) -> None:
        """Clear specified storage components."""
        
    @abstractmethod
    def save_history(self) -> bool:
        """Save conversation history to persistent storage."""
        
    @abstractmethod
    def load_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from persistent storage."""

    # Async methods
    @abstractmethod
    async def async_add_history(self, message: dict) -> None:
        """Asynchronously add a message to conversation history."""

    @abstractmethod
    async def async_get_history(self) -> list:
        """Asynchronously get conversation history."""

    @abstractmethod
    async def async_set_cache(self, key: str, value: Any) -> None:
        """Asynchronously set a value in the cache."""

    @abstractmethod
    async def async_fetch_cache(self, key: Optional[str] = None) -> Any:
        """Asynchronously get value(s) from cache."""

    @abstractmethod
    async def async_clear_history(self) -> None:
        """Asynchronously clear conversation history."""

    @abstractmethod
    async def async_clear_cache(self) -> None:
        """Asynchronously clear the cache.""" 