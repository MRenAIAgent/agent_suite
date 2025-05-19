#!/usr/bin/env python3
"""
Intermediate Storage

This module provides storage classes for tracking intermediate results during agent execution.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class IntermediateStorageBase(ABC):
    """Base abstract class for intermediate storage implementations."""
    
    @abstractmethod
    def store(self, key: str, value: Any) -> None:
        """Store a value with the given key."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored values."""
        pass


class InMemoryIntermediateStorage(IntermediateStorageBase):
    """In-memory implementation of intermediate storage.
    
    This storage uses a simple dictionary to store intermediate data.
    It's efficient but data is lost when the program exits.
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self.data = {}
        
    def store(self, key: str, value: Any) -> None:
        """Store a value with the given key.
        
        Args:
            key: Storage key
            value: Value to store
        """
        self.data[key] = value
        
    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key.
        
        Args:
            key: Storage key
            default: Default value if key not found
            
        Returns:
            Retrieved value or default
        """
        return self.data.get(key, default)
    
    def clear(self) -> None:
        """Clear all stored values."""
        self.data.clear()
    
    def get_all(self) -> Dict[str, Any]:
        """Get all stored values."""
        return self.data.copy()


class TokenEfficientIntermediateStorage(IntermediateStorageBase):
    """
    Token-efficient implementation of intermediate storage.
    Only keeps the most recent iterations to save tokens.
    """
    
    def __init__(self, max_iterations_to_keep: int = 3):
        """Initialize storage with token efficiency settings."""
        self.storage: Dict[str, Any] = {}
        self.max_iterations_to_keep = max_iterations_to_keep
        
    def store(self, key: str, value: Any) -> None:
        """Store a value with the given key."""
        self.storage[key] = value
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        return self.storage.get(key)
    
    def clear(self) -> None:
        """Clear all stored values."""
        self.storage.clear() 