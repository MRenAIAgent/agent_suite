"""StorageRouter middleware for the RAG system.

This module provides the router that determines which storage backend(s)
to use for different types of data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Tuple
from enum import Enum

from agent_suite.rag.models.document import Document
from agent_suite.rag.models.knowledge import Entity, Relationship, KnowledgeGraph
from agent_suite.rag.models.context import ContextItem


class StorageType(Enum):
    """Types of storage backends."""
    VECTOR = "vector"
    GRAPH = "graph"
    KEY_VALUE = "key_value"
    ALL = "all"


class StorageStrategy(ABC):
    """Abstract base class for storage routing strategies."""
    
    @abstractmethod
    def determine_storage(self, data: Any, metadata: Dict[str, Any]) -> List[StorageType]:
        """
        Determine which storage types should be used for the given data.
        
        Args:
            data: The data to store
            metadata: Additional metadata about the data
            
        Returns:
            List of storage types to use
        """
        pass


class DefaultStorageStrategy(StorageStrategy):
    """Default strategy for routing data to appropriate storage."""
    
    def determine_storage(self, data: Any, metadata: Dict[str, Any]) -> List[StorageType]:
        """
        Determine storage based on data type and metadata.
        
        Default routing:
        - Documents -> Vector
        - Entity/Relationship/KnowledgeGraph -> Graph
        - ContextItem -> Key-Value
        - Metadata can override with "storage_type" key
        """
        # Check for explicit storage directive in metadata
        if "storage_type" in metadata:
            storage_type = metadata["storage_type"]
            if isinstance(storage_type, str):
                try:
                    return [StorageType(storage_type)]
                except ValueError:
                    pass
            elif isinstance(storage_type, list):
                try:
                    return [StorageType(st) for st in storage_type]
                except ValueError:
                    pass
        
        # Default routing based on data type
        if isinstance(data, Document):
            return [StorageType.VECTOR]
        elif isinstance(data, (Entity, Relationship, KnowledgeGraph)):
            return [StorageType.GRAPH]
        elif isinstance(data, ContextItem):
            return [StorageType.KEY_VALUE]
        else:
            # Unknown data type, store in all for safety
            return [StorageType.ALL]


class StorageRouter:
    """
    Router that determines which storage backend(s) to use for different data.
    """
    
    def __init__(self):
        """Initialize the storage router."""
        self.strategies = []
        # Register default strategy
        self.register_strategy(DefaultStorageStrategy())
    
    def register_strategy(self, strategy: StorageStrategy) -> None:
        """
        Register a new storage routing strategy.
        
        Args:
            strategy: The strategy to register
        """
        self.strategies.append(strategy)
    
    def route(self, data: Any, metadata: Dict[str, Any] = None) -> List[StorageType]:
        """
        Determine where data should be stored based on registered strategies.
        
        Args:
            data: The data to route
            metadata: Optional metadata to help with routing
            
        Returns:
            List of storage types to use
        """
        metadata = metadata or {}
        
        # Try each strategy in order
        for strategy in self.strategies:
            storage_types = strategy.determine_storage(data, metadata)
            if storage_types:
                return storage_types
        
        # Default to all storage types if no strategy matched
        return [StorageType.ALL] 