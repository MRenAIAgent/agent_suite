"""Storage service for managing multiple storage backends.

This module provides a unified service for interacting with different storage backends,
supporting operations across vector, graph, and document storage.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Type

from agents.storage.base_storage import BaseStorage
from agents.storage.vector_storage import VectorStorage
from agents.storage.graph_storage import GraphStorage
from agents.storage.document_storage import DocumentStorage


class StorageService:
    """Unified storage service for managing different backends.
    
    This service provides a central point for interacting with different storage
    technologies while maintaining a consistent API.
    """
    
    def __init__(self):
        """Initialize the storage service."""
        self.logger = logging.getLogger(__name__)
        self.backends: Dict[str, BaseStorage] = {}
        
    async def register_backend(self, name: str, backend: BaseStorage) -> None:
        """Register a storage backend.
        
        Args:
            name: Name to identify the backend
            backend: Storage backend implementation
        """
        self.backends[name] = backend
        self.logger.info(f"Registered storage backend: {name}")
        
    async def get_backend(self, name: str) -> Optional[BaseStorage]:
        """Get a registered backend by name.
        
        Args:
            name: Name of the backend to retrieve
            
        Returns:
            The backend or None if not found
        """
        return self.backends.get(name)
    
    async def get_backend_by_type(self, storage_type: Type) -> List[BaseStorage]:
        """Get all backends of a specific type.
        
        Args:
            storage_type: Type of storage to retrieve
            
        Returns:
            List of matching backends
        """
        return [b for b in self.backends.values() if isinstance(b, storage_type)]
    
    async def get_vector_backends(self) -> List[VectorStorage]:
        """Get all registered vector storage backends.
        
        Returns:
            List of vector storage backends
        """
        return await self.get_backend_by_type(VectorStorage)
    
    async def get_graph_backends(self) -> List[GraphStorage]:
        """Get all registered graph storage backends.
        
        Returns:
            List of graph storage backends
        """
        return await self.get_backend_by_type(GraphStorage)
    
    async def get_document_backends(self) -> List[DocumentStorage]:
        """Get all registered document storage backends.
        
        Returns:
            List of document storage backends
        """
        return await self.get_backend_by_type(DocumentStorage)
    
    async def remove_backend(self, name: str) -> bool:
        """Remove a registered backend.
        
        Args:
            name: Name of the backend to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.backends:
            del self.backends[name]
            self.logger.info(f"Removed storage backend: {name}")
            return True
        return False 