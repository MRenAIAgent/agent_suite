"""Vector storage interface.

This module defines specialized interfaces for vector storage operations,
extending the base storage interface with vector-specific functionality.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np

from agents.storage.base_storage import BaseStorage


class VectorStorage(BaseStorage[Dict[str, Any]]):
    """Interface for vector storage operations.
    
    Extends the base storage interface with vector-specific operations
    like similarity search and metadata filtering.
    """
    
    async def store_vector(
        self, 
        key: str, 
        vector: Union[List[float], np.ndarray], 
        metadata: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> str:
        """Store a vector with associated metadata.
        
        Args:
            key: Unique identifier for the vector
            vector: The vector to store
            metadata: Additional metadata to associate with the vector
            **kwargs: Additional storage-specific parameters
            
        Returns:
            Identifier for the stored vector
        """
        data = {
            "vector": vector,
            "metadata": metadata or {}
        }
        return await self.store(key, data, **kwargs)
    
    async def store_batch(
        self, 
        items: List[Tuple[str, Union[List[float], np.ndarray], Dict[str, Any]]], 
        **kwargs
    ) -> List[str]:
        """Store multiple vectors in a batch operation.
        
        Args:
            items: List of (key, vector, metadata) tuples
            **kwargs: Additional storage-specific parameters
            
        Returns:
            List of identifiers for the stored vectors
        """
        raise NotImplementedError("Batch storage must be implemented by subclasses")
    
    async def search_by_vector(
        self, 
        query_vector: Union[List[float], np.ndarray], 
        limit: int = 10, 
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Vector to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_metadata: Metadata filter criteria
            **kwargs: Additional search-specific parameters
            
        Returns:
            List of results containing vectors, metadata, and similarity scores
        """
        raise NotImplementedError("Vector search must be implemented by subclasses")
    
    async def search_by_metadata(
        self, 
        filter_metadata: Dict[str, Any], 
        limit: int = 10, 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for vectors by metadata.
        
        Args:
            filter_metadata: Metadata filter criteria
            limit: Maximum number of results to return
            **kwargs: Additional search-specific parameters
            
        Returns:
            List of results containing vectors and metadata
        """
        raise NotImplementedError("Metadata search must be implemented by subclasses") 