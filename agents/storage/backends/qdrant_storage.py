"""Qdrant vector storage implementation.

This module provides a VectorStorage implementation using Qdrant,
a high-performance vector database for similarity search.
"""

import json
import uuid
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple

from agents.storage.vector_storage import VectorStorage

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantStorage(VectorStorage):
    """Qdrant implementation of the vector storage interface.
    
    Provides vector storage and similarity search capabilities using
    Qdrant as the backend.
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        vector_size: int = 1536,
        url: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        **kwargs
    ):
        """Initialize the Qdrant storage.
        
        Args:
            collection_name: Name of the Qdrant collection to use
            vector_size: Dimensionality of vectors to store
            url: URL for Qdrant (e.g., "https://example.com")
            host: Hostname for Qdrant server
            port: Port for Qdrant server
            api_key: API key for authenticated access
            prefer_grpc: Whether to prefer gRPC over HTTP
            **kwargs: Additional Qdrant client configuration
        
        Raises:
            ImportError: If Qdrant client is not installed
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client is required for QdrantStorage. "
                "Install it with: pip install qdrant-client"
            )
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize client
        if url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                **kwargs
            )
        else:
            self.client = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                **kwargs
            )
        
        # Ensure collection exists
        self._ensure_collection()
        
    def _ensure_collection(self) -> None:
        """Ensure the collection exists in Qdrant."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
    
    async def store(self, key: str, data: Dict[str, Any], **kwargs) -> str:
        """Store data with the given key.
        
        Args:
            key: Unique identifier for the data
            data: The data to store (containing vector and metadata)
            **kwargs: Additional storage parameters
            
        Returns:
            Identifier for the stored data
        """
        # Generate point ID if not provided
        point_id = kwargs.get("id", key) or str(uuid.uuid4())
        
        vector = data.get("vector", [])
        
        # Convert nested dictionaries to strings in metadata for compatibility
        metadata = data.get("metadata", {})
        for k, v in metadata.items():
            if isinstance(v, (dict, list)):
                metadata[k] = json.dumps(v)
        
        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=metadata
                )
            ]
        )
        
        return point_id
    
    async def retrieve(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve data by key.
        
        Args:
            key: Unique identifier for the data
            **kwargs: Additional retrieval parameters
            
        Returns:
            The stored data or None if not found
        """
        try:
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[key]
            )
            
            if not point:
                return None
                
            point = point[0]
            
            # Convert string representations back to objects
            payload = point.payload or {}
            for k, v in payload.items():
                if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                    try:
                        payload[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            
            return {
                "id": point.id,
                "vector": point.vector,
                "metadata": payload
            }
        except Exception as e:
            # Handle case where point doesn't exist
            return None
    
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete data by key.
        
        Args:
            key: Unique identifier for the data to delete
            **kwargs: Additional deletion parameters
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[key]
                )
            )
            return True
        except Exception:
            return False
    
    async def update(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """Update data associated with the key.
        
        Args:
            key: Unique identifier for the data to update
            data: New data to store
            **kwargs: Additional update parameters
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Retrieve existing point to see if it exists
            existing = await self.retrieve(key)
            if not existing:
                return False
            
            # Implement as store operation
            await self.store(key, data, id=key, **kwargs)
            return True
        except Exception:
            return False
    
    async def list(self, pattern: str = "*", **kwargs) -> List[str]:
        """List keys matching a pattern.
        
        Args:
            pattern: Pattern to match keys against
            **kwargs: Additional parameters for the listing operation
            
        Returns:
            List of matching keys
        """
        # Qdrant doesn't have a direct pattern matching for IDs
        # Instead, implement a scan operation with limit
        limit = kwargs.get("limit", 100)
        
        try:
            # Limited implementation - get points and extract IDs
            # For production, consider a scrolling API or pagination
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit
            )
            
            points = scroll_result[0]  # First element contains points
            return [str(point.id) for point in points]
        except Exception:
            return []
    
    async def clear(self, **kwargs) -> bool:
        """Clear all data from the storage.
        
        Args:
            **kwargs: Additional parameters for the clear operation
            
        Returns:
            True if clear was successful, False otherwise
        """
        try:
            # If we want to keep the collection but remove points
            if kwargs.get("keep_collection", True):
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter()  # Empty filter means all points
                    )
                )
            else:
                # Delete the entire collection
                self.client.delete_collection(
                    collection_name=self.collection_name
                )
                # Recreate it
                self._ensure_collection()
                
            return True
        except Exception:
            return False
            
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
        point_ids = []
        points = []
        
        for key, vector, metadata in items:
            # Generate point ID if not provided
            point_id = key or str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Convert nested dictionaries to strings in metadata for compatibility
            metadata_copy = metadata.copy() if metadata else {}
            for k, v in metadata_copy.items():
                if isinstance(v, (dict, list)):
                    metadata_copy[k] = json.dumps(v)
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=metadata_copy
                )
            )
        
        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return point_ids
    
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
        # Convert filter_metadata to Qdrant filter format
        qdrant_filter = None
        if filter_metadata:
            filter_conditions = []
            for key, value in filter_metadata.items():
                if isinstance(value, list):
                    # For list values, use any of the values
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    # For single values, use exact match
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            
            qdrant_filter = models.Filter(
                must=filter_conditions
            )
        
        # Perform the search
        search_params = models.SearchParams(
            hnsw_ef=kwargs.get("hnsw_ef", 128),
            exact=kwargs.get("exact", False)
        )
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filter=qdrant_filter,
            search_params=search_params,
            with_vectors=True
        )
        
        # Convert results to standard format
        results = []
        for scored_point in search_result:
            # Convert string representations back to objects
            payload = scored_point.payload or {}
            for k, v in payload.items():
                if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                    try:
                        payload[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            
            results.append({
                "id": scored_point.id,
                "vector": scored_point.vector,
                "metadata": payload,
                "score": scored_point.score
            })
        
        return results
    
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
        # Convert filter_metadata to Qdrant filter format
        filter_conditions = []
        for key, value in filter_metadata.items():
            if isinstance(value, list):
                # For list values, use any of the values
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )
            else:
                # For single values, use exact match
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        qdrant_filter = models.Filter(
            must=filter_conditions
        )
        
        # Perform the search
        search_result = self.client.scroll(
            collection_name=self.collection_name,
            filter=qdrant_filter,
            limit=limit,
            with_vectors=True
        )
        
        # Convert results to standard format
        results = []
        for point in search_result[0]:  # First element contains points
            # Convert string representations back to objects
            payload = point.payload or {}
            for k, v in payload.items():
                if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                    try:
                        payload[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            
            results.append({
                "id": point.id,
                "vector": point.vector,
                "metadata": payload
            })
        
        return results 