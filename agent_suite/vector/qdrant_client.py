"""
Centralized Qdrant vector database client for Agent Suite.

This module provides a shared interface for Qdrant operations that can be used
by both the knowledge module and long-term storage features.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import uuid

# Load environment variables for API keys
load_dotenv()

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Centralized manager for Qdrant vector database operations.
    
    This class provides shared functionality for creating collections,
    managing points, and querying vectors across the Agent Suite.
    """
    
    def __init__(
        self,
        connection_url: Optional[str] = None,
        api_key: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the Qdrant manager.
        
        Args:
            connection_url: URL for Qdrant server (cloud or self-hosted)
            api_key: API key for Qdrant cloud (if not provided in env)
            persist_directory: Directory for local storage (if not using a server)
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        # Determine client connection type
        if connection_url:
            # Connect to cloud or self-hosted Qdrant
            self.client = QdrantClient(
                url=connection_url,
                api_key=self.api_key
            )
            self.is_remote = True
        elif persist_directory:
            # Use local persistence
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(persist_path))
            self.is_remote = False
        else:
            # Use in-memory storage (not persistent)
            self.client = QdrantClient(":memory:")
            self.is_remote = False
            
        self._collection_cache = set()
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "cosine",
        recreate: bool = False
    ) -> bool:
        """
        Create a new collection or verify an existing one.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors to store
            distance: Distance metric to use (cosine, euclid, dot)
            recreate: Whether to recreate if collection exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            if collection_exists:
                if recreate:
                    # Delete existing collection if recreate is True
                    self.client.delete_collection(collection_name)
                else:
                    # Collection exists and we're not recreating
                    self._collection_cache.add(collection_name)
                    return True
            
            # Normalize distance string (Qdrant needs capitalized value)
            distance_map = {
                "cosine": "Cosine",
                "euclid": "Euclid",
                "euclidean": "Euclid",
                "dot": "Dot",
                "manhattan": "Manhattan",
                # Already capitalized versions
                "Cosine": "Cosine",
                "Euclid": "Euclid",
                "Dot": "Dot",
                "Manhattan": "Manhattan"
            }
            
            # Get normalized distance or default to Cosine
            normalized_distance = distance_map.get(distance.lower(), "Cosine")
            
            # Create the collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=normalized_distance
                )
            )
            
            # Add to cache
            self._collection_cache.add(collection_name)
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {str(e)}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if exists, False otherwise
        """
        # Check cache first
        if collection_name in self._collection_cache:
            return True
            
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists:
                self._collection_cache.add(collection_name)
                
            return exists
        except Exception as e:
            logger.error(f"Error checking collection {collection_name}: {str(e)}")
            return False
    
    def add_points(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Add points to a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of point IDs
            vectors: List of vector embeddings
            payloads: Optional list of metadata payloads
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure collection exists
            if not self.collection_exists(collection_name):
                raise ValueError(f"Collection {collection_name} does not exist")
                
            # Prepare points
            if payloads is None:
                payloads = [{} for _ in ids]
            
            # Convert ids to UUIDs if needed
            converted_ids = []
            for id_str in ids:
                # If it's already a valid UUID, use it directly
                try:
                    # Try to parse as UUID to validate
                    uuid_obj = uuid.UUID(id_str)
                    converted_ids.append(str(uuid_obj))
                except ValueError:
                    # If not a valid UUID, create a deterministic UUID from the string
                    # Using uuid5 with the DNS namespace for deterministic generation
                    converted_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))
                    converted_ids.append(converted_id)
                    logger.info(f"Converted ID '{id_str}' to UUID: {converted_id}")
                
            points = [
                models.PointStruct(
                    id=id,
                    vector=vector,
                    payload=payload
                )
                for id, vector, payload in zip(converted_ids, vectors, payloads)
            ]
            
            # Add points to collection
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding points to {collection_name}: {str(e)}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Vector to search for
            limit: Maximum number of results
            filter_condition: Optional filter for results
            
        Returns:
            List of matched points with scores and payloads
        """
        try:
            # Convert filter to Qdrant format if provided
            filter_query = None
            if filter_condition:
                filter_query = models.Filter(**filter_condition)
            
            # Execute search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_query
            )
            
            # Format results
            results = []
            for point in search_results:
                # Extract and format result
                result = {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching in {collection_name}: {str(e)}")
            return []
    
    def delete_points(
        self,
        collection_name: str,
        point_ids: List[str]
    ) -> bool:
        """
        Delete points from a collection.
        
        Args:
            collection_name: Name of the collection
            point_ids: IDs of points to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure collection exists
            if not self.collection_exists(collection_name):
                return False
                
            # Delete points
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting points from {collection_name}: {str(e)}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information or None if not found
        """
        try:
            # Ensure collection exists
            if not self.collection_exists(collection_name):
                return None
                
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            
            # Format as dictionary
            info = {
                "name": collection_info.name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting info for {collection_name}: {str(e)}")
            return None 