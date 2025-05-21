"""
Redis vector database implementation for long-term memory.
"""

import uuid
import json
import datetime
from typing import Any, Dict, List, Optional, Union

import redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from agent_suite.memory.long_term.vector.base import VectorMemory


class RedisVectorMemory(VectorMemory):
    """
    Long-term memory implementation using Redis Vector Search.
    
    This implementation uses Redis with RediSearch for vector similarity search
    to store and retrieve memories based on semantic similarity.
    """
    
    def __init__(
        self,
        embedding_service,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "agent_memory",
        prefix: str = "agent:",
        vector_dimensions: int = 384,
        distance_metric: str = "COSINE",
        user_id: Optional[str] = None
    ):
        """
        Initialize Redis vector memory.
        
        Args:
            embedding_service: Service to generate embeddings
            redis_url: Redis connection string
            index_name: Name of the Redis search index
            prefix: Prefix for Redis keys
            vector_dimensions: Size of embedding vectors
            distance_metric: Distance metric for similarity ('COSINE', 'IP', 'L2')
            user_id: Optional user ID for user-specific memory
        """
        self.embedding_service = embedding_service
        self.redis_client = redis.from_url(redis_url)
        self.index_name = index_name
        self.prefix = prefix
        self.vector_dimensions = vector_dimensions
        self.distance_metric = distance_metric
        self.user_id = user_id
        
        # Initialize the Redis search index
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """
        Initialize the Redis search index if it doesn't exist.
        """
        try:
            # Check if index exists
            self.redis_client.ft(self.index_name).info()
            return  # Index already exists
        except redis.exceptions.ResponseError:
            # Index doesn't exist, create it
            pass
        
        # Define fields for the index
        schema = [
            # Text fields
            TextField("text"),
            TextField("user_id"),
            TextField("memory_id"),
            # Vector field for embeddings
            VectorField("embedding",
                        "FLAT", {
                            "TYPE": "FLOAT32",
                            "DIM": self.vector_dimensions,
                            "DISTANCE_METRIC": self.distance_metric,
                        }),
            # Metadata fields
            TextField("metadata_json"),
            NumericField("timestamp")
        ]
        
        # Create the index
        self.redis_client.ft(self.index_name).create_index(
            schema,
            definition=IndexDefinition(
                prefix=[self.prefix],
                index_type=IndexType.HASH
            )
        )
    
    def _get_key(self, memory_id: str) -> str:
        """Get the Redis key for a memory item."""
        return f"{self.prefix}{memory_id}"
    
    def store(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store text data with vector embedding.
        
        Args:
            data: Text to store
            metadata: Optional metadata
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        # Generate embedding
        embedding = self.embedding_service.generate_embedding(data)
        
        # Generate a unique ID
        memory_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
        
        # Prepare data for Redis
        redis_data = {
            "text": data,
            "embedding": embedding.tobytes(),
            "memory_id": memory_id,
            "metadata_json": json.dumps(metadata),
            "timestamp": int(datetime.datetime.fromisoformat(metadata["timestamp"]).timestamp())
        }
        
        # Add user_id if present
        if self.user_id:
            redis_data["user_id"] = self.user_id
        
        # Store in Redis
        self.redis_client.hset(self._get_key(memory_id), mapping=redis_data)
        
        return memory_id
    
    def query(self, query: Union[str, Dict], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query for similar vectors.
        
        Args:
            query: Text query or dict with query parameters
            limit: Maximum number of results
            
        Returns:
            List of matching memories with similarity scores
        """
        # Process the query
        if isinstance(query, str):
            # Convert text query to embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            query_string = f"*=>[KNN {limit} @embedding $embedding]"
            params = {"embedding": query_embedding.tobytes()}
            
            # Add user filter if user_id is set
            if self.user_id:
                query_string = f"@user_id:{self.user_id} {query_string}"
            
        else:
            # Handle structured query
            if "text" in query:
                # Convert text query to embedding
                query_embedding = self.embedding_service.generate_embedding(query["text"])
                params = {"embedding": query_embedding.tobytes()}
                filter_parts = []
                
                # Add user filter if provided in query or as instance var
                user_id = query.get("user_id", self.user_id)
                if user_id:
                    filter_parts.append(f"@user_id:{user_id}")
                
                # Build the query string
                filter_str = " ".join(filter_parts)
                if filter_str:
                    query_string = f"{filter_str} *=>[KNN {limit} @embedding $embedding]"
                else:
                    query_string = f"*=>[KNN {limit} @embedding $embedding]"
            else:
                raise ValueError("Structured query must include 'text' field")
        
        # Execute the query
        result = self.redis_client.ft(self.index_name).search(
            Query(query_string).dialect(2),
            query_params=params
        )
        
        # Process and return results
        memories = []
        for doc in result.docs:
            # Parse metadata
            try:
                metadata = json.loads(doc.metadata_json)
            except (json.JSONDecodeError, AttributeError):
                metadata = {}
            
            # Create memory item
            memory = {
                "memory_id": doc.memory_id,
                "text": doc.text,
                "metadata": metadata,
                "score": 1 - float(doc.dist) if hasattr(doc, "dist") else 1.0  # Convert distance to similarity
            }
            
            memories.append(memory)
        
        return memories
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory item.
        
        Args:
            memory_id: The unique identifier of the memory
            
        Returns:
            True if deleted successfully, False otherwise
        """
        result = self.redis_client.delete(self._get_key(memory_id))
        return result > 0
    
    def clear(self) -> None:
        """
        Clear all stored memories.
        
        If user_id is set, only clear that user's memories.
        """
        if self.user_id:
            # Find all keys for this user
            query = f"@user_id:{self.user_id}"
            result = self.redis_client.ft(self.index_name).search(Query(query))
            
            # Delete each document
            for doc in result.docs:
                self.redis_client.delete(doc.id)
        else:
            # Delete all keys with prefix
            keys = self.redis_client.keys(f"{self.prefix}*")
            if keys:
                self.redis_client.delete(*keys) 