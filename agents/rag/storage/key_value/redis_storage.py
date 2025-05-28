"""Redis key-value storage adaptor for RAG system.

This module provides a Redis-based implementation of the KeyValueStorageAdaptor
interface for storing and retrieving key-value data in the RAG system.
"""

import json
import pickle
from typing import Any, Dict, List, Optional

from agents.rag.storage.key_value.base import KeyValueStorageAdaptor
from agents.rag.core.entities import ContextItem

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisKeyValueStorageAdaptor(KeyValueStorageAdaptor):
    """Redis implementation of the key-value storage adaptor for RAG."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        namespace: str = "rag_math_learning",
        **kwargs
    ):
        """Initialize the Redis key-value storage adaptor.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            namespace: Namespace for keys
            **kwargs: Additional Redis connection parameters
            
        Raises:
            ImportError: If Redis client is not installed
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis client is required for RedisKeyValueStorageAdaptor. "
                "Install it with: pip install redis"
            )
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.namespace = namespace
        
        # Initialize Redis client
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # We'll handle encoding ourselves
            **kwargs
        )
        
        # Test connection
        try:
            self.client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def _make_key(self, key: str) -> str:
        """Create a namespaced key.
        
        Args:
            key: The original key
            
        Returns:
            Namespaced key
        """
        return f"{self.namespace}:{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize a value for storage.
        
        Args:
            value: The value to serialize
            
        Returns:
            Serialized bytes
        """
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize a value from storage.
        
        Args:
            data: The serialized data
            
        Returns:
            Deserialized value
        """
        return pickle.loads(data)
    
    async def set(self, key: str, value: Any) -> bool:
        """Set a key-value pair.
        
        Args:
            key: The key
            value: The value
            
        Returns:
            True if successful
        """
        try:
            namespaced_key = self._make_key(key)
            serialized_value = self._serialize_value(value)
            return self.client.set(namespaced_key, serialized_value)
        except Exception:
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key.
        
        Args:
            key: The key
            
        Returns:
            The value if found, None otherwise
        """
        try:
            namespaced_key = self._make_key(key)
            data = self.client.get(namespaced_key)
            if data is None:
                return None
            return self._deserialize_value(data)
        except Exception:
            return None
    
    async def get_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """Get all key-value pairs with keys matching a prefix.
        
        Args:
            prefix: The key prefix
            
        Returns:
            Dictionary of matching key-value pairs
        """
        try:
            pattern = self._make_key(f"{prefix}*")
            keys = self.client.keys(pattern)
            
            result = {}
            for namespaced_key in keys:
                # Remove namespace prefix to get original key
                original_key = namespaced_key.decode('utf-8').replace(f"{self.namespace}:", "", 1)
                data = self.client.get(namespaced_key)
                if data is not None:
                    result[original_key] = self._deserialize_value(data)
            
            return result
        except Exception:
            return {}
    
    async def store_context_item(self, item: ContextItem) -> str:
        """Store a context item.
        
        Args:
            item: The context item to store
            
        Returns:
            The item key
        """
        await self.set(item.key, item.value)
        return item.key
    
    async def get_all_context(self, prefix: str = "context_") -> List[ContextItem]:
        """Get all context items with a given prefix.
        
        Args:
            prefix: The key prefix for context items
            
        Returns:
            List of context items
        """
        try:
            context_data = await self.get_by_prefix(prefix)
            context_items = []
            
            for key, value in context_data.items():
                context_items.append(ContextItem(key=key, value=value))
            
            return context_items
        except Exception:
            return []
    
    async def store(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data in the key-value storage.
        
        Args:
            data: The data to store
            metadata: Optional metadata
            
        Returns:
            Storage identifier
        """
        # Use 'id' field as key if available, otherwise generate one
        key = data.get("id", f"item_{hash(str(data))}")
        
        # Include metadata in the stored data
        stored_data = data.copy()
        if metadata:
            stored_data["_metadata"] = metadata
        
        await self.set(key, stored_data)
        return key
    
    async def retrieve(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve data by key.
        
        Args:
            key: The data key
            **kwargs: Additional parameters
            
        Returns:
            The retrieved data
        """
        return await self.get(key)
    
    async def update(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """Update data by key.
        
        Args:
            key: The data key
            data: Updated data
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        # Get existing data
        existing = await self.get(key)
        if existing is None:
            return False
        
        # Merge with new data
        if isinstance(existing, dict):
            existing.update(data)
            return await self.set(key, existing)
        else:
            # Replace entirely if not a dict
            return await self.set(key, data)
    
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete data by key.
        
        Args:
            key: The data key
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        try:
            namespaced_key = self._make_key(key)
            return bool(self.client.delete(namespaced_key))
        except Exception:
            return False
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for data matching a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Search results
        """
        try:
            # Simple pattern matching on keys
            pattern = self._make_key(f"*{query}*")
            keys = self.client.keys(pattern)
            
            results = []
            for namespaced_key in keys[:limit]:
                original_key = namespaced_key.decode('utf-8').replace(f"{self.namespace}:", "", 1)
                data = self.client.get(namespaced_key)
                if data is not None:
                    value = self._deserialize_value(data)
                    results.append({
                        "key": original_key,
                        "value": value
                    })
            
            return results
        except Exception:
            return []
    
    async def close(self) -> None:
        """Close the storage connection."""
        if hasattr(self, 'client'):
            self.client.close() 