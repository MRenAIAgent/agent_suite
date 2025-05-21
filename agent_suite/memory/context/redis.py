"""
Redis-based implementation of context memory.
"""

import json
from typing import Dict, List, Any, Optional

import redis

from agent_suite.memory.context.base import ContextMemory


class RedisContextMemory(ContextMemory):
    """
    Redis-based implementation of context memory.
    
    This implementation stores context memory in Redis, making it persistent
    and shareable across processes and services.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        namespace: str = "agent_context:",
        ttl: Optional[int] = None  # Time-to-live in seconds
    ):
        """
        Initialize Redis context memory.
        
        Args:
            redis_url: Redis connection string
            namespace: Prefix for Redis keys
            ttl: Optional time-to-live for memory items (in seconds)
        """
        self.redis_client = redis.from_url(redis_url)
        self.namespace = namespace
        self.ttl = ttl
    
    def _get_key(self, key: str) -> str:
        """
        Get the namespaced Redis key.
        
        Args:
            key: Original key name
            
        Returns:
            Namespaced key for Redis
        """
        return f"{self.namespace}{key}"
    
    def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new memory item.
        
        Args:
            key: The key/type of the memory item
            value: The value to store
            metadata: Optional metadata about this item
        """
        redis_key = self._get_key(key)
        
        # Serialize the data
        data = {
            "value": value,
            "metadata": metadata or {}
        }
        serialized = json.dumps(data)
        
        # Append to list in Redis
        self.redis_client.lpush(redis_key, serialized)
        
        # Apply TTL if specified
        if self.ttl is not None and not self.redis_client.ttl(redis_key) > 0:
            self.redis_client.expire(redis_key, self.ttl)
    
    def get(self, key: Optional[str] = None) -> Any:
        """
        Retrieve memory items with optional filtering.
        
        Args:
            key: Optional key to filter items (if None, return all items)
            
        Returns:
            Memory items matching the key, or all items if key is None
        """
        if key is not None:
            # Get items for a specific key
            redis_key = self._get_key(key)
            raw_items = self.redis_client.lrange(redis_key, 0, -1)
            return [json.loads(item) for item in raw_items]
        
        # Get all items across all keys
        result = {}
        # Get all keys matching the namespace pattern
        all_keys = self.redis_client.keys(f"{self.namespace}*")
        
        for redis_key in all_keys:
            # Extract original key from Redis key
            original_key = redis_key.decode('utf-8').replace(self.namespace, "")
            raw_items = self.redis_client.lrange(redis_key, 0, -1)
            result[original_key] = [json.loads(item) for item in raw_items]
        
        return result
    
    def get_formatted(self, format_type: str = "default") -> str:
        """
        Get formatted memory for inclusion in LLM prompts.
        
        Args:
            format_type: Type of formatting to use ('default', 'compact', 'detailed')
            
        Returns:
            Formatted string representation of context memory
        """
        # Get all memory items
        memory_dict = self.get()
        
        if format_type == "default":
            return self._format_default(memory_dict)
        elif format_type == "compact":
            return self._format_compact(memory_dict)
        elif format_type == "detailed":
            return self._format_detailed(memory_dict)
        else:
            return self._format_default(memory_dict)
    
    def clear(self) -> None:
        """
        Clear all memory items.
        """
        # Delete all keys matching the namespace pattern
        keys = self.redis_client.keys(f"{self.namespace}*")
        if keys:
            self.redis_client.delete(*keys)
    
    def _format_default(self, memory_dict: Dict[str, List[Dict]]) -> str:
        """
        Default format for memory presentation.
        """
        result = []
        for key, items in memory_dict.items():
            result.append(f"==== {key.upper()} ====")
            for item in items:
                result.append(f"- {item['value']}")
            result.append("")  # Empty line
        return "\n".join(result)
    
    def _format_compact(self, memory_dict: Dict[str, List[Dict]]) -> str:
        """
        Compact format for memory presentation.
        """
        result = []
        for key, items in memory_dict.items():
            values = [str(item["value"]) for item in items]
            result.append(f"{key}: {', '.join(values)}")
        return "\n".join(result)
    
    def _format_detailed(self, memory_dict: Dict[str, List[Dict]]) -> str:
        """
        Detailed format for memory presentation including metadata.
        """
        result = []
        for key, items in memory_dict.items():
            result.append(f"==== {key.upper()} ====")
            for idx, item in enumerate(items):
                result.append(f"Item {idx+1}:")
                result.append(f"  Value: {item['value']}")
                result.append(f"  Metadata: {json.dumps(item['metadata'], indent=2)}")
            result.append("")  # Empty line
        return "\n".join(result) 