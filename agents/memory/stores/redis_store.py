"""Redis implementation of the store interface.

This module provides a Redis-backed implementation of the Store base class,
offering persistent storage for conversation history and cache.
"""

import json
from typing import List, Dict, Any, Optional
import redis

from agents.memory.stores.store import Store


class RedisStore(Store):
    """Redis-backed implementation of Store.
    
    This implementation stores conversation history and cache in Redis,
    providing persistence across sessions and potential sharing between instances.
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6379, 
        db: int = 0, 
        prefix: str = "agent:",
        password: Optional[str] = None,
        expire_seconds: Optional[int] = None
    ):
        """Initialize the Redis store.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            prefix: Key prefix for all Redis keys
            password: Optional Redis password
            expire_seconds: Optional TTL for stored entries (None = no expiration)
        """
        super().__init__()  # Initialize empty in-memory history and cache
        
        # Initialize Redis client
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True  # Auto-decode bytes to str
        )
        
        self.prefix = prefix
        self.expire_seconds = expire_seconds
        
        # Keys for storage
        self.history_key = f"{prefix}history"
        self.cache_key = f"{prefix}cache"
        
        # Load any existing data
        self._load_from_redis()
    
    def add_history(self, message: Dict[str, str]) -> None:
        """Add a message to conversation history.
        
        Args:
            message: Dict with 'role' and 'content' keys for LLM conversation
        """
        if "role" in message and "content" in message:
            # Add to in-memory history
            self.history.append(message)
            
            # Update in Redis
            self._save_history_to_redis()
    
    def set_cache(self, key: str, value: Any) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        # Update in-memory cache
        self.cache[key] = value
        
        # Update in Redis
        self._save_cache_to_redis()
    
    def get_history(self, limit: int = 20) -> list:
        """Get conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.history[-limit:] if limit > 0 else self.history
    
    def fetch_cache(self, key: Optional[str] = None) -> Any:
        """Get value(s) from cache.
        
        Args:
            key: Optional key to get specific value. If None, returns all values.
            
        Returns:
            Cached value(s)
        """
        if key is None:
            return self.cache
        return self.cache.get(key)
    
    def clear(self, history: bool = True, cache: bool = True) -> None:
        """Clear all items from the store."""
        # Clear in-memory storage
        if history:
            self.history = []
        if cache:
            self.cache = {}
        
        # Clear in Redis
        if history:
            self.redis.delete(self.history_key)
        if cache:
            self.redis.delete(self.cache_key)
    
    def save_history(self) -> bool:
        """Save conversation history to Redis.
        
        Returns:
            Success status
        """
        return self._save_history_to_redis()
    
    def load_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from Redis.
        
        Returns:
            List of conversation messages
        """
        self._load_from_redis(history_only=True)
        return self.history
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        self.redis.delete(self.history_key)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.redis.delete(self.cache_key)
    
    # Async methods
    
    async def async_add_history(self, message: Dict[str, str]) -> None:
        """Asynchronously add a message to conversation history."""
        # For simplicity, using sync implementation
        self.add_history(message)
    
    async def async_set_cache(self, key: str, value: Any) -> None:
        """Asynchronously set a value in the cache."""
        # For simplicity, using sync implementation
        self.set_cache(key, value)
    
    async def async_get_history(self) -> list:
        """Asynchronously get conversation history."""
        return self.history
    
    async def async_fetch_cache(self, key: Optional[str] = None) -> Any:
        """Asynchronously get value(s) from cache."""
        return self.fetch_cache(key)
    
    async def async_clear_history(self) -> None:
        """Asynchronously clear conversation history."""
        self.clear_history()
    
    async def async_clear_cache(self) -> None:
        """Asynchronously clear the cache."""
        self.clear_cache()
    
    # Helper methods
    
    def _save_history_to_redis(self) -> bool:
        """Save history to Redis.
        
        Returns:
            Success status
        """
        try:
            serialized = json.dumps(self.history)
            self.redis.set(self.history_key, serialized)
            
            # Set expiration if configured
            if self.expire_seconds:
                self.redis.expire(self.history_key, self.expire_seconds)
                
            return True
        except Exception:
            return False
    
    def _save_cache_to_redis(self) -> bool:
        """Save cache to Redis.
        
        Returns:
            Success status
        """
        try:
            serialized = json.dumps(self.cache)
            self.redis.set(self.cache_key, serialized)
            
            # Set expiration if configured
            if self.expire_seconds:
                self.redis.expire(self.cache_key, self.expire_seconds)
                
            return True
        except Exception:
            return False
    
    def _load_from_redis(self, history_only: bool = False) -> None:
        """Load data from Redis.
        
        Args:
            history_only: If True, only load history
        """
        # Load history
        history_data = self.redis.get(self.history_key)
        if history_data:
            try:
                self.history = json.loads(history_data)
            except json.JSONDecodeError:
                self.history = []
        
        # Load cache
        if not history_only:
            cache_data = self.redis.get(self.cache_key)
            if cache_data:
                try:
                    self.cache = json.loads(cache_data)
                except json.JSONDecodeError:
                    self.cache = {} 