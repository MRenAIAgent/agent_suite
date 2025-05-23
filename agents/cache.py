from typing import List, Dict, Any, Optional
from agents.memory.stores.in_memory_store import InMemoryStore

class CacheManager:
    """Manages persistent data cache for the agent."""
    
    def __init__(self):
        self.store = InMemoryStore()
        
    def set(self, key: str, value: Any):
        """Set a value in cache."""
        self.store.set_cache(key, value)
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        return self.store.fetch_cache(key)
    
    def clear(self):
        """Clear the cache."""
        self.store.clear(history=False, cache=True)

