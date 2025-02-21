from typing import List, Dict, Any, Optional
from agents.memory_stores.mem_store import MemoryStore

class CacheManager:
    """Manages persistent data cache for the agent."""
    
    def __init__(self):
        self.store = MemoryStore()
        
    def set(self, key: str, value: Any):
        """Set a value in cache."""
        self.store.set_cache(key, value)
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        return self.store.fetch_cache(key)
    
    def clear(self):
        """Clear the cache."""
        self.store.clear(history=False, cache=True)

