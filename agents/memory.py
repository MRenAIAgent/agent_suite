from typing import List, Dict, Any, Optional
from agents.memory_stores.mem_store import MemoryStore


class MemoryManager:
    """Manages conversation history and summarization."""
    
    def __init__(self, max_history: int = 20):
        self.store = MemoryStore()  # MemoryStore is a concrete implementation of Store
        self.max_history = max_history
        
    def add(self, message: Dict):
        """Add a message to history."""
        self.store.add_history(message)
        history = self.store.get_history()
        if len(history) > self.max_history:
            # Create new truncated history
            new_history = history[1:]
            self.store.clear(history=True)
            for msg in new_history:
                self.store.add_history(msg)
            
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get current conversation history."""
        return self.store.get_history(limit=limit)
    
    def clear(self):
        """Clear conversation history."""
        self.store.clear(history=True, cache=False)