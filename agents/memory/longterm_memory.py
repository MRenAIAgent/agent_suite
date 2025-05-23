"""Long-term memory manager for persistent storage.

This module provides a memory manager for long-term persistence
of conversation history and other important information.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from agents.memory.base_memory_manager import MemoryManagerBase
from agents.memory.stores.in_memory_store import InMemoryStore
from agents.memory.stores.store import Store


class LongtermMemory(MemoryManagerBase):
    """Memory manager for long-term persistent storage.
    
    This manager handles persisting conversation history and important
    information across sessions, with capabilities for retrieval
    and search.
    """
    
    def __init__(
        self,
        max_history: int = 100,
        storage_dir: str = "./memory_storage",
        session_id: Optional[str] = None,
        store: Optional[Store] = None
    ):
        """Initialize the long-term memory manager.
        
        Args:
            max_history: Maximum number of history items to keep in memory
            storage_dir: Directory to store persistent memory files
            session_id: Optional session identifier for grouping memories
            store: Optional custom store implementation (defaults to InMemoryStore with file persistence)
        """
        super().__init__(max_history)
        self.storage_dir = storage_dir
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use provided store or create default file-backed store
        self.store = store if store is not None else InMemoryStore()
        self.custom_store = store is not None
        
        # Ensure storage directory exists if using file-based storage
        if not self.custom_store:
            os.makedirs(self.storage_dir, exist_ok=True)
    
    def save_memory(self, message: Dict) -> None:
        """Save a message to both in-memory and persistent storage.
        
        Args:
            message: Message to save
        """
        # Add to store
        self.store.add_history(message)
        
        # For the default store, also persist to disk
        if not self.custom_store:
            self._persist_message(message)
        else:
            # For custom stores like GraphitiStore, ensure data is saved
            self.store.save_history()
        
    def load_memory(self) -> List[Dict]:
        """Load memory from storage.
        
        Returns:
            List of conversation messages
        """
        if self.custom_store:
            # For custom stores, use their native load method
            return self.store.load_history()
        else:
            # For default store, load from disk
            loaded_memory = self._load_from_disk()
            
            # Update in-memory store with loaded memory
            if loaded_memory:
                self.store.clear(history=True)
                for msg in loaded_memory:
                    self.store.add_history(msg)
            
            return self.store.get_history()
    
    def add(self, message: Dict) -> None:
        """Add a message to history and persistent storage.
        
        Args:
            message: Message to add
        """
        # Add timestamp if not present
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
            
        # Add to store
        self.store.add_history(message)
        
        # Persist to permanent storage
        if not self.custom_store:
            self._persist_message(message)
        else:
            # For custom stores like GraphitiStore, ensure data is saved
            self.store.save_history()
        
        # Manage history size in memory
        history = self.store.get_history()
        if len(history) > self.max_history:
            # Create new truncated history (only for in-memory representation)
            new_history = history[len(history) - self.max_history:]
            self.store.clear(history=True)
            for msg in new_history:
                self.store.add_history(msg)
            
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get current conversation history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of conversation messages
        """
        return self.store.get_history(limit=limit)
    
    def clear(self) -> None:
        """Clear conversation history from memory but not from disk if using file-based storage."""
        self.store.clear(history=True)
    
    def purge(self) -> bool:
        """Permanently delete all persistent storage.
        
        Returns:
            True if successful, False otherwise
        """
        if self.custom_store:
            # For custom stores, use their clear method
            try:
                self.store.clear(history=True)
                return True
            except Exception:
                return False
        else:
            # For default file-based persistence
            try:
                session_file = os.path.join(self.storage_dir, f"{self.session_id}.json")
                if os.path.exists(session_file):
                    os.remove(session_file)
                return True
            except Exception:
                return False
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search long-term memory for relevant messages.
        
        For custom stores, this delegates to their native search capabilities.
        For the default store, implements a simple text search.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching messages
        """
        # If the store has a search method (like GraphitiStore), use it
        if hasattr(self.store, "search") and callable(getattr(self.store, "search")):
            return self.store.search(query, limit)
        
        # Otherwise, use default implementation
        # Load all memories from disk
        all_memories = self._load_from_disk()
        
        # Simple search implementation
        results = []
        for memory in all_memories:
            if 'content' in memory and query.lower() in memory['content'].lower():
                results.append(memory)
                if len(results) >= limit:
                    break
        
        return results
    
    def _persist_message(self, message: Dict) -> None:
        """Save a message to persistent storage.
        
        Args:
            message: Message to save
        """
        session_file = os.path.join(self.storage_dir, f"{self.session_id}.json")
        
        # Load existing memories
        existing = []
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, start with an empty list
                existing = []
        
        # Add new memory and save
        existing.append(message)
        with open(session_file, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def _load_from_disk(self) -> List[Dict]:
        """Load all memories from persistent storage.
        
        Returns:
            List of conversation messages
        """
        session_file = os.path.join(self.storage_dir, f"{self.session_id}.json")
        
        if not os.path.exists(session_file):
            return []
        
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [] 