import pytest
from typing import Dict, Any

from agents.memory.stores.in_memory_store import InMemoryStore


class TestInMemoryStore:
    """Test suite for the InMemoryStore class."""
    
    def test_init(self):
        """Test initialization."""
        store = InMemoryStore()
        assert len(store.history) == 0
        assert len(store.cache) == 0
    
    def test_add_history(self):
        """Test adding items to history."""
        store = InMemoryStore()
        
        # Valid message
        message1 = {"role": "user", "content": "Hello"}
        store.add_history(message1)
        assert len(store.history) == 1
        assert store.history[0] == message1
        
        # Invalid message (missing content) - should NOT be added
        message2 = {"role": "system"}
        store.add_history(message2)
        assert len(store.history) == 1  # Still only 1 message
        
        # Invalid message (missing role) - should NOT be added
        message3 = {"content": "Test content"}
        store.add_history(message3)
        assert len(store.history) == 1  # Still only 1 message
        
        # Another valid message
        message4 = {"role": "assistant", "content": "I'm here to help"}
        store.add_history(message4)
        assert len(store.history) == 2
        assert store.history[1] == message4
    
    def test_get_history(self):
        """Test retrieving history with different limits."""
        store = InMemoryStore()
        
        # Add multiple messages
        for i in range(10):
            store.add_history({"role": "user", "content": f"Message {i}"})
        
        # Get all history
        history = store.get_history()
        assert len(history) == 10
        
        # Get limited history
        history = store.get_history(limit=5)
        assert len(history) == 5
        assert history[0]["content"] == "Message 5"
        assert history[4]["content"] == "Message 9"
        
        # Get history with invalid limit (should return all)
        history = store.get_history(limit=0)
        assert len(history) == 10
    
    def test_set_and_fetch_cache(self):
        """Test setting and retrieving cache items."""
        store = InMemoryStore()
        
        # Set and get simple values
        store.set_cache("string", "value")
        store.set_cache("number", 42)
        store.set_cache("boolean", True)
        
        assert store.fetch_cache("string") == "value"
        assert store.fetch_cache("number") == 42
        assert store.fetch_cache("boolean") is True
        
        # Set and get complex values
        complex_obj = {"name": "test", "data": [1, 2, 3]}
        store.set_cache("complex", complex_obj)
        
        result = store.fetch_cache("complex")
        assert result["name"] == "test"
        assert result["data"] == [1, 2, 3]
        
        # Get non-existent key
        assert store.fetch_cache("nonexistent") is None
        
        # Get all cache
        all_cache = store.fetch_cache()
        assert len(all_cache) == 4
        assert "string" in all_cache
        assert "complex" in all_cache
    
    def test_clear(self):
        """Test clearing history and cache."""
        store = InMemoryStore()
        
        # Add data
        store.add_history({"role": "user", "content": "Test message"})
        store.set_cache("key", "value")
        
        assert len(store.history) == 1
        assert len(store.cache) == 1
        
        # Clear just history
        store.clear(history=True, cache=False)
        assert len(store.history) == 0
        assert len(store.cache) == 1
        
        # Add history again
        store.add_history({"role": "user", "content": "New message"})
        
        # Clear just cache
        store.clear(history=False, cache=True)
        assert len(store.history) == 1
        assert len(store.cache) == 0
        
        # Add cache again
        store.set_cache("new_key", "new_value")
        
        # Clear both
        store.clear(history=True, cache=True)
        assert len(store.history) == 0
        assert len(store.cache) == 0
    
    def test_clear_history(self):
        """Test the clear_history shortcut method."""
        store = InMemoryStore()
        
        # Add data
        store.add_history({"role": "user", "content": "Test message"})
        store.set_cache("key", "value")
        
        # Clear history
        store.clear_history()
        assert len(store.history) == 0
        assert len(store.cache) == 1  # Cache should be untouched
    
    def test_clear_cache(self):
        """Test the clear_cache shortcut method."""
        store = InMemoryStore()
        
        # Add data
        store.add_history({"role": "user", "content": "Test message"})
        store.set_cache("key", "value")
        
        # Clear cache
        store.clear_cache()
        assert len(store.cache) == 0
        assert len(store.history) == 1  # History should be untouched


if __name__ == "__main__":
    pytest.main() 