import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import inspect

from agents.memory.stores.store import Store


class SimpleStore(Store):
    """Simple concrete implementation of Store for testing."""
    
    def add_history(self, message: Dict) -> None:
        """Add a message to conversation history."""
        self.history.append(message)

    def get_history(self, limit: int = -1) -> List:
        """Get conversation history."""
        return self.history[-limit:] if limit > 0 else self.history

    def set_cache(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        self.cache[key] = value

    def fetch_cache(self, key: Optional[str] = None) -> Any:
        """Get value(s) from cache."""
        if key is None:
            return self.cache
        return self.cache.get(key)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []

    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
    
    def clear(self, history: bool = True, cache: bool = True) -> None:
        """Clear specified storage components."""
        if history:
            self.clear_history()
        if cache:
            self.clear_cache()
    
    def save_history(self) -> bool:
        """Save conversation history to persistent storage."""
        return True
        
    def load_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from persistent storage."""
        return self.history
    
    # Implement async methods
    async def async_add_history(self, message: Dict) -> None:
        """Asynchronously add a message to conversation history."""
        self.add_history(message)

    async def async_get_history(self) -> List:
        """Asynchronously get conversation history."""
        return self.history

    async def async_set_cache(self, key: str, value: Any) -> None:
        """Asynchronously set a value in the cache."""
        self.set_cache(key, value)

    async def async_fetch_cache(self, key: Optional[str] = None) -> Any:
        """Asynchronously get value(s) from cache."""
        return self.fetch_cache(key)

    async def async_clear_history(self) -> None:
        """Asynchronously clear conversation history."""
        self.clear_history()

    async def async_clear_cache(self) -> None:
        """Asynchronously clear the cache."""
        self.clear_cache()
        

class TestStore:
    """Test suite for the Store abstract base class."""
    
    def test_init(self):
        """Test initialization."""
        store = SimpleStore()
        assert hasattr(store, 'history')
        assert hasattr(store, 'cache')
        assert store.history == []
        assert store.cache == {}
    
    def test_add_history(self):
        """Test adding history entry."""
        store = SimpleStore()
        message = {"role": "user", "content": "Test"}
        store.add_history(message)
        assert len(store.history) == 1
        assert store.history[0] == message
    
    def test_get_history(self):
        """Test retrieving history."""
        store = SimpleStore()
        for i in range(5):
            store.add_history({"role": "user", "content": f"Test {i}"})
        
        # Get all history
        history = store.get_history()
        assert len(history) == 5
        
        # Get with limit
        history = store.get_history(limit=2)
        assert len(history) == 2
        assert history[0]["content"] == "Test 3"
        assert history[1]["content"] == "Test 4"
    
    def test_set_and_fetch_cache(self):
        """Test setting and fetching cache values."""
        store = SimpleStore()
        
        # Set a value
        store.set_cache("test_key", "test_value")
        
        # Fetch that value
        value = store.fetch_cache("test_key")
        assert value == "test_value"
        
        # Fetch non-existent key
        value = store.fetch_cache("non_existent")
        assert value is None
        
        # Fetch all cache
        store.set_cache("another_key", 123)
        all_cache = store.fetch_cache()
        assert len(all_cache) == 2
        assert all_cache["test_key"] == "test_value"
        assert all_cache["another_key"] == 123
    
    def test_clear_history(self):
        """Test clearing history."""
        store = SimpleStore()
        
        # Add some data
        store.add_history({"role": "user", "content": "Test"})
        store.set_cache("key", "value")
        
        # Clear history
        store.clear_history()
        
        # Check history cleared but cache remains
        assert len(store.history) == 0
        assert store.cache == {"key": "value"}
    
    def test_clear_cache(self):
        """Test clearing cache."""
        store = SimpleStore()
        
        # Add some data
        store.add_history({"role": "user", "content": "Test"})
        store.set_cache("key", "value")
        
        # Clear cache
        store.clear_cache()
        
        # Check cache cleared but history remains
        assert len(store.cache) == 0
        assert len(store.history) == 1
    
    def test_clear(self):
        """Test clearing history and cache."""
        store = SimpleStore()
        
        # Add some data
        store.add_history({"role": "user", "content": "Test"})
        store.set_cache("key", "value")
        
        # Test clearing both
        store.clear(history=True, cache=True)
        assert len(store.history) == 0
        assert len(store.cache) == 0
        
        # Add data again
        store.add_history({"role": "user", "content": "Test"})
        store.set_cache("key", "value")
        
        # Test clearing only history
        store.clear(history=True, cache=False)
        assert len(store.history) == 0
        assert len(store.cache) == 1
        
        # Add history again
        store.add_history({"role": "user", "content": "Test"})
        
        # Test clearing only cache
        store.clear(history=False, cache=True)
        assert len(store.history) == 1
        assert len(store.cache) == 0
    
    def test_save_and_load_history(self):
        """Test saving and loading history."""
        store = SimpleStore()
        
        # Add some data
        store.add_history({"role": "user", "content": "Test"})
        
        # Save
        result = store.save_history()
        assert result is True
        
        # Load
        history = store.load_history()
        assert len(history) == 1
        assert history[0]["content"] == "Test"
    
    def test_abstract_methods(self):
        """Test that Store is properly defined as an abstract class."""
        # Check that Store has abstract methods
        abstract_methods = [name for name, method in inspect.getmembers(Store) 
                           if getattr(method, '__isabstractmethod__', False)]
        
        # Verify we have abstract methods
        assert len(abstract_methods) > 0
        
        # Verify key methods are abstract
        assert 'add_history' in abstract_methods
        assert 'get_history' in abstract_methods
        assert 'set_cache' in abstract_methods
        assert 'fetch_cache' in abstract_methods


if __name__ == "__main__":
    pytest.main() 