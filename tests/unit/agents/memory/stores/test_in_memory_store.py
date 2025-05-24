"""Unit tests for the InMemoryStore class."""

import pytest
import asyncio
from agents.memory.stores.in_memory_store import InMemoryStore


class TestInMemoryStore:
    """Test suite for the InMemoryStore class."""

    def test_init(self):
        """Test initialization of InMemoryStore."""
        store = InMemoryStore()
        assert store.history == []
        assert store.cache == {}

    def test_add_history(self):
        """Test adding a message to history."""
        store = InMemoryStore()
        message = {"role": "user", "content": "Hello"}
        store.add_history(message)
        assert store.history == [message]
        
        # Test adding a message without required fields
        incomplete_message = {"text": "This should not be added"}
        store.add_history(incomplete_message)
        assert len(store.history) == 1  # Should still only have the first message

    def test_get_history(self):
        """Test retrieving history with and without limits."""
        store = InMemoryStore()
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(30)
        ]
        
        for msg in messages:
            store.add_history(msg)
            
        # Test with default limit (20)
        default_limit_history = store.get_history()
        assert len(default_limit_history) == 20
        assert default_limit_history == messages[-20:]
        
        # Test with custom limit
        limited_history = store.get_history(limit=5)
        assert len(limited_history) == 5
        assert limited_history == messages[-5:]
        
        # Test with no limit
        all_history = store.get_history(limit=0)
        assert len(all_history) == 30
        assert all_history == messages

    def test_set_and_fetch_cache(self):
        """Test setting and retrieving cache values."""
        store = InMemoryStore()
        
        # Test setting and getting a single value
        store.set_cache("key1", "value1")
        assert store.fetch_cache("key1") == "value1"
        
        # Test setting and getting multiple values
        store.set_cache("key2", {"nested": "value"})
        store.set_cache("key3", [1, 2, 3])
        
        # Test fetching all cache
        all_cache = store.fetch_cache()
        assert all_cache == {
            "key1": "value1",
            "key2": {"nested": "value"},
            "key3": [1, 2, 3]
        }
        
        # Test fetching non-existent key
        assert store.fetch_cache("nonexistent") is None

    def test_clear(self):
        """Test clearing history and cache."""
        store = InMemoryStore()
        
        # Add some data
        store.add_history({"role": "user", "content": "Test"})
        store.set_cache("key", "value")
        
        assert len(store.history) == 1
        assert len(store.cache) == 1
        
        # Test clearing only history
        store.clear(history=True, cache=False)
        assert len(store.history) == 0
        assert len(store.cache) == 1
        
        # Add history again
        store.add_history({"role": "user", "content": "Test2"})
        
        # Test clearing only cache
        store.clear(history=False, cache=True)
        assert len(store.history) == 1
        assert len(store.cache) == 0
        
        # Add cache again
        store.set_cache("key2", "value2")
        
        # Test clearing both
        store.clear(history=True, cache=True)
        assert len(store.history) == 0
        assert len(store.cache) == 0

    def test_clear_history(self):
        """Test clearing only history."""
        store = InMemoryStore()
        store.add_history({"role": "user", "content": "Test"})
        store.set_cache("key", "value")
        
        store.clear_history()
        assert len(store.history) == 0
        assert len(store.cache) == 1  # Cache should remain

    def test_clear_cache(self):
        """Test clearing only cache."""
        store = InMemoryStore()
        store.add_history({"role": "user", "content": "Test"})
        store.set_cache("key", "value")
        
        store.clear_cache()
        assert len(store.history) == 1  # History should remain
        assert len(store.cache) == 0

    def test_save_and_load_history(self):
        """Test saving and loading history."""
        store = InMemoryStore()
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
        
        for msg in messages:
            store.add_history(msg)
            
        # Since InMemoryStore doesn't actually persist data,
        # save_history should just return True
        assert store.save_history() is True
        
        # load_history should return the current history
        loaded = store.load_history()
        assert loaded == messages

    @pytest.mark.asyncio
    async def test_async_add_history(self):
        """Test async adding of message to history."""
        store = InMemoryStore()
        message = {"role": "user", "content": "Hello"}
        await store.async_add_history(message)
        assert store.history == [message]

    @pytest.mark.asyncio
    async def test_async_get_history(self):
        """Test async retrieval of history."""
        store = InMemoryStore()
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
        
        for msg in messages:
            store.add_history(msg)
            
        history = await store.async_get_history()
        assert history == messages

    @pytest.mark.asyncio
    async def test_async_set_and_fetch_cache(self):
        """Test async setting and retrieving cache values."""
        store = InMemoryStore()
        
        # Test setting and getting a single value
        await store.async_set_cache("key1", "value1")
        result = await store.async_fetch_cache("key1")
        assert result == "value1"
        
        # Test setting and getting multiple values
        await store.async_set_cache("key2", {"nested": "value"})
        
        # Test fetching all cache
        all_cache = await store.async_fetch_cache()
        assert all_cache == {
            "key1": "value1",
            "key2": {"nested": "value"}
        }

    @pytest.mark.asyncio
    async def test_async_clear_history(self):
        """Test async clearing of history."""
        store = InMemoryStore()
        store.add_history({"role": "user", "content": "Test"})
        store.set_cache("key", "value")
        
        await store.async_clear_history()
        assert len(store.history) == 0
        assert len(store.cache) == 1  # Cache should remain

    @pytest.mark.asyncio
    async def test_async_clear_cache(self):
        """Test async clearing of cache."""
        store = InMemoryStore()
        store.add_history({"role": "user", "content": "Test"})
        store.set_cache("key", "value")
        
        await store.async_clear_cache()
        assert len(store.history) == 1  # History should remain
        assert len(store.cache) == 0 