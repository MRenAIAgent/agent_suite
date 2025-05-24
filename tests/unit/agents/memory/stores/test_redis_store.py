"""Unit tests for the RedisStore class."""

import json
import pytest
from unittest.mock import patch, MagicMock
from agents.memory.stores.redis_store import RedisStore


class TestRedisStore:
    """Test suite for the RedisStore class."""

    @patch('redis.Redis')
    def test_init(self, mock_redis_class):
        """Test initialization of RedisStore."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client

        # Test default params
        store = RedisStore()
        assert store.prefix == "agent:"
        assert store.history_key == "agent:history"
        assert store.cache_key == "agent:cache"
        assert store.expire_seconds is None
        
        # Verify Redis was initialized with correct params
        mock_redis_class.assert_called_with(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            decode_responses=True
        )
        
        # Test custom params
        store = RedisStore(
            host="redis.example.com",
            port=6380,
            db=1,
            prefix="test:",
            password="secret",
            expire_seconds=3600
        )
        
        assert store.prefix == "test:"
        assert store.history_key == "test:history"
        assert store.cache_key == "test:cache"
        assert store.expire_seconds == 3600
        
        # Verify Redis was initialized with custom params
        mock_redis_class.assert_called_with(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            decode_responses=True
        )

    @patch('redis.Redis')
    def test_add_history(self, mock_redis_class):
        """Test adding a message to history."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client
        
        store = RedisStore()
        message = {"role": "user", "content": "Hello"}
        
        # Add message
        store.add_history(message)
        
        # Verify message was added to in-memory history
        assert store.history == [message]
        
        # Verify Redis set was called with serialized history
        expected_serialized = json.dumps([message])
        mock_client.set.assert_called_with(store.history_key, expected_serialized)
        
        # Test adding a message without required fields
        incomplete_message = {"text": "This should not be added"}
        store.add_history(incomplete_message)
        
        # History should remain unchanged
        assert store.history == [message]

    @patch('redis.Redis')
    def test_get_history(self, mock_redis_class):
        """Test retrieving history with and without limits."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client
        
        store = RedisStore()
        
        # Generate test messages
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(30)
        ]
        
        # Add messages to history without triggering Redis operations
        store.history = messages.copy()
        
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

    @patch('redis.Redis')
    def test_set_and_fetch_cache(self, mock_redis_class):
        """Test setting and retrieving cache values."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client
        
        store = RedisStore()
        
        # Test setting and getting a single value
        store.set_cache("key1", "value1")
        assert store.fetch_cache("key1") == "value1"
        
        # Verify Redis set was called with serialized cache
        expected_serialized = json.dumps({"key1": "value1"})
        mock_client.set.assert_called_with(store.cache_key, expected_serialized)
        
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

    @patch('redis.Redis')
    def test_clear(self, mock_redis_class):
        """Test clearing history and cache."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client
        
        store = RedisStore()
        
        # Add test data
        store.history = [{"role": "user", "content": "Test"}]
        store.cache = {"key": "value"}
        
        # Test clearing only history
        store.clear(history=True, cache=False)
        
        # Verify in-memory history was cleared but cache remains
        assert store.history == []
        assert store.cache == {"key": "value"}
        
        # Verify Redis delete was called for history key only
        mock_client.delete.assert_called_with(store.history_key)
        
        # Reset mock and test data
        mock_client.reset_mock()
        store.history = [{"role": "user", "content": "Test"}]
        
        # Test clearing only cache
        store.clear(history=False, cache=True)
        
        # Verify in-memory cache was cleared but history remains
        assert store.history == [{"role": "user", "content": "Test"}]
        assert store.cache == {}
        
        # Verify Redis delete was called for cache key only
        mock_client.delete.assert_called_with(store.cache_key)
        
        # Reset mock and test data
        mock_client.reset_mock()
        store.history = [{"role": "user", "content": "Test"}]
        store.cache = {"key": "value"}
        
        # Test clearing both
        store.clear(history=True, cache=True)
        
        # Verify both in-memory history and cache were cleared
        assert store.history == []
        assert store.cache == {}
        
        # Verify Redis delete was called for both keys
        assert mock_client.delete.call_count == 2

    @patch('redis.Redis')
    def test_clear_history(self, mock_redis_class):
        """Test clearing only history."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client
        
        store = RedisStore()
        store.history = [{"role": "user", "content": "Test"}]
        store.cache = {"key": "value"}
        
        store.clear_history()
        
        # Verify in-memory history was cleared but cache remains
        assert store.history == []
        assert store.cache == {"key": "value"}
        
        # Verify Redis delete was called for history key
        mock_client.delete.assert_called_with(store.history_key)

    @patch('redis.Redis')
    def test_clear_cache(self, mock_redis_class):
        """Test clearing only cache."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client
        
        store = RedisStore()
        store.history = [{"role": "user", "content": "Test"}]
        store.cache = {"key": "value"}
        
        store.clear_cache()
        
        # Verify in-memory cache was cleared but history remains
        assert store.history == [{"role": "user", "content": "Test"}]
        assert store.cache == {}
        
        # Verify Redis delete was called for cache key
        mock_client.delete.assert_called_with(store.cache_key)

    @patch('redis.Redis')
    def test_save_and_load_history(self, mock_redis_class):
        """Test saving and loading history."""
        # Setup mock
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Setup mock history data in Redis
        test_history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
        serialized_history = json.dumps(test_history)
        mock_client.get.side_effect = lambda key: serialized_history if key == "agent:history" else None
        
        # Create store and verify history is loaded
        store = RedisStore()
        
        # Reset mock and test data
        mock_client.reset_mock()
        store.history = []
        
        # Test loading history
        loaded = store.load_history()
        assert loaded == test_history
        
        # Verify Redis get was called for history key
        mock_client.get.assert_called_with(store.history_key)
        
        # Test saving history
        store.history = test_history.copy()
        result = store.save_history()
        
        # Verify Redis set was called with serialized history
        assert result is True
        mock_client.set.assert_called_with(store.history_key, serialized_history)

    @patch('redis.Redis')
    def test_expiration(self, mock_redis_class):
        """Test expiration settings for Redis keys."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client
        
        # Create store with expiration
        store = RedisStore(expire_seconds=3600)
        
        # Test adding history
        store.add_history({"role": "user", "content": "Hello"})
        
        # Verify expiration was set
        assert mock_client.expire.call_count == 1
        mock_client.expire.assert_called_with(store.history_key, 3600)
        
        # Reset mock
        mock_client.reset_mock()
        
        # Test setting cache
        store.set_cache("key", "value")
        
        # Verify expiration was set
        assert mock_client.expire.call_count == 1
        mock_client.expire.assert_called_with(store.cache_key, 3600)

    @patch('redis.Redis')
    def test_redis_error_handling(self, mock_redis_class):
        """Test handling Redis errors."""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_client.set.side_effect = Exception("Redis connection error")
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client
        
        store = RedisStore()
        
        # Test saving history with Redis error
        store.history = [{"role": "user", "content": "Test"}]
        result = store._save_history_to_redis()
        
        # Should return False on error
        assert result is False
        
        # Test saving cache with Redis error
        store.cache = {"key": "value"}
        result = store._save_cache_to_redis()
        
        # Should return False on error
        assert result is False

    @patch('redis.Redis')
    def test_json_error_handling(self, mock_redis_class):
        """Test handling JSON decode errors."""
        # Setup mock with invalid JSON data
        mock_client = MagicMock()
        mock_client.get.side_effect = lambda key: "invalid json" if key == "agent:history" else "{invalid json" if key == "agent:cache" else None
        mock_redis_class.return_value = mock_client
        
        # Create store and verify errors are handled gracefully
        store = RedisStore()
        
        # History and cache should be empty on JSON errors
        assert store.history == []
        assert store.cache == {}

    @pytest.mark.asyncio
    @patch('redis.Redis')
    async def test_async_methods(self, mock_redis_class):
        """Test asynchronous methods."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client
        
        store = RedisStore()
        
        # Test async add history
        await store.async_add_history({"role": "user", "content": "Hello"})
        assert len(store.history) == 1
        
        # Test async set cache
        await store.async_set_cache("key", "value")
        assert store.cache["key"] == "value"
        
        # Test async get history
        history = await store.async_get_history()
        assert history == store.history
        
        # Test async fetch cache
        cache_value = await store.async_fetch_cache("key")
        assert cache_value == "value"
        
        # Test async clear methods
        await store.async_clear_history()
        assert store.history == []
        assert len(store.cache) == 1
        
        await store.async_clear_cache()
        assert store.cache == {} 