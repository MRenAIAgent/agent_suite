import pytest
import json
from unittest.mock import MagicMock, patch

from agents.memory.stores.redis_store import RedisStore


class TestRedisStore:
    """Test suite for the RedisStore class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.get.return_value = None  # Default return value for get
        return mock
    
    @patch('redis.Redis')
    def test_init(self, mock_redis_class):
        """Test initialization with default parameters."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid loading during init
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore()
            
            # Check Redis initialization
            mock_redis_class.assert_called_once_with(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True
            )
            
            # Check default values
            assert store.prefix == "agent:"
            assert store.expire_seconds is None
            assert len(store.history) == 0
            assert len(store.cache) == 0
    
    @patch('redis.Redis')
    def test_init_with_custom_params(self, mock_redis_class):
        """Test initialization with custom parameters."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid loading during init
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore(
                host="redis.example.com",
                port=6380,
                db=1,
                prefix="test:",
                expire_seconds=3600
            )
            
            # Check Redis initialization
            mock_redis_class.assert_called_once_with(
                host="redis.example.com",
                port=6380,
                db=1,
                decode_responses=True
            )
            
            # Check custom values
            assert store.prefix == "test:"
            assert store.expire_seconds == 3600
    
    @patch('redis.Redis')
    def test_load_data(self, mock_redis_class):
        """Test loading data from Redis."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Setup mock to return history data
        history_data = [
            {"role": "user", "content": "Test 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
        mock_client.get.return_value = json.dumps(history_data)
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore(prefix="test:")
            
            # Force reload data
            store._load_from_redis()
            
            # Verify history loaded
            assert len(store.history) == 2
            assert store.history[0]["role"] == "user"
            assert store.history[1]["role"] == "assistant"
    
    @patch('redis.Redis')
    def test_add_history(self, mock_redis_class):
        """Test adding a message to history."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore()
            message = {"role": "user", "content": "Test message"}
            
            # Initially empty history
            mock_client.get.return_value = None
            
            # Add message
            store.add_history(message)
            
            # Verify message added to local history
            assert len(store.history) == 1
            assert store.history[0] == message
            
            # Verify history saved to Redis
            mock_client.set.assert_called_once()
            
            # Check what was saved
            args, kwargs = mock_client.set.call_args
            key, value = args
            assert key == "agent:history"
            saved_data = json.loads(value)
            assert len(saved_data) == 1
            assert saved_data[0]["role"] == "user"
            assert saved_data[0]["content"] == "Test message"
    
    @patch('redis.Redis')
    def test_get_history(self, mock_redis_class):
        """Test getting history with different limits."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore()
            
            # Add messages directly to history
            for i in range(10):
                store.history.append({"role": "user", "content": f"Message {i}"})
            
            # Test default (all)
            history = store.get_history()
            assert len(history) == 10
            
            # Test with limit
            history = store.get_history(limit=5)
            assert len(history) == 5
            assert history[0]["content"] == "Message 5"
            assert history[4]["content"] == "Message 9"
            
            # Test with no limit
            history = store.get_history(limit=-1)
            assert len(history) == 10
    
    @patch('redis.Redis')
    def test_set_cache(self, mock_redis_class):
        """Test setting a cache value."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore(prefix="test:")
            
            # Test simple value
            store.set_cache("key1", "value1")
            
            # Verify local cache updated
            assert store.cache["key1"] == "value1"
            
            # Verify Redis updated
            mock_client.set.assert_called_once()
            
            args, kwargs = mock_client.set.call_args
            key, value = args
            assert key == "test:cache"
            assert json.loads(value)["key1"] == "value1"
    
    @patch('redis.Redis')
    def test_fetch_cache(self, mock_redis_class):
        """Test fetching cache values."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore()
            
            # Setup cache
            store.cache = {
                "key1": "value1",
                "key2": 42,
                "key3": {"data": [1, 2, 3]}
            }
            
            # Test getting specific keys
            assert store.fetch_cache("key1") == "value1"
            assert store.fetch_cache("key2") == 42
            assert store.fetch_cache("key3") == {"data": [1, 2, 3]}
            
            # Test getting non-existent key
            assert store.fetch_cache("nonexistent") is None
            
            # Test getting all cache
            all_cache = store.fetch_cache()
            assert len(all_cache) == 3
            assert all_cache["key1"] == "value1"
            assert all_cache["key2"] == 42
            assert all_cache["key3"] == {"data": [1, 2, 3]}
    
    @patch('redis.Redis')
    def test_clear(self, mock_redis_class):
        """Test clearing history and cache."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore(prefix="test:")
            
            # Setup data
            store.history = [{"role": "user", "content": "Test"}]
            store.cache = {"key": "value"}
            
            # Test clearing both
            store.clear()
            
            # Verify local data cleared
            assert store.history == []
            assert store.cache == {}
            
            # Verify Redis operations
            assert mock_client.delete.call_count >= 2
    
    @patch('redis.Redis')
    def test_clear_history_only(self, mock_redis_class):
        """Test clearing only history."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore(prefix="test:")
            
            # Setup data
            store.history = [{"role": "user", "content": "Test"}]
            store.cache = {"key": "value"}
            
            # Test clearing only history
            store.clear(history=True, cache=False)
            
            # Verify only history cleared
            assert store.history == []
            assert store.cache == {"key": "value"}
            
            # Verify Redis operations
            mock_client.delete.assert_called_once_with("test:history")
    
    @patch('redis.Redis')
    def test_clear_cache_only(self, mock_redis_class):
        """Test clearing only cache."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore(prefix="test:")
            
            # Setup data
            store.history = [{"role": "user", "content": "Test"}]
            store.cache = {"key": "value"}
            
            # Test clearing only cache
            store.clear(history=False, cache=True)
            
            # Verify only cache cleared
            assert store.history == [{"role": "user", "content": "Test"}]
            assert store.cache == {}
            
            # Verify Redis operations
            mock_client.delete.assert_called_once_with("test:cache")
    
    @patch('redis.Redis')
    def test_clear_history_shortcut(self, mock_redis_class):
        """Test clear_history shortcut method."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore()
            
            # Setup data
            store.history = [{"role": "user", "content": "Test"}]
            store.cache = {"key": "value"}
            
            # Clear history
            store.clear_history()
            
            # Verify only history cleared
            assert store.history == []
            assert store.cache == {"key": "value"}
            
            # Verify Redis operations
            mock_client.delete.assert_called_once_with("agent:history")
    
    @patch('redis.Redis')
    def test_clear_cache_shortcut(self, mock_redis_class):
        """Test clear_cache shortcut method."""
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid auto loading
        with patch.object(RedisStore, '_load_from_redis'):
            store = RedisStore(prefix="test:")
            
            # Setup data
            store.history = [{"role": "user", "content": "Test"}]
            store.cache = {"key1": "value1", "key2": "value2"}
            
            # Clear cache
            store.clear_cache()
            
            # Verify only cache cleared
            assert store.history == [{"role": "user", "content": "Test"}]
            assert store.cache == {}
            
            # Verify Redis operations
            mock_client.delete.assert_called_once_with("test:cache")


if __name__ == "__main__":
    pytest.main() 