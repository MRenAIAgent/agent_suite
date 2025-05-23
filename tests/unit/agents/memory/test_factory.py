"""Test suite for memory factory functions."""

import pytest
from unittest.mock import patch, MagicMock

from agents.memory.factory import (
    create_memory_manager, 
    create_redis_memory_manager,
    create_context_memory
)
from agents.memory.memory_manager import MemoryManager
from agents.memory.context_memory import ContextMemory
from agents.memory.stores.in_memory_store import InMemoryStore
from agents.memory.stores.redis_store import RedisStore


class TestMemoryFactory:
    """Test suite for memory factory functions."""
    
    def test_create_memory_manager(self):
        """Test creating a basic memory manager."""
        # Create with default params
        memory = create_memory_manager()
        assert isinstance(memory, MemoryManager)
        assert isinstance(memory.context_memory.store, InMemoryStore)
        assert not memory.use_longterm
        
        # Create with custom params
        memory = create_memory_manager(
            max_history=50,
            use_longterm=True,
            storage_dir="./test_storage",
            session_id="test-session"
        )
        assert isinstance(memory, MemoryManager)
        assert memory.max_history == 50
        assert memory.use_longterm
        assert hasattr(memory, 'longterm_memory')
    
    @patch('redis.Redis')
    def test_create_redis_memory_manager(self, mock_redis_class):
        """Test creating a memory manager with Redis backing."""
        # Mock Redis client
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Patch _load_from_redis to avoid Redis operations
        with patch('agents.memory.stores.redis_store.RedisStore._load_from_redis'):
            # Create with default Redis config
            memory = create_redis_memory_manager()
            assert isinstance(memory, MemoryManager)
            assert isinstance(memory.context_memory.store, RedisStore)
            
            # Create with custom Redis config
            memory = create_redis_memory_manager(
                redis_config={
                    'host': 'test-host',
                    'port': 6380,
                    'db': 1,
                    'prefix': 'test:',
                    'expire_seconds': 3600
                }
            )
            assert isinstance(memory, MemoryManager)
            assert isinstance(memory.context_memory.store, RedisStore)
            
            # Verify Redis config was passed to RedisStore
            store = memory.context_memory.store
            assert store.prefix == 'test:'
            assert store.expire_seconds == 3600
            
            # Verify Redis client was initialized with correct params
            mock_redis_class.assert_called_with(
                host='test-host',
                port=6380,
                db=1,
                password=None,
                decode_responses=True
            )
    
    @patch('redis.Redis')
    def test_create_context_memory(self, mock_redis_class):
        """Test creating context memory with different store types."""
        # Mock Redis client
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Test with in-memory store (default)
        with patch('agents.memory.stores.redis_store.RedisStore._load_from_redis'):
            context = create_context_memory(use_redis=False)
            assert isinstance(context, ContextMemory)
            assert isinstance(context.store, InMemoryStore)
            
            # Test with Redis store
            context = create_context_memory(
                use_redis=True,
                redis_config={'prefix': 'test:'}
            )
            assert isinstance(context, ContextMemory)
            assert isinstance(context.store, RedisStore)
            assert context.store.prefix == 'test:'


if __name__ == "__main__":
    pytest.main() 