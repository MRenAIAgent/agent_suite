import pytest
import os
import json
import shutil
from unittest.mock import patch, MagicMock

from agents.memory.context_memory import ContextMemory
from agents.memory.longterm_memory import LongtermMemory
from agents.memory.memory_manager import MemoryManager
from agents.memory.factory import create_redis_memory_manager
from agents.memory.stores.redis_store import RedisStore


class TestMemoryIntegration:
    """Integration tests for memory components working together."""
    
    @pytest.fixture
    def test_dir(self):
        """Create a temporary test directory for longterm storage."""
        test_dir = "./tests/test_memory_storage"
        os.makedirs(test_dir, exist_ok=True)
        yield test_dir
        shutil.rmtree(test_dir)
    
    def test_context_to_memory_manager(self):
        """Test using ContextMemory with MemoryManager."""
        # Create a context memory
        context = ContextMemory(max_history=5)
        
        # Add messages to context
        context.add({"role": "user", "content": "Hello"})
        context.add({"role": "assistant", "content": "Hi there"})
        
        # Create memory manager using same max_history
        manager = MemoryManager(max_history=5)
        
        # Use context memory's store in manager's context memory
        manager.context_memory.store = context.store
        
        # Verify manager can access context memory's history
        history = manager.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_longterm_persistence(self, test_dir):
        """Test longterm memory persists messages."""
        # Create longterm memory
        longterm = LongtermMemory(storage_dir=test_dir)
        
        # Add messages
        longterm.add({"role": "user", "content": "Test message 1"})
        longterm.add({"role": "assistant", "content": "Test response 1"})
        
        # Create a new instance pointing to same directory
        longterm2 = LongtermMemory(storage_dir=test_dir)
        
        # Verify it loads the history
        history = longterm2.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_memory_manager_with_longterm(self, test_dir):
        """Test memory manager with longterm storage."""
        # Create memory manager with longterm
        manager = MemoryManager(
            max_history=5,
            use_longterm=True,
            storage_dir=test_dir
        )
        
        # Add some messages
        manager.add({"role": "user", "content": "Hello from manager"})
        manager.add({"role": "assistant", "content": "Hello back from manager"})
        
        # Create new manager pointing to same directory
        manager2 = MemoryManager(
            max_history=5,
            use_longterm=True,
            storage_dir=test_dir
        )
        
        # Load history
        history = manager2.load_memory()
        
        # Verify history loaded
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_context_memory_formatting(self):
        """Test different formatting options for context memory."""
        context = ContextMemory()
        
        # Add messages
        context.add({"role": "user", "content": "Tell me a joke"})
        context.add({"role": "assistant", "content": "Why did the chicken cross the road?"})
        context.add({"role": "user", "content": "Why?"})
        context.add({"role": "assistant", "content": "To get to the other side!"})
        
        # Test default formatting
        default_format = context.get_formatted_history()
        assert "USER: Tell me a joke" in default_format
        assert "ASSISTANT: Why did the chicken cross the road?" in default_format
        
        # Test compact formatting
        compact_format = context.get_formatted_history("compact")
        assert "U: Tell me a joke" in compact_format
        assert "A: Why did the chicken cross the road?" in compact_format
    
    def test_context_history_truncation(self):
        """Test history truncation in context memory."""
        context = ContextMemory(max_history=3)
        
        # Add more messages than max_history
        for i in range(5):
            context.add({"role": "user", "content": f"Message {i}"})
        
        # Verify only max_history messages are kept
        history = context.get_history()
        assert len(history) == 3
        assert history[0]["content"] == "Message 2"
        assert history[2]["content"] == "Message 4"
    
    def test_search_across_components(self, test_dir):
        """Test search functionality across memory components."""
        manager = MemoryManager(
            max_history=5,
            use_longterm=True,
            storage_dir=test_dir
        )
        
        # Add messages to both memories
        # Recent messages in context memory
        manager.add({"role": "user", "content": "What is Python?"})
        manager.add({"role": "assistant", "content": "Python is a programming language."})
        
        # Add older messages directly to longterm memory (simulating history)
        manager.longterm_memory.add({"role": "user", "content": "Tell me about Python libraries"})
        manager.longterm_memory.add({"role": "assistant", "content": "There are many Python libraries like NumPy."})
        
        # Search for "Python" in both memories
        results = manager.search("Python")
        
        # Should find results from both memories
        assert len(results) >= 4
        
        # Verify content exists in results
        contents = [msg["content"] for msg in results]
        assert any("programming language" in content for content in contents)
        assert any("libraries" in content for content in contents)
    
    @pytest.mark.skip(reason="Requires Redis server")
    def test_redis_store_integration(self):
        """Test Redis store integration (requires Redis server)."""
        # Create a Redis-backed memory manager
        manager = create_redis_memory_manager(
            max_history=10,
            redis_config={
                'prefix': 'test:',
                'expire_seconds': 3600
            }
        )
        
        # Add messages
        manager.add({"role": "user", "content": "Redis test message"})
        manager.add({"role": "assistant", "content": "Redis test response"})
        
        # Verify messages are in history
        history = manager.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Redis test message"
        
        # Create a new manager with same Redis config
        manager2 = create_redis_memory_manager(
            max_history=10,
            redis_config={
                'prefix': 'test:',
                'expire_seconds': 3600
            }
        )
        
        # Load history
        history2 = manager2.get_history()
        
        # Verify history loaded from Redis
        assert len(history2) == 2
        assert history2[0]["role"] == "user"
        assert history2[0]["content"] == "Redis test message"
        
        # Clean up
        manager2.clear()
    
    @patch('redis.Redis')
    def test_redis_store_mock_integration(self, mock_redis_class):
        """Test Redis store integration with mocked Redis."""
        # Mock Redis client
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        
        # Setup mock to return history data
        history_data = [
            {"role": "user", "content": "Mock Redis test"},
            {"role": "assistant", "content": "Mock Redis response"}
        ]
        
        # Mock get method to return our test data
        def mock_get(key):
            if key == "test:history":
                return json.dumps(history_data)
            return None
            
        mock_client.get.side_effect = mock_get
        
        # Create memory manager with mocked Redis
        with patch('agents.memory.stores.redis_store.RedisStore._load_from_redis'):
            manager = create_redis_memory_manager(
                redis_config={'prefix': 'test:'}
            )
            
            # Replace the _load_from_redis method to use our mock data
            manager.context_memory.store.history = history_data.copy()
            
            # Test getting history
            history = manager.get_history()
            assert len(history) == 2
            assert history[0]["content"] == "Mock Redis test"
            
            # Test adding a message
            manager.add({"role": "user", "content": "Another test"})
            
            # Verify message was added
            history = manager.get_history()
            assert len(history) == 3
            assert history[2]["content"] == "Another test"
            
            # Verify Redis set was called
            assert mock_client.set.call_count > 0 