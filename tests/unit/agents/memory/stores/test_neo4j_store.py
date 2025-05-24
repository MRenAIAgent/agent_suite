"""Unit tests for the Neo4jStore class."""

import json
import pytest
from unittest.mock import patch, MagicMock, call
from agents.memory.stores.neo4j_store import Neo4jStore


class TestNeo4jStore:
    """Test suite for the Neo4jStore class."""

    @patch('neo4j.GraphDatabase.driver')
    def test_init(self, mock_driver_class):
        """Test initialization of Neo4jStore."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Test default params
        store = Neo4jStore(
            uri="neo4j+s://example.databases.neo4j.io",
            username="neo4j",
            password="password"
        )
        
        assert store.prefix == "agent:"
        assert store.database == "neo4j"
        
        # Verify driver was initialized with correct params
        mock_driver_class.assert_called_with(
            "neo4j+s://example.databases.neo4j.io",
            auth=("neo4j", "password")
        )
        
        # Test custom params
        store = Neo4jStore(
            uri="neo4j+s://custom.databases.neo4j.io",
            username="custom_user",
            password="custom_pass",
            database="custom_db",
            prefix="test:"
        )
        
        assert store.prefix == "test:"
        assert store.database == "custom_db"
        
        # Verify driver was initialized with custom params
        mock_driver_class.assert_called_with(
            "neo4j+s://custom.databases.neo4j.io",
            auth=("custom_user", "custom_pass")
        )

    @patch('neo4j.GraphDatabase.driver')
    def test_initialize_schema(self, mock_driver_class):
        """Test schema initialization error handling."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Test 1: Verify _initialize_schema is called during initialization
        with patch.object(Neo4jStore, '_initialize_schema') as mock_init_schema, \
             patch.object(Neo4jStore, '_load_from_neo4j'):
            # Create a store instance
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            # Verify the method was called during initialization
            mock_init_schema.assert_called_once()
        
        # Test 2: Verify exception handling
        # First, create a session that throws an exception when execute_write is called
        mock_session.execute_write = MagicMock(side_effect=Exception("Test constraint error"))
        
        # Then create a store, which should not crash despite the exception
        with patch.object(Neo4jStore, '_load_from_neo4j'):
            # This should not raise an exception
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            # If we reach here, the exception was properly caught

    @patch('neo4j.GraphDatabase.driver')
    def test_add_history(self, mock_driver_class):
        """Test adding a message to history."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_tx = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_session.execute_write.return_value = None
        mock_driver_class.return_value = mock_driver
        
        # Patch the init methods to avoid issues
        with patch('agents.memory.stores.neo4j_store.Neo4jStore._initialize_schema'), \
             patch('agents.memory.stores.neo4j_store.Neo4jStore._load_from_neo4j'), \
             patch('agents.memory.stores.neo4j_store.Neo4jStore._save_history_to_neo4j'):
            
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            # Clear history for clean test
            store.history = []
            
            message = {"role": "user", "content": "Hello"}
            
            # Add message
            store.add_history(message)
            
            # Verify message was added to in-memory history
            assert store.history == [message]
            
            # Test adding a message without required fields
            incomplete_message = {"text": "This should not be added"}
            store.add_history(incomplete_message)
            
            # History should remain unchanged
            assert store.history == [message]

    @patch('neo4j.GraphDatabase.driver')
    def test_get_history(self, mock_driver_class):
        """Test retrieving history with and without limits."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Patch the init methods to avoid issues
        with patch('agents.memory.stores.neo4j_store.Neo4jStore._initialize_schema'), \
             patch('agents.memory.stores.neo4j_store.Neo4jStore._load_from_neo4j'):
            
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            # Generate test messages
            messages = [
                {"role": "user", "content": f"Message {i}"}
                for i in range(30)
            ]
            
            # Add messages to history
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

    @patch('neo4j.GraphDatabase.driver')
    def test_set_and_fetch_cache(self, mock_driver_class):
        """Test setting and retrieving cache values."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Patch the init methods to avoid issues
        with patch('agents.memory.stores.neo4j_store.Neo4jStore._initialize_schema'), \
             patch('agents.memory.stores.neo4j_store.Neo4jStore._load_from_neo4j'), \
             patch('agents.memory.stores.neo4j_store.Neo4jStore._save_cache_to_neo4j'):
            
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            # Clear cache for clean test
            store.cache = {}
            
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

    @patch('neo4j.GraphDatabase.driver')
    def test_clear(self, mock_driver_class):
        """Test clearing history and cache."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Patch the init methods to avoid issues
        with patch('agents.memory.stores.neo4j_store.Neo4jStore._initialize_schema'), \
             patch('agents.memory.stores.neo4j_store.Neo4jStore._load_from_neo4j'):
            
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            # Add test data
            store.history = [{"role": "user", "content": "Test"}]
            store.cache = {"key": "value"}
            
            # Test clearing only history
            store.clear(history=True, cache=False)
            
            # Verify in-memory history was cleared but cache remains
            assert store.history == []
            assert store.cache == {"key": "value"}
            
            # Add history again
            store.history = [{"role": "user", "content": "Test"}]
            
            # Test clearing only cache
            store.clear(history=False, cache=True)
            
            # Verify in-memory cache was cleared but history remains
            assert store.history == [{"role": "user", "content": "Test"}]
            assert store.cache == {}
            
            # Add cache again
            store.cache = {"key": "value"}
            
            # Test clearing both
            store.clear(history=True, cache=True)
            
            # Verify both in-memory history and cache were cleared
            assert store.history == []
            assert store.cache == {}

    @patch('neo4j.GraphDatabase.driver')
    def test_clear_history(self, mock_driver_class):
        """Test clearing only history."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Patch the init methods to avoid issues
        with patch('agents.memory.stores.neo4j_store.Neo4jStore._initialize_schema'), \
             patch('agents.memory.stores.neo4j_store.Neo4jStore._load_from_neo4j'):
            
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            store.history = [{"role": "user", "content": "Test"}]
            store.cache = {"key": "value"}
            
            store.clear_history()
            
            # Verify in-memory history was cleared but cache remains
            assert store.history == []
            assert store.cache == {"key": "value"}

    @patch('neo4j.GraphDatabase.driver')
    def test_clear_cache(self, mock_driver_class):
        """Test clearing only cache."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Patch the init methods to avoid issues
        with patch('agents.memory.stores.neo4j_store.Neo4jStore._initialize_schema'), \
             patch('agents.memory.stores.neo4j_store.Neo4jStore._load_from_neo4j'):
            
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            store.history = [{"role": "user", "content": "Test"}]
            store.cache = {"key": "value"}
            
            store.clear_cache()
            
            # Verify in-memory cache was cleared but history remains
            assert store.history == [{"role": "user", "content": "Test"}]
            assert store.cache == {}

    @patch('neo4j.GraphDatabase.driver')
    def test_save_and_load_history(self, mock_driver_class):
        """Test saving and loading history."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Setup mock history data in Neo4j
        test_history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
        serialized_history = json.dumps(test_history)
        
        # Patch the init methods to avoid issues
        with patch.object(Neo4jStore, '_initialize_schema'), \
             patch.object(Neo4jStore, '_load_from_neo4j'):
            
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            # Mock direct load_history method
            with patch.object(store, '_load_from_neo4j') as mock_load:
                def side_effect(*args, **kwargs):
                    store.history = test_history
                
                mock_load.side_effect = side_effect
                
                # Test loading history
                loaded_history = store.load_history()
                assert loaded_history == test_history
                
                # Verify _load_from_neo4j was called
                mock_load.assert_called_once()
            
            # Now reset history for save test
            store.history = test_history.copy()
            
            # Test successful save
            with patch.object(store, '_save_history_to_neo4j') as mock_save:
                mock_save.return_value = True
                result = store.save_history()
                assert result is True
                mock_save.assert_called_once()
            
            # Test failed save
            with patch.object(store, '_save_history_to_neo4j') as mock_save:
                mock_save.return_value = False
                result = store.save_history()
                assert result is False
                mock_save.assert_called_once()

    @patch('neo4j.GraphDatabase.driver')
    def test_json_error_handling(self, mock_driver_class):
        """Test handling JSON decode errors."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Patch the init methods to avoid issues
        with patch.object(Neo4jStore, '_initialize_schema'):
            
            # Configure mock for loading invalid JSON
            record_h = MagicMock()
            record_h.__getitem__.return_value = "invalid json"
            record_c = MagicMock()
            record_c.__getitem__.return_value = "{invalid json"
            
            mock_result_h = MagicMock()
            mock_result_h.single.return_value = record_h
            
            mock_result_c = MagicMock()
            mock_result_c.single.return_value = record_c
            
            mock_session.execute_read.side_effect = [mock_result_h, mock_result_c]
            
            # Create store and verify errors are handled gracefully
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            # Reset history and cache
            store.history = []
            store.cache = {}
            
            # Call load explicitly to trigger JSON decode error handling
            store._load_from_neo4j()
            
            # History and cache should be empty on JSON errors
            assert store.history == []
            assert store.cache == {}

    @pytest.mark.asyncio
    @patch('neo4j.GraphDatabase.driver')
    async def test_async_methods(self, mock_driver_class):
        """Test asynchronous methods."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Patch the init methods to avoid issues
        with patch.object(Neo4jStore, '_initialize_schema'), \
             patch.object(Neo4jStore, '_load_from_neo4j'), \
             patch.object(Neo4jStore, '_save_history_to_neo4j'), \
             patch.object(Neo4jStore, '_save_cache_to_neo4j'):
            
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            # Clear store for clean test
            store.history = []
            store.cache = {}
            
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

    @patch('neo4j.GraphDatabase.driver')
    def test_close(self, mock_driver_class):
        """Test closing the driver connection."""
        # Setup mock
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver_class.return_value = mock_driver
        
        # Patch the init methods to avoid issues
        with patch.object(Neo4jStore, '_initialize_schema'), \
             patch.object(Neo4jStore, '_load_from_neo4j'):
            
            store = Neo4jStore(
                uri="neo4j+s://example.databases.neo4j.io",
                username="neo4j",
                password="password"
            )
            
            store.close()
            
            # Verify driver.close() was called
            mock_driver.close.assert_called_once() 