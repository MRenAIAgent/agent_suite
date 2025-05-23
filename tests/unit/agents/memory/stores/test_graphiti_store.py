"""Test suite for the GraphitiStore class."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import sys

# Create a mock graphiti module
graphiti_mock = MagicMock()
graphiti_mock.Graph = MagicMock()

# Create proper node mock with attributes that will be accessed
node_props = {
    "id": "test-id-1",
    "role": "user",
    "content": "Test content",
    "namespace": "test",
    "timestamp": 1234567890,
    "metadata": "{}"
}
node_mock = MagicMock()
# Setup properties so that when accessed like properties["role"], it returns the correct value
node_mock.properties.__getitem__.side_effect = lambda k: node_props[k]
node_mock.configure_mock(**{"properties": node_props})

# Patch sys.modules with our mock before importing
sys.modules['graphiti'] = graphiti_mock

# Import after patching
from agents.memory.stores.store import Store
from agents.memory.stores.graphiti_store import GraphitiStore, GRAPHITI_AVAILABLE


# Create a concrete test class that implements the abstract methods
class TestableGraphitiStore(GraphitiStore):
    """Concrete implementation of GraphitiStore for testing."""
    
    async def async_add_history(self, message):
        return self.add_history(message)
    
    async def async_get_history(self, limit=None):
        return self.get_history(limit)
    
    async def async_clear_history(self):
        self.clear(history=True, cache=False)
        return True
    
    async def async_set_cache(self, key, value):
        return self.set_cache(key, value)
    
    async def async_fetch_cache(self, key):
        return self.fetch_cache(key)
    
    async def async_clear_cache(self):
        self.clear_cache()
        return True


@pytest.fixture
def mock_graph():
    """Create a mock Graph."""
    mock = MagicMock()
    mock.node_types = {}
    mock.edge_types = {}
    mock.transaction.return_value.__enter__.return_value = MagicMock()
    return mock


class TestGraphitiStore:
    """Test suite for the GraphitiStore class."""
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_init(self, mock_graph):
        """Test initialization with default parameters."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        
        # Execute
        store = TestableGraphitiStore(db_path="test.graph", namespace="test_ns")
        
        # Assert
        assert store.db_path == "test.graph"
        assert store.namespace == "test_ns"
        assert store.graph is not None
        graphiti_mock.Graph.assert_called_with("test.graph", create_if_missing=True)
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', False)
    def test_init_graphiti_not_available(self):
        """Test initialization when Graphiti is not available."""
        with pytest.raises(ImportError):
            TestableGraphitiStore()
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_connect(self, mock_graph):
        """Test connecting to the graph database."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        
        # Execute
        store = TestableGraphitiStore(auto_connect=False)
        store.graph = None  # Ensure it's None
        
        success = store.connect()
        
        # Assert
        assert success
        assert store.graph is not None
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_disconnect(self, mock_graph):
        """Test disconnecting from the graph database."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        
        # Execute
        store = TestableGraphitiStore()
        store.graph = mock_graph  # Ensure it has a graph
        
        success = store.disconnect()
        
        # Assert
        assert success
        assert store.graph is None
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_ensure_schema(self, mock_graph):
        """Test schema creation."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        tx_mock = MagicMock()
        mock_graph.transaction.return_value.__enter__.return_value = tx_mock
        
        # Execute
        store = TestableGraphitiStore()
        
        # Call _ensure_schema explicitly
        store._ensure_schema()
        
        # Verify node types were created with correct method
        assert mock_graph.create_node_type.call_count >= 2
        assert mock_graph.create_edge_type.call_count >= 3
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_add_history(self, mock_graph):
        """Test adding a message to history."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        tx_mock = MagicMock()
        mock_graph.transaction.return_value.__enter__.return_value = tx_mock
        
        # Execute
        store = TestableGraphitiStore()
        message = {"role": "user", "content": "Test message"}
        
        store.add_history(message)
        
        # Verify message was added to local history
        assert len(store.history) == 1
        assert store.history[0] == message
        
        # Verify transaction was created
        assert mock_graph.transaction.called
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_get_history(self, mock_graph):
        """Test getting conversation history."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        
        # Execute
        store = TestableGraphitiStore()
        
        # Add messages
        store.history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Message 2"},
            {"role": "user", "content": "Message 3"}
        ]
        
        # Test get all
        history = store.get_history()
        assert len(history) == 3
        
        # Test with limit
        history = store.get_history(limit=2)
        assert len(history) == 2
        assert history[0]["content"] == "Message 2"
        assert history[1]["content"] == "Message 3"
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_cache_operations(self, mock_graph):
        """Test cache operations."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        
        # Execute
        store = TestableGraphitiStore()
        
        # Test set and fetch
        store.set_cache("key1", "value1")
        assert store.fetch_cache("key1") == "value1"
        
        # Test clear cache
        store.clear_cache()
        assert len(store.cache) == 0
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_clear(self, mock_graph):
        """Test clear operations."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        tx_mock = MagicMock()
        mock_graph.transaction.return_value.__enter__.return_value = tx_mock
        
        # Execute
        store = TestableGraphitiStore()
        store.history = [{"role": "user", "content": "Test"}]
        store.cache = {"key": "value"}
        
        # Test clear both
        store.clear()
        assert len(store.history) == 0
        assert len(store.cache) == 0
        
        # Test clear history only
        store.history = [{"role": "user", "content": "Test"}]
        store.cache = {"key": "value"}
        store.clear(history=True, cache=False)
        assert len(store.history) == 0
        assert len(store.cache) == 1
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_load_history(self, mock_graph):
        """Test loading history from database."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        tx_mock = MagicMock()
        
        # Create record mock with node properly configured
        record_mock = MagicMock()
        record_mock.__getitem__.return_value = node_mock
        
        tx_mock.run.return_value = [record_mock]
        mock_graph.transaction.return_value.__enter__.return_value = tx_mock
        
        # Make sure node properties work for json.loads
        node_props["metadata"] = "{}"
        
        # Execute
        store = TestableGraphitiStore()
        store.history = []  # Ensure history is empty
        
        # Define how the implementation handles the data
        def message_from_node(node):
            return {
                "id": node.properties["id"],
                "role": node.properties["role"],
                "content": node.properties["content"],
                "metadata": json.loads(node.properties["metadata"]) if node.properties["metadata"] else {}
            }
            
        # Patch the implementation to use our conversion
        with patch.object(TestableGraphitiStore, '_message_from_node', side_effect=message_from_node):
            # Call load_history
            history = store.load_history()
        
        # Verify history loaded
        assert len(history) >= 1
        if len(history) >= 1:
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "Test content"
    
    @patch('agents.memory.stores.graphiti_store.GRAPHITI_AVAILABLE', True)
    def test_search(self, mock_graph):
        """Test search functionality."""
        # Setup
        graphiti_mock.Graph.return_value = mock_graph
        tx_mock = MagicMock()
        
        # Create record mock with node properly configured
        record_mock = MagicMock()
        record_mock.__getitem__.return_value = node_mock
        
        # Setup for two different query results
        tx_mock.run.side_effect = [
            [record_mock],  # Content search results
            []  # Entity search results (empty)
        ]
        mock_graph.transaction.return_value.__enter__.return_value = tx_mock
        
        # Make sure node properties work for json.loads
        node_props["metadata"] = "{}"
        
        # Execute
        store = TestableGraphitiStore()
        
        # Define how the implementation handles the data
        def message_from_node(node):
            data = {
                "id": node.properties["id"],
                "role": node.properties["role"],
                "content": node.properties["content"],
                "metadata": json.loads(node.properties["metadata"]) if node.properties["metadata"] else {}
            }
            data["match_type"] = "content"
            return data
            
        # Patch the implementation to use our conversion
        with patch.object(TestableGraphitiStore, '_message_from_node', side_effect=message_from_node):
            # Search for query
            results = store.search("test query")
        
        # Verify results
        assert len(results) >= 1
        if len(results) >= 1:
            assert results[0]["role"] == "user"
            assert results[0]["content"] == "Test content"
            assert results[0]["match_type"] == "content"


# Skip the async tests if pytest-asyncio is not installed
pytestmark = pytest.mark.skipif(
    not hasattr(pytest, "mark") or not hasattr(pytest.mark, "asyncio"),
    reason="pytest-asyncio not installed"
)


if __name__ == "__main__":
    pytest.main() 