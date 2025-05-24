"""Unit tests for the GraphStorage interface."""

import pytest
from unittest.mock import AsyncMock, patch

from agents.storage.graph_storage import GraphStorage


@pytest.mark.asyncio
async def test_graph_storage_base_methods(mock_graph_storage):
    """Test that the graph storage implements BaseStorage methods."""
    # Test inherited BaseStorage methods
    key = "test-key"
    data = {"type": "node", "node_type": "Person", "properties": {"name": "Alice"}}
    
    # Test store method
    result = await mock_graph_storage.store(key, data)
    assert result == "mock-id"
    mock_graph_storage.store_mock.assert_called_once_with(key, data)
    
    # Test retrieve method
    result = await mock_graph_storage.retrieve(key)
    assert result["type"] == "node"
    assert "properties" in result
    mock_graph_storage.retrieve_mock.assert_called_once_with(key)
    
    # Test delete method
    result = await mock_graph_storage.delete(key)
    assert result is True
    mock_graph_storage.delete_mock.assert_called_once_with(key)
    
    # Test update method
    new_data = {"properties": {"name": "Alice Smith", "age": 30}}
    result = await mock_graph_storage.update(key, new_data)
    assert result is True
    mock_graph_storage.update_mock.assert_called_once_with(key, new_data)
    
    # Test list method
    pattern = "test-*"
    result = await mock_graph_storage.list(pattern)
    assert result == ["id1", "id2"]
    mock_graph_storage.list_mock.assert_called_once_with(pattern)
    
    # Test clear method
    result = await mock_graph_storage.clear()
    assert result is True
    mock_graph_storage.clear_mock.assert_called_once()


@pytest.mark.asyncio
async def test_graph_storage_node_methods(mock_graph_storage):
    """Test graph node-specific methods."""
    # Test create_node method
    node_type = "Person"
    properties = {"name": "Alice", "age": 30}
    result = await mock_graph_storage.create_node(node_type, properties)
    assert result == "node-id"
    mock_graph_storage.create_node_mock.assert_called_once_with(node_type, properties)
    
    # Test get_nodes method
    node_type = "Person"
    properties = {"name": "Alice"}
    limit = 10
    result = await mock_graph_storage.get_nodes(node_type, properties, limit)
    assert len(result) == 2
    assert result[0]["type"] == "Person"
    assert result[0]["properties"]["name"] == "Alice"
    mock_graph_storage.get_nodes_mock.assert_called_once_with(node_type, properties, limit)


@pytest.mark.asyncio
async def test_graph_storage_relationship_methods(mock_graph_storage):
    """Test graph relationship-specific methods."""
    # Test create_relationship method
    source_id = "node1"
    target_id = "node2"
    rel_type = "KNOWS"
    properties = {"since": "2020", "strength": "strong"}
    
    result = await mock_graph_storage.create_relationship(
        source_id, target_id, rel_type, properties
    )
    assert result == "rel-id"
    mock_graph_storage.create_relationship_mock.assert_called_once_with(
        source_id, target_id, rel_type, properties
    )
    
    # Test get_relationships method
    source_id = "node1"
    rel_type = "KNOWS"
    result = await mock_graph_storage.get_relationships(
        source_id=source_id, relationship_type=rel_type
    )
    assert len(result) == 1
    assert result[0]["relationship"]["type"] == "KNOWS"
    assert result[0]["source"]["id"] == "node1"
    assert result[0]["target"]["id"] == "node2"
    mock_graph_storage.get_relationships_mock.assert_called_once_with(
        source_id, None, rel_type, None, 100
    )


@pytest.mark.asyncio
async def test_graph_storage_query_methods(mock_graph_storage):
    """Test graph query methods."""
    # Test query method
    query = "MATCH (n:Person) WHERE n.name = $name RETURN n"
    parameters = {"name": "Alice"}
    
    result = await mock_graph_storage.query(query, parameters)
    assert len(result) == 1
    assert result[0]["n"]["name"] == "test"
    mock_graph_storage.query_mock.assert_called_once_with(query, parameters)
    
    # Test traverse method
    start_node_id = "node1"
    relationship_types = ["KNOWS", "FOLLOWS"]
    direction = "outgoing"
    max_depth = 2
    
    result = await mock_graph_storage.traverse(
        start_node_id, relationship_types, direction, max_depth
    )
    assert result["start_node"]["id"] == "node1"
    assert len(result["paths"]) == 1
    mock_graph_storage.traverse_mock.assert_called_once_with(
        start_node_id, relationship_types, direction, max_depth
    )


@pytest.mark.asyncio
async def test_graph_storage_with_kwargs(mock_graph_storage):
    """Test that graph storage methods properly handle kwargs."""
    # Test with extra parameters
    await mock_graph_storage.create_node("Person", {"name": "Alice"}, labels=["User"])
    mock_graph_storage.create_node_mock.assert_called_with(
        "Person", {"name": "Alice"}, labels=["User"]
    )
    
    await mock_graph_storage.get_nodes("Person", {"active": True}, include_labels=True)
    mock_graph_storage.get_nodes_mock.assert_called_with(
        "Person", {"active": True}, 100, include_labels=True
    )
    
    await mock_graph_storage.traverse("node1", direction="both", use_cache=False)
    mock_graph_storage.traverse_mock.assert_called_with(
        "node1", None, "both", 1, use_cache=False
    ) 