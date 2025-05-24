"""Integration tests for graph storage backends."""

import pytest
import uuid

from agents.storage import (
    GraphStorage,
    NEO4J_AVAILABLE
)


@pytest.mark.asyncio
@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j is not available")
async def test_neo4j_graph_storage(graph_storage):
    """Test the Neo4j graph storage implementation."""
    # Create nodes
    person1_props = {
        "name": "Alice",
        "age": 30,
        "interests": ["AI", "Programming"]
    }
    person1_id = await graph_storage.create_node("Person", person1_props)
    
    person2_props = {
        "name": "Bob",
        "age": 25,
        "interests": ["Machine Learning", "Data Science"]
    }
    person2_id = await graph_storage.create_node("Person", person2_props)
    
    topic_props = {
        "name": "Artificial Intelligence",
        "level": "Advanced"
    }
    topic_id = await graph_storage.create_node("Topic", topic_props)
    
    assert person1_id is not None
    assert person2_id is not None
    assert topic_id is not None
    
    # Create relationships
    rel1_props = {
        "since": "2020",
        "level": "Expert"
    }
    rel1_id = await graph_storage.create_relationship(
        person1_id, topic_id, "INTERESTED_IN", rel1_props
    )
    
    rel2_props = {
        "since": "2022",
        "level": "Beginner"
    }
    rel2_id = await graph_storage.create_relationship(
        person2_id, topic_id, "LEARNING", rel2_props
    )
    
    rel3_props = {
        "since": "2021",
        "relationship": "Colleague"
    }
    rel3_id = await graph_storage.create_relationship(
        person1_id, person2_id, "KNOWS", rel3_props
    )
    
    assert rel1_id is not None
    assert rel2_id is not None
    assert rel3_id is not None
    
    # Test retrieving a node
    node = await graph_storage.retrieve(person1_id)
    assert node is not None
    assert "properties" in node
    assert node["properties"]["name"] == "Alice"
    
    # Test retrieving a relationship
    # Usually relationships don't have direct retrieve unless custom implementation
    # supports it, so we'll use get_relationships instead
    
    # Test get_nodes method
    nodes = await graph_storage.get_nodes("Person")
    assert len(nodes) == 2
    
    # Test get_nodes with property filter
    nodes = await graph_storage.get_nodes("Person", {"name": "Alice"})
    assert len(nodes) == 1
    assert nodes[0]["properties"]["name"] == "Alice"
    
    # Test get_relationships method
    rels = await graph_storage.get_relationships(source_id=person1_id)
    assert len(rels) == 2  # INTERESTED_IN to topic and KNOWS to person2
    
    # Test get_relationships with type filter
    rels = await graph_storage.get_relationships(
        source_id=person1_id, relationship_type="KNOWS"
    )
    assert len(rels) == 1
    assert rels[0]["relationship"]["type"] == "KNOWS"
    assert rels[0]["target"]["id"] == person2_id
    
    # Test query method (Cypher query for Neo4j)
    results = await graph_storage.query(
        "MATCH (p:Person) WHERE p.name = $name RETURN p",
        {"name": "Alice"}
    )
    assert len(results) == 1
    
    # Test traverse method
    traversal = await graph_storage.traverse(
        person1_id,
        relationship_types=["KNOWS", "INTERESTED_IN"],
        max_depth=2
    )
    assert "start_node" in traversal
    assert "paths" in traversal
    assert len(traversal["paths"]) > 0
    
    # Test update method
    updated_props = {"age": 31, "job": "Developer"}
    result = await graph_storage.update(person1_id, {"properties": updated_props})
    assert result is True
    
    # Verify update
    node = await graph_storage.retrieve(person1_id)
    assert node["properties"]["age"] == 31
    assert node["properties"]["job"] == "Developer"
    
    # Test delete method
    result = await graph_storage.delete(person1_id)
    assert result is True
    
    # Verify deletion
    node = await graph_storage.retrieve(person1_id)
    assert node is None
    
    # Test list method
    node_ids = await graph_storage.list()
    assert person2_id in node_ids
    assert topic_id in node_ids
    assert person1_id not in node_ids  # We deleted this
    
    # Test clear method
    result = await graph_storage.clear()
    assert result is True
    
    # Verify clear
    nodes = await graph_storage.get_nodes()
    assert len(nodes) == 0 