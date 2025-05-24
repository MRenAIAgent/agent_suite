"""Integration tests for the GraphitiNeo4jStore class.

These tests verify that the GraphitiNeo4jStore correctly integrates
Graphiti with a Neo4j backend for persistent memory storage.
"""

import os
import json
import time
import uuid
import pytest
from typing import Dict, Any

from agents.memory.stores.graphiti_neo4j_store import GraphitiNeo4jStore

# Skip if no Neo4j credentials
neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_username = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")
neo4j_database = os.environ.get("NEO4J_DATABASE", "neo4j")

pytestmark = pytest.mark.skipif(
    not all([neo4j_uri, neo4j_username, neo4j_password]),
    reason="Neo4j credentials required for Neo4j integration tests"
)


class TestGraphitiNeo4jStore:
    """Tests for the GraphitiNeo4jStore class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.store = GraphitiNeo4jStore(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            namespace="test_suite",
            prefix="test_integration:"
        )
        self.store.clear_history()
        self.store.clear_cache()
    
    def teardown_method(self):
        """Clean up after test."""
        self.store.clear_history()
        self.store.clear_cache()
        self.store.close()
    
    def test_add_and_get_history(self):
        """Test adding and retrieving history."""
        # Add a message
        message1 = {"role": "user", "content": "Test message 1", "id": str(uuid.uuid4())}
        self.store.add_history(message1)
        
        # Add another message
        message2 = {"role": "assistant", "content": "Test response 1", "id": str(uuid.uuid4())}
        self.store.add_history(message2)
        
        # Get history
        history = self.store.get_history()
        
        # Verify results
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Test message 1"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Test response 1"
    
    def test_cache_operations(self):
        """Test cache operations."""
        # Set cache values
        self.store.set_cache("key1", "value1")
        self.store.set_cache("key2", {"nested": "value2"})
        
        # Get single cache value
        value1 = self.store.fetch_cache("key1")
        assert value1 == "value1"
        
        # Get nested cache value
        value2 = self.store.fetch_cache("key2")
        assert value2["nested"] == "value2"
        
        # Get all cache values
        all_cache = self.store.fetch_cache()
        assert len(all_cache) == 2
        assert all_cache["key1"] == "value1"
        assert all_cache["key2"]["nested"] == "value2"
        
        # Clear cache
        self.store.clear_cache()
        assert len(self.store.fetch_cache()) == 0
    
    def test_clear_operations(self):
        """Test clear operations."""
        # Add history
        message = {"role": "user", "content": "Test clear", "id": str(uuid.uuid4())}
        self.store.add_history(message)
        
        # Set cache
        self.store.set_cache("test_key", "test_value")
        
        # Verify data exists
        assert len(self.store.get_history()) == 1
        assert self.store.fetch_cache("test_key") == "test_value"
        
        # Clear just history
        self.store.clear(history=True, cache=False)
        assert len(self.store.get_history()) == 0
        assert self.store.fetch_cache("test_key") == "test_value"
        
        # Add history again
        self.store.add_history(message)
        
        # Clear just cache
        self.store.clear(history=False, cache=True)
        assert len(self.store.get_history()) == 1
        assert self.store.fetch_cache("test_key") is None
        
        # Clear both
        self.store.clear()
        assert len(self.store.get_history()) == 0
        assert self.store.fetch_cache("test_key") is None
    
    def test_save_and_load_history(self):
        """Test saving and loading history."""
        # Add messages
        message1 = {"role": "user", "content": "Save test", "id": str(uuid.uuid4())}
        message2 = {"role": "assistant", "content": "Load test", "id": str(uuid.uuid4())}
        self.store.add_history(message1)
        self.store.add_history(message2)
        
        # Create a new store with the same namespace
        new_store = GraphitiNeo4jStore(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            namespace="test_suite",
            prefix="test_integration:"
        )
        
        # Load history in the new store
        loaded_history = new_store.load_history()
        
        # Verify results
        assert len(loaded_history) == 2
        assert loaded_history[0]["role"] == "user"
        assert loaded_history[0]["content"] == "Save test"
        assert loaded_history[1]["role"] == "assistant"
        assert loaded_history[1]["content"] == "Load test"
        
        # Clean up
        new_store.close()

    def test_entity_extraction(self):
        """Test that entity extraction and linking works."""
        # Add a message with entities
        message = {
            "role": "user", 
            "content": "Neo4j is a graph database used with Graphiti.",
            "id": str(uuid.uuid4())
        }
        self.store.add_history(message)
        
        # This test just verifies that the message was added without errors
        # as the entity extraction is mostly handled internally in Neo4j
        history = self.store.get_history()
        assert len(history) == 1
        assert history[0]["content"] == "Neo4j is a graph database used with Graphiti." 