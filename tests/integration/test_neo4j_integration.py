#!/usr/bin/env python
"""Integration test for Neo4jStore with a real Neo4j database.

Before running this test, set the following environment variables:
NEO4J_URI - The URI for your Neo4j AuraDB Free instance
NEO4J_USERNAME - The username for your Neo4j database (default: neo4j)
NEO4J_PASSWORD - The password for your Neo4j database

To run this test:
python -m tests.integration.test_neo4j_integration
"""

import os
import time
import uuid
import json
import pytest
from agents.memory.stores.neo4j_store import Neo4jStore


# Get connection details from environment (or use defaults for params that have them)
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


@pytest.mark.skipif(
    not (NEO4J_URI and NEO4J_PASSWORD),
    reason="Neo4j connection details not provided in environment variables"
)
class TestNeo4jIntegration:
    """Test suite for Neo4jStore with real Neo4j database."""
    
    @classmethod
    def setup_class(cls):
        """Create a unique prefix for this test run to avoid collisions."""
        cls.test_prefix = f"test-{uuid.uuid4()}:"
        cls.store = Neo4jStore(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            prefix=cls.test_prefix
        )
        
        # Clean up any existing test data
        cls.store.clear()
    
    @classmethod
    def teardown_class(cls):
        """Clean up after tests."""
        cls.store.clear()
        cls.store.close()
    
    def test_full_workflow(self):
        """Test the entire workflow with a real Neo4j database."""
        # Clear the store first to ensure clean state
        self.store.clear()
        
        # 1. Add messages to history
        self.store.add_history({"role": "user", "content": "What is Neo4j?"})
        self.store.add_history({"role": "assistant", "content": "Neo4j is a graph database platform."})
        
        # 2. Verify history was added correctly
        history = self.store.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is Neo4j?"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Neo4j is a graph database platform."
        
        # 3. Add and verify cache items
        self.store.set_cache("db_type", "graph")
        self.store.set_cache("languages", ["cypher", "gql"])
        
        cache = self.store.fetch_cache()
        assert len(cache) == 2
        assert cache["db_type"] == "graph"
        assert cache["languages"] == ["cypher", "gql"]
        
        # 4. Test persistence by creating a new store with the same prefix
        new_store = Neo4jStore(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            prefix=self.test_prefix
        )
        
        # 5. Verify data is still there in the new store
        history = new_store.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is Neo4j?"
        
        cache = new_store.fetch_cache()
        assert len(cache) == 2
        assert cache["db_type"] == "graph"
        
        # 6. Test save and load functions
        new_store.add_history({"role": "user", "content": "How do I connect to Neo4j?"})
        assert len(new_store.get_history()) == 3
        
        # Save history
        success = new_store.save_history()
        assert success is True
        
        # Create a third store to verify loading
        third_store = Neo4jStore(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            prefix=self.test_prefix
        )
        
        # Load history
        loaded_history = third_store.load_history()
        assert len(loaded_history) == 3
        assert loaded_history[2]["content"] == "How do I connect to Neo4j?"
        
        # 7. Test clear functions
        third_store.clear_history()
        assert len(third_store.get_history()) == 0
        assert len(third_store.fetch_cache()) == 2  # Cache should still be there
        
        third_store.clear_cache()
        assert len(third_store.fetch_cache()) == 0
        
        # Clean up
        new_store.close()
        third_store.close()


if __name__ == "__main__":
    if not (NEO4J_URI and NEO4J_PASSWORD):
        print("Error: Neo4j connection details not provided.")
        print("Please set the following environment variables:")
        print("  NEO4J_URI - The URI for your Neo4j AuraDB Free instance")
        print("  NEO4J_USERNAME - The username (default: neo4j)")
        print("  NEO4J_PASSWORD - The password for your Neo4j database")
        exit(1)
    
    # Create a test instance
    test = TestNeo4jIntegration()
    test.setup_class()
    
    try:
        print("Running Neo4j integration test...")
        test.test_full_workflow()
        print("✅ Integration test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        test.teardown_class() 