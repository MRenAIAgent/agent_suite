#!/usr/bin/env python
"""Integration test for Neo4jStore with LongtermMemory.

This test verifies that the Neo4jStore implementation works correctly
when used with the LongtermMemory manager.

Before running this test, set the following environment variables:
NEO4J_URI - The URI for your Neo4j AuraDB Free instance
NEO4J_USERNAME - The username for your Neo4j database (default: neo4j)
NEO4J_PASSWORD - The password for your Neo4j database

To run this test:
python -m tests.integration.test_neo4j_with_longterm_memory
"""

import os
import uuid
import pytest
from agents.memory.stores.neo4j_store import Neo4jStore
from agents.memory.longterm_memory import LongtermMemory


# Get connection details from environment (or use defaults for params that have them)
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


@pytest.mark.skipif(
    not (NEO4J_URI and NEO4J_PASSWORD),
    reason="Neo4j connection details not provided in environment variables"
)
class TestNeo4jWithLongtermMemory:
    """Test suite for Neo4jStore working with LongtermMemory manager."""
    
    @classmethod
    def setup_class(cls):
        """Create a unique prefix for this test run to avoid collisions."""
        cls.test_prefix = f"ltm-{uuid.uuid4()}:"
        
        # Create a Neo4jStore instance
        cls.store = Neo4jStore(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            prefix=cls.test_prefix
        )
        
        # Initialize LongtermMemory with the Neo4jStore
        cls.memory = LongtermMemory(
            max_history=100,
            session_id="neo4j-test",
            store=cls.store
        )
        
        # Clean up any existing test data
        cls.store.clear()
    
    @classmethod
    def teardown_class(cls):
        """Clean up after tests."""
        cls.store.clear()
        cls.store.close()
    
    def test_longterm_memory_integration(self):
        """Test that Neo4jStore works correctly with LongtermMemory."""
        # Add some messages to memory
        self.memory.add({"role": "user", "content": "Hello Neo4j!"})
        self.memory.add({"role": "assistant", "content": "Hi there! I'm using Neo4j for storage."})
        self.memory.add({"role": "user", "content": "How does it work?"})
        
        # Get history
        history = self.memory.get_history()
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello Neo4j!"
        
        # Test that memory is persisted by creating a new memory manager
        new_memory = LongtermMemory(
            max_history=100,
            session_id="neo4j-test",  # Same session ID to retrieve the same data
            store=Neo4jStore(
                uri=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
                prefix=self.test_prefix
            )
        )
        
        # Load memory
        loaded_memory = new_memory.load_memory()
        assert len(loaded_memory) == 3
        assert loaded_memory[2]["role"] == "user"
        assert loaded_memory[2]["content"] == "How does it work?"
        
        # Add more messages to the new memory
        new_memory.add({"role": "assistant", "content": "Neo4j stores data as a graph."})
        
        # Verify both memories are in sync when loaded
        assert len(new_memory.get_history()) == 4
        
        # Reload the original memory
        self.memory.load_memory()
        assert len(self.memory.get_history()) == 4
        
        # Clean up the new memory
        new_memory.purge()
    
    def test_memory_purge(self):
        """Test purging memory with Neo4jStore."""
        # Initialize with a different session ID
        purge_memory = LongtermMemory(
            max_history=100,
            session_id="neo4j-purge-test",
            store=Neo4jStore(
                uri=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
                prefix=f"purge-{uuid.uuid4()}:"  # Unique prefix
            )
        )
        
        # Add some messages
        purge_memory.add({"role": "user", "content": "Testing purge."})
        purge_memory.add({"role": "assistant", "content": "I will be purged soon."})
        
        # Verify messages were added
        assert len(purge_memory.get_history()) == 2
        
        # Purge memory
        success = purge_memory.purge()
        assert success is True
        
        # Verify memory was purged
        assert len(purge_memory.get_history()) == 0
        
        # Try loading memory from storage
        loaded = purge_memory.load_memory()
        assert len(loaded) == 0


if __name__ == "__main__":
    if not (NEO4J_URI and NEO4J_PASSWORD):
        print("Error: Neo4j connection details not provided.")
        print("Please set the following environment variables:")
        print("  NEO4J_URI - The URI for your Neo4j AuraDB Free instance")
        print("  NEO4J_USERNAME - The username (default: neo4j)")
        print("  NEO4J_PASSWORD - The password for your Neo4j database")
        exit(1)
    
    # Create a test instance
    test = TestNeo4jWithLongtermMemory()
    test.setup_class()
    
    try:
        print("Testing Neo4j integration with LongtermMemory...")
        test.test_longterm_memory_integration()
        print("✅ LongtermMemory integration test passed!")
        
        test.test_memory_purge()
        print("✅ Memory purge test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        test.teardown_class() 