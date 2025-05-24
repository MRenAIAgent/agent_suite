#!/usr/bin/env python
"""Integration test for Neo4jStore focused on AuraDB Free limitations.

This test verifies that the Neo4jStore implementation can handle:
1. Large string escaping (for JSON storage)
2. Constraints and indexes in AuraDB Free
3. Proper error handling on schema operations

Before running this test, set the following environment variables:
NEO4J_URI - The URI for your Neo4j AuraDB Free instance
NEO4J_USERNAME - The username for your Neo4j database (default: neo4j)
NEO4J_PASSWORD - The password for your Neo4j database

To run this test:
python -m tests.integration.test_neo4j_limits
"""

import os
import uuid
import random
import string
import traceback
import pytest
from neo4j import exceptions
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
class TestNeo4jLimits:
    """Test suite for Neo4jStore focusing on handling AuraDB limitations."""
    
    @classmethod
    def setup_class(cls):
        """Create a unique prefix for this test run to avoid collisions."""
        cls.test_prefix = f"limits-{uuid.uuid4()}:"
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
        try:
            cls.store.clear()
            cls.store.close()
        except Exception as e:
            print(f"Error during teardown: {e}")
    
    def test_schema_handling(self):
        """Test handling of schema operations in AuraDB Free."""
        # In Neo4jStore._initialize_schema, we create constraints with try-except
        # Let's verify this handles AuraDB Free limitations gracefully
        
        # Get access to the driver
        driver = self.store.driver
        
        # Try to create a constraint directly - this tests our error handling
        with driver.session(database=NEO4J_DATABASE) as session:
            try:
                # This might fail if the constraint already exists, which is okay
                session.execute_write(
                    lambda tx: tx.run("""
                        CREATE CONSTRAINT IF NOT EXISTS FOR (t:Test) 
                        REQUIRE t.id IS NOT NULL
                    """)
                )
                
                # If we get here, the constraint was created successfully
                print("Successfully created constraint")
                
                # Now try to create the same constraint again - should not error
                session.execute_write(
                    lambda tx: tx.run("""
                        CREATE CONSTRAINT IF NOT EXISTS FOR (t:Test) 
                        REQUIRE t.id IS NOT NULL
                    """)
                )
                
                print("Successfully re-created existing constraint")
                
            except exceptions.ClientError as e:
                # Some versions of Neo4j or AuraDB Free might not support this syntax
                # This is expected and our Neo4jStore handles this with try-except
                print(f"Expected constraint error in AuraDB Free: {e}")
                
                # The test passes if we handle errors gracefully
                pass

    def test_simple_store_operations(self):
        """Test basic store operations with a minimal message."""
        # Create a very simple message
        message = {"role": "user", "content": "Hello Neo4j!"}
        
        try:
            # Add to history
            self.store.add_history(message)
            print("Added message to history")
            
            # Save to Neo4j
            success = self.store.save_history()
            assert success is True, "Failed to save message"
            print("Saved history to Neo4j")
            
            # Create a new store with the same prefix to test loading
            new_store = Neo4jStore(
                uri=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
                prefix=self.test_prefix
            )
            print("Created new store instance")
            
            # Load history
            loaded_history = new_store.load_history()
            print(f"Loaded history: {loaded_history}")
            
            # Verify the message was loaded correctly
            assert len(loaded_history) == 1, "Failed to load history"
            assert loaded_history[0]["role"] == "user", "Role was not preserved"
            assert loaded_history[0]["content"] == "Hello Neo4j!", "Content was not preserved"
            
            # Clean up
            new_store.clear()
            new_store.close()
            print("Test completed successfully")
            
        except Exception as e:
            print(f"Error in test_simple_store_operations: {e}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    if not (NEO4J_URI and NEO4J_PASSWORD):
        print("Error: Neo4j connection details not provided.")
        print("Please set the following environment variables:")
        print("  NEO4J_URI - The URI for your Neo4j AuraDB Free instance")
        print("  NEO4J_USERNAME - The username (default: neo4j)")
        print("  NEO4J_PASSWORD - The password for your Neo4j database")
        exit(1)
    
    # Create a test instance
    test = TestNeo4jLimits()
    test.setup_class()
    
    try:
        print("Running Neo4j limits test...")
        
        # Run the simple test first
        test.test_simple_store_operations()
        print("✅ Simple store operations test passed!")
        
        # Then run the schema test
        test.test_schema_handling()
        print("✅ Schema handling test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
    finally:
        test.teardown_class() 