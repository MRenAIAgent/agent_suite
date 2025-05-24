#!/usr/bin/env python
"""
Demo script for using Neo4jStore with AuraDB Free.

This script demonstrates how to connect to a Neo4j AuraDB Free database
and use it for storing agent memory.

Setup instructions:
1. Sign up for Neo4j AuraDB Free at https://console.neo4j.io/
2. Create a new database (AuraDB Free)
3. Get the connection URI, username and password
4. Update the connection details below
"""

import os
import time
from agents.memory.stores.neo4j_store import Neo4jStore

# Update these with your AuraDB Free connection details
NEO4J_URI = "neo4j+s://xxxxxxxx.databases.neo4j.io"  # Replace with your AuraDB connection URI
NEO4J_USERNAME = "neo4j"  # Replace with your AuraDB username
NEO4J_PASSWORD = "your_password"  # Replace with your AuraDB password
NEO4J_DATABASE = "neo4j"  # Default database name (usually "neo4j")

# For security, you can also use environment variables instead of hardcoding credentials
# NEO4J_URI = os.environ.get("NEO4J_URI")
# NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
# NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


def main():
    """Run the Neo4jStore demo."""
    print("Connecting to Neo4j AuraDB Free...")
    
    try:
        # Initialize the Neo4j store
        store = Neo4jStore(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            prefix="demo:"
        )
        
        print("Connected successfully!")
        
        # Clear any existing data
        print("Clearing existing data...")
        store.clear()
        
        # Add some messages to history
        print("\nAdding messages to history...")
        store.add_history({"role": "user", "content": "Hello, Neo4j!"})
        store.add_history({"role": "assistant", "content": "Hi there! How can I help you with Neo4j today?"})
        store.add_history({"role": "user", "content": "Show me how to store data in Neo4j."})
        
        # Add some items to cache
        print("Adding items to cache...")
        store.set_cache("favorite_db", "Neo4j")
        store.set_cache("graph_types", ["property", "rdf", "labeled"])
        store.set_cache("year_founded", 2007)
        
        # Retrieve and display history
        print("\nHistory:")
        for idx, msg in enumerate(store.get_history()):
            print(f"  {idx+1}. {msg['role']}: {msg['content']}")
        
        # Retrieve and display cache
        print("\nCache:")
        cache = store.fetch_cache()
        for key, value in cache.items():
            print(f"  {key}: {value}")
        
        # Demonstrate persistence by creating a new store instance
        print("\nCreating new store instance to verify persistence...")
        new_store = Neo4jStore(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            prefix="demo:"
        )
        
        # Verify data is still there
        print("\nHistory (from new store instance):")
        for idx, msg in enumerate(new_store.get_history()):
            print(f"  {idx+1}. {msg['role']}: {msg['content']}")
        
        print("\nCache (from new store instance):")
        cache = new_store.fetch_cache()
        for key, value in cache.items():
            print(f"  {key}: {value}")
        
        # Clean up - remove test data
        print("\nCleaning up...")
        new_store.clear()
        
        # Close connections
        store.close()
        new_store.close()
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease check your connection details and make sure your AuraDB instance is running.")
        print("Sign up for AuraDB Free at https://console.neo4j.io/ if you haven't already.")


if __name__ == "__main__":
    main() 