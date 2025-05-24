#!/usr/bin/env python
"""
Simple test script for GraphitiNeo4jStore.

This script demonstrates using the GraphitiNeo4jStore to store and retrieve
conversation history with Neo4j as the backend for Graphiti.

Usage:
    python examples/graphiti_neo4j_test.py
"""

import os
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv

from agents.memory.stores.graphiti_neo4j_store import GraphitiNeo4jStore


async def add_conversation(store: GraphitiNeo4jStore) -> None:
    """Add a sample conversation to the store."""
    print("Adding sample conversation...")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Neo4j."},
        {"role": "assistant", "content": "Neo4j is a graph database that is designed to store, query, and analyze highly connected data."},
        {"role": "user", "content": "How does it connect with Graphiti?"},
        {"role": "assistant", "content": "Graphiti can use Neo4j as its backend storage, leveraging Neo4j's powerful graph capabilities for storing and querying the knowledge graph."}
    ]
    
    for message in messages:
        await store.async_add_history(message)
        print(f"Added message: {message['role']}")
    
    print("Conversation added!")


async def get_history(store: GraphitiNeo4jStore) -> None:
    """Get the conversation history."""
    print("\nGetting conversation history...")
    
    history = await store.async_get_history()
    print(f"Found {len(history)} messages:")
    
    for i, message in enumerate(history):
        print(f"  {i+1}. [{message.get('role', 'unknown')}]: {message.get('content', 'No content')[:50]}...")


async def main() -> None:
    """Main function to test GraphitiNeo4jStore."""
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j connection details from environment
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_username = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    neo4j_database = os.environ.get("NEO4J_DATABASE", "neo4j")
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("Error: Missing Neo4j connection details in .env file.")
        print("Please run: python examples/create_neo4j_env.py")
        return
    
    print(f"Connecting to Neo4j at {neo4j_uri}...")
    
    try:
        # Initialize the GraphitiNeo4jStore
        store = GraphitiNeo4jStore(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            namespace="test",
            prefix="graphiti_test:"
        )
        
        print("Connected to Neo4j!")
        
        # Clear any existing data
        await store.async_clear_history()
        await store.async_clear_cache()
        print("Cleared existing data.")
        
        # Add sample conversation
        await add_conversation(store)
        
        # Get the conversation history
        await get_history(store)
        
        # Close the connection
        store.close()
        print("\nConnection closed.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 