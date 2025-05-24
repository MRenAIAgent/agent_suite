#!/usr/bin/env python
"""
Simple Neo4j connection test script.

This script tests the connection to a Neo4j database and 
performs basic operations to verify functionality.

Usage:
    NEO4J_URI=<uri> NEO4J_USERNAME=<username> NEO4J_PASSWORD=<password> python examples/neo4j_test.py
"""

import os
import sys
from neo4j import GraphDatabase, exceptions

# Get connection details from environment
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

def test_connection():
    """Test the connection to Neo4j and perform basic operations."""
    if not (NEO4J_URI and NEO4J_PASSWORD):
        print("Error: Neo4j connection details not provided.")
        print("Please set the following environment variables:")
        print("  NEO4J_URI - The URI for your Neo4j instance (e.g., neo4j+s://xxx.databases.neo4j.io)")
        print("  NEO4J_USERNAME - The username (default: neo4j)")
        print("  NEO4J_PASSWORD - The password for your Neo4j database")
        sys.exit(1)
    
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    
    try:
        # Create a driver instance
        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
        # Verify connectivity
        with driver.session(database=NEO4J_DATABASE) as session:
            # Run a simple query
            result = session.run("MATCH (n) RETURN count(n) AS count").single()
            count = result["count"]
            print(f"Successfully connected! Found {count} nodes in the database.")
            
            # Create a test node with required properties (to satisfy any constraints)
            session.run(
                "CREATE (n:TestNode {name: $name, created: datetime(), id: $id}) RETURN n",
                name="Connection Test",
                id=str(hash("test"))
            )
            print("Created a test node.")
            
            # Query to find the test node
            result = session.run(
                "MATCH (n:TestNode {name: $name}) RETURN n.name AS name, toString(n.created) AS created",
                name="Connection Test"
            ).single()
            
            if result:
                print(f"Found test node: {result['name']} (created at {result['created']})")
            
            # Clean up - delete test node
            session.run(
                "MATCH (n:TestNode {name: $name}) DELETE n",
                name="Connection Test"
            )
            print("Deleted the test node.")
        
        # Close the driver
        driver.close()
        print("Connection test completed successfully!")
        
    except exceptions.ServiceUnavailable:
        print("❌ Error: Could not connect to Neo4j. Please check your connection URI.")
        sys.exit(1)
    except exceptions.AuthError:
        print("❌ Error: Authentication failed. Please check your username and password.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_connection() 