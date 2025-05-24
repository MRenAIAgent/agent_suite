#!/usr/bin/env python
"""
Count Neo4j nodes created by the GraphitiNeo4jStore tests.

This script connects to Neo4j and counts the number of nodes and relationships
created during the GraphitiNeo4jStore tests.

Usage:
    python examples/count_neo4j_nodes.py
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

def main():
    """Connect to Neo4j and count nodes."""
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j connection details
    uri = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    
    if not all([uri, password]):
        print("Error: Missing Neo4j connection details in environment.")
        sys.exit(1)
    
    print(f"Connecting to Neo4j at {uri}...")
    
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Count nodes and relationships by type in test namespace
        with driver.session(database=database) as session:
            # Count message nodes
            message_count = session.run(
                """
                MATCH (m:Message)
                WHERE m.namespace = 'test_suite' AND m.prefix = 'test_integration:'
                RETURN count(m) as count
                """
            ).single()["count"]
            
            # Count entity nodes
            entity_count = session.run(
                """
                MATCH (e:Entity)
                WHERE e.namespace = 'test_suite' AND e.prefix = 'test_integration:'
                RETURN count(e) as count
                """
            ).single()["count"]
            
            # Count relationships
            next_rel_count = session.run(
                """
                MATCH (:Message)-[r:NEXT]->(:Message)
                WHERE r.timestamp IS NOT NULL
                RETURN count(r) as count
                """
            ).single()["count"]
            
            mentions_rel_count = session.run(
                """
                MATCH (:Message)-[r:MENTIONS]->(:Entity)
                RETURN count(r) as count
                """
            ).single()["count"]
            
            # Get a sample of entity values
            entities = session.run(
                """
                MATCH (e:Entity)
                WHERE e.namespace = 'test_suite' AND e.prefix = 'test_integration:'
                RETURN e.value as value
                LIMIT 10
                """
            ).values()
            entity_values = [record[0] for record in entities]
            
        # Display results
        print("\nNeo4j Node Counts for 'test_suite' namespace:")
        print(f"Message nodes: {message_count}")
        print(f"Entity nodes: {entity_count}")
        print(f"Total nodes: {message_count + entity_count}")
        print("\nNeo4j Relationship Counts:")
        print(f"NEXT relationships: {next_rel_count}")
        print(f"MENTIONS relationships: {mentions_rel_count}")
        print(f"Total relationships: {next_rel_count + mentions_rel_count}")
        
        if entity_values:
            print("\nSample entity values extracted:")
            for value in entity_values:
                print(f"- {value}")
        
        # Close the connection
        driver.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 