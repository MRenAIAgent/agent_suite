#!/usr/bin/env python
"""
Example demonstrating the centralized graph infrastructure with Graphiti.

This example will:
1. Create graphs in both memory and file storage modes
2. Add entities and relations
3. Perform queries and traversals
4. Show how the graph can be used for knowledge and storage

Usage:
    python example_graph.py
"""

import os
import uuid
from pathlib import Path
from pprint import pprint

# Import the centralized graph manager
from agent_suite.graph import GraphManager

def print_separator(title):
    """Print a section separator with title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def main():
    """Run the graph database example."""
    print_separator("Centralized Graph Infrastructure Example")
    
    # Create a graph manager with in-memory storage
    print("Creating in-memory graph...")
    memory_graph = GraphManager(
        graph_name="memory_example",
        persist=False  # In-memory only, not persisted
    )
    
    # Create a graph manager with file-based storage
    print("Creating file-based graph...")
    file_graph = GraphManager(
        graph_name="file_example",
        persist=True,  # Persist to disk
        persistence_path="./example_data"  # Store in example_data directory
    )
    
    # Let's use the file-based graph for our example
    graph = file_graph
    print(f"Using: {graph.graph_name} (persisted: {graph.persist})")
    
    # Add some entities
    print("\nAdding entities...")
    
    # Add person entities
    person1_id = graph.add_entity(
        entity_type="person",
        properties={
            "name": "Alice Smith",
            "age": 32,
            "occupation": "Data Scientist"
        }
    )
    
    person2_id = graph.add_entity(
        entity_type="person",
        properties={
            "name": "Bob Johnson",
            "age": 28,
            "occupation": "Software Engineer"
        }
    )
    
    # Add a project entity
    project_id = graph.add_entity(
        entity_type="project",
        properties={
            "name": "Knowledge Graph API",
            "status": "In Progress",
            "priority": "High"
        }
    )
    
    # Add some unstructured text
    text1_id = graph.add_text(
        text="The knowledge graph API helps organize information in a structured, connected manner.",
        properties={
            "title": "Knowledge Graph Overview",
            "category": "Documentation"
        }
    )
    
    text2_id = graph.add_text(
        text="Entities in the graph represent real-world objects or concepts, while relations show how they're connected.",
        properties={
            "title": "Graph Concepts",
            "category": "Tutorial"
        }
    )
    
    # Print entity IDs
    print(f"Added person: {person1_id} (Alice)")
    print(f"Added person: {person2_id} (Bob)")
    print(f"Added project: {project_id}")
    print(f"Added text: {text1_id}")
    print(f"Added text: {text2_id}")
    
    # Add relations between entities
    print("\nAdding relations...")
    
    # Alice works on the project
    relation1_id = graph.add_relation(
        source_id=person1_id,
        relation_type="WORKS_ON",
        target_id=project_id,
        properties={
            "role": "Lead",
            "since": "2023-01-15"
        }
    )
    
    # Bob works on the project
    relation2_id = graph.add_relation(
        source_id=person2_id,
        relation_type="WORKS_ON",
        target_id=project_id,
        properties={
            "role": "Developer",
            "since": "2023-02-01"
        }
    )
    
    # Alice knows Bob
    relation3_id = graph.add_relation(
        source_id=person1_id,
        relation_type="KNOWS",
        target_id=person2_id,
        properties={
            "relationship": "Colleague"
        }
    )
    
    # Project is described by text
    relation4_id = graph.add_relation(
        source_id=project_id,
        relation_type="DESCRIBED_BY",
        target_id=text1_id
    )
    
    print(f"Added relation: {relation1_id} (Alice WORKS_ON Project)")
    print(f"Added relation: {relation2_id} (Bob WORKS_ON Project)")
    print(f"Added relation: {relation3_id} (Alice KNOWS Bob)")
    print(f"Added relation: {relation4_id} (Project DESCRIBED_BY Text)")
    
    # Query entities
    print_separator("Entity Queries")
    
    # Query all persons
    print("Querying all persons:")
    persons = graph.query_entities(entity_type="person")
    for person in persons:
        print(f"- {person['properties'].get('name')} ({person['id']})")
    
    # Query relations
    print("\nQuerying who works on the project:")
    relations = graph.query_relations(
        relation_type="WORKS_ON",
        target_id=project_id
    )
    
    for relation in relations:
        source_id = relation["source_id"]
        person = graph.get_entity(source_id)
        role = relation["properties"].get("role", "Unknown")
        print(f"- {person['properties'].get('name')} as {role}")
    
    # Keyword search
    print_separator("Keyword Search")
    
    # Search for "knowledge graph"
    print("Searching for 'knowledge graph':")
    results = graph.keyword_search("knowledge graph")
    for result in results:
        if result["type"] == "text":
            print(f"- [{result['score']:.2f}] TEXT: {result['content'][:60]}...")
        else:
            print(f"- [{result['score']:.2f}] {result['type'].upper()}: {result['properties']}")
    
    # Get neighbors
    print_separator("Entity Neighbors")
    
    # Get project neighbors
    print(f"Neighbors of the project:")
    neighbors = graph.get_neighbors(project_id)
    
    for neighbor in neighbors:
        entity = neighbor["entity"]
        relation = neighbor["relation"]
        direction = relation["direction"]
        relation_type = relation["type"]
        
        if direction == "in":
            print(f"- {entity['properties'].get('name')} {relation_type} → Project")
        else:
            print(f"- Project {relation_type} → {entity['properties'].get('name') or entity['properties'].get('title')}")
    
    # Graph traversal
    print_separator("Graph Traversal")
    
    # Traverse from Alice
    print("Traversing graph from Alice (2 levels):")
    traversal = graph.graph_traversal(person1_id, max_depth=2)
    
    print(f"Found {len(traversal['nodes'])} nodes and {len(traversal['edges'])} edges")
    
    print("\nNodes:")
    for node in traversal["nodes"]:
        entity_type = node["type"]
        if entity_type == "person":
            print(f"- PERSON: {node['properties'].get('name')}")
        elif entity_type == "project":
            print(f"- PROJECT: {node['properties'].get('name')}")
        elif entity_type == "text":
            print(f"- TEXT: {node['properties'].get('title')}")
    
    print("\nConnections:")
    for edge in traversal["edges"]:
        source = graph.get_entity(edge["source"])
        target = graph.get_entity(edge["target"])
        
        source_name = source["properties"].get("name", source["properties"].get("title", "Unknown"))
        target_name = target["properties"].get("name", target["properties"].get("title", "Unknown"))
        
        print(f"- {source_name} {edge['type']} → {target_name}")
    
    # Get statistics
    print_separator("Graph Statistics")
    
    stats = graph.get_stats()
    print(f"Graph: {stats['graph_name']}")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total relations: {stats['total_relations']}")
    
    print("\nEntity types:")
    for entity_type, count in stats["entity_counts"].items():
        print(f"- {entity_type}: {count}")
    
    print("\nRelation types:")
    for relation_type, count in stats["relation_counts"].items():
        print(f"- {relation_type}: {count}")
    
    print_separator("Example Completed")
    
    # Cleanup if needed
    # If you want to delete the example data:
    # import shutil
    # shutil.rmtree("./example_data", ignore_errors=True)

if __name__ == "__main__":
    main() 