#!/usr/bin/env python
"""
Example demonstrating the centralized graph infrastructure with the long-term storage module.

This example will:
1. Initialize the graph-based memory adapter
2. Store facts and text
3. Query and retrieve information
4. Show how the graph represents the agent's memory

Usage:
    python example_graph_memory.py
"""

import os
import datetime
from pprint import pprint

# Import the memory adapter
from agent_suite.agents.storage.long_term.graphiti import GraphitiKnowledgeGraphMemoryAdapter

def print_separator(title):
    """Print a section separator with title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def main():
    """Run the graph memory example."""
    print_separator("Long-Term Graph Memory Example")
    
    # Initialize memory with file-based persistence
    memory = GraphitiKnowledgeGraphMemoryAdapter(
        graph_name="agent_memory_example",
        persist=True,
        persistence_path="./example_memory"
    )
    
    print(f"Initialized graph memory (persisted: {memory.graph_manager.persist})")
    
    # Store some facts about the world
    print_separator("Storing Facts")
    
    # Store facts about a person
    print("Storing facts about a person...")
    
    fact1_id = memory.store_fact(
        subject="person:alice",
        predicate="name",
        object_value="Alice Smith",
        metadata={"confidence": 0.95, "source": "user_introduction"}
    )
    
    fact2_id = memory.store_fact(
        subject="person:alice",
        predicate="occupation",
        object_value="Data Scientist",
        metadata={"confidence": 0.9, "source": "conversation"}
    )
    
    fact3_id = memory.store_fact(
        subject="person:alice",
        predicate="interested_in",
        object_value="knowledge_graphs",
        metadata={"confidence": 0.85, "source": "inferred"}
    )
    
    # Store facts about a topic
    print("Storing facts about a topic...")
    
    fact4_id = memory.store_fact(
        subject="topic:knowledge_graphs",
        predicate="description",
        object_value="A knowledge graph represents information as entities and their relationships",
        metadata={"confidence": 1.0, "source": "knowledge_base"}
    )
    
    # Store relationship between entities
    print("Storing relationships between entities...")
    
    fact5_id = memory.store_fact(
        subject="person:alice",
        predicate="KNOWS_ABOUT",
        object_value="topic:knowledge_graphs",
        metadata={"confidence": 0.9, "source": "conversation", "expertise_level": "advanced"}
    )
    
    # Store some conversation text
    print("\nStoring conversation snippets...")
    
    text1_id = memory.store_text(
        text="I've been working with graph databases for about 5 years now, primarily with Neo4j and Graphiti.",
        metadata={"speaker": "alice", "timestamp": datetime.datetime.now().isoformat(), "context": "introduction"}
    )
    
    text2_id = memory.store_text(
        text="The centralized graph infrastructure helps standardize how we store and query relationships.",
        metadata={"speaker": "alice", "timestamp": datetime.datetime.now().isoformat(), "context": "explanation"}
    )
    
    # Print stored IDs
    print(f"\nStored facts: {fact1_id}, {fact2_id}, {fact3_id}, {fact4_id}, {fact5_id}")
    print(f"Stored texts: {text1_id}, {text2_id}")
    
    # Query information
    print_separator("Querying Information")
    
    # Query facts by natural language
    print("Querying facts with natural language:")
    fact_results = memory.query_facts("Alice occupation", limit=3)
    
    for i, fact in enumerate(fact_results):
        print(f"\nFact {i+1}:")
        print(f"  Subject: {fact['subject']}")
        print(f"  Predicate: {fact['predicate']}")
        print(f"  Object: {fact['object']}")
        print(f"  Confidence: {fact['confidence']:.2f}")
        print(f"  Source: {fact['source']}")
    
    # Query using structured query
    print("\nStructured query for a specific entity:")
    alice_facts = memory.query_facts({"subject": "person:alice"})
    
    print(f"Found {len(alice_facts)} facts about Alice:")
    for fact in alice_facts:
        print(f"  - {fact['predicate']}: {fact['object']}")
    
    # Query text
    print("\nQuerying text about 'graph':")
    text_results = memory.query_text("graph", limit=2)
    
    for i, result in enumerate(text_results):
        print(f"\nText {i+1} [score: {result.get('score', 0):.2f}]:")
        print(f"  {result['content']}")
        if 'metadata' in result:
            print(f"  Speaker: {result['metadata'].get('speaker', 'unknown')}")
            print(f"  Context: {result['metadata'].get('context', 'unknown')}")
    
    # Get entity facts
    print_separator("Entity Information")
    
    print("Getting all facts about Alice:")
    all_alice_facts = memory.get_entity_facts("person:alice")
    
    print(f"Found {len(all_alice_facts)} facts (properties and relationships):")
    for fact in all_alice_facts:
        if fact["subject"] == "person:alice":
            print(f"  - Alice {fact['predicate']} {fact['object']}")
        else:
            print(f"  - {fact['subject']} {fact['predicate']} Alice")
    
    # Get statistics
    print_separator("Memory Statistics")
    
    stats = memory.get_stats()
    print(f"Graph: {stats['graph_name']}")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total relations: {stats['total_relations']}")
    
    print("\nEntity types:")
    for entity_type, count in stats.get("entity_counts", {}).items():
        print(f"  - {entity_type}: {count}")
    
    print("\nRelation types:")
    for relation_type, count in stats.get("relation_counts", {}).items():
        print(f"  - {relation_type}: {count}")
    
    print_separator("Example Completed")
    
    # Cleanup if needed
    # If you want to delete the example data:
    # import shutil
    # shutil.rmtree("./example_memory", ignore_errors=True)

if __name__ == "__main__":
    main() 