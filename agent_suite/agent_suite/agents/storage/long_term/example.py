#!/usr/bin/env python
"""
Example script demonstrating the use of long-term memory implementations.

This script shows how to:
1. Create different types of long-term memory
2. Store facts and unstructured text
3. Query the memory to retrieve relevant information
4. Integrate memory with an agent
"""

import os
import json
from typing import Dict, Any

from agent_suite.agents.storage.long_term.factory import LongTermMemoryFactory
from agent_suite.agents.storage.long_term.base import LongTermMemoryBase


def print_separator(title: str = None):
    """Print a separator line with optional title."""
    width = 80
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - 12 - len(title)) + "\n")
    else:
        print("\n" + "=" * width + "\n")


def pretty_print(data: Any):
    """Pretty print data structures."""
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2))
    else:
        print(data)


def demo_knowledge_graph_memory():
    """Demonstrate the knowledge graph memory."""
    print_separator("Knowledge Graph Memory Demo")
    
    # Create a memory instance
    memory_dir = os.path.join(os.getcwd(), "memory")
    os.makedirs(memory_dir, exist_ok=True)
    
    memory = LongTermMemoryFactory.create_memory("kg", {
        "graph_name": "demo_kg",
        "persist": True,
        "persistence_path": memory_dir
    })
    
    # Store facts about fruits
    print("Storing facts about fruits...")
    apple_fact = memory.store_fact(
        subject="apple",
        predicate="is_a",
        object_value="fruit",
        metadata={"confidence": 0.95, "source": "knowledge_base"}
    )
    
    memory.store_fact(
        subject="apple",
        predicate="color",
        object_value="red",
        metadata={"confidence": 0.9, "source": "knowledge_base"}
    )
    
    memory.store_fact(
        subject="apple",
        predicate="grows_on",
        object_value="tree",
        metadata={"confidence": 0.95, "source": "knowledge_base"}
    )
    
    memory.store_fact(
        subject="banana",
        predicate="is_a",
        object_value="fruit",
        metadata={"confidence": 0.95, "source": "knowledge_base"}
    )
    
    memory.store_fact(
        subject="banana",
        predicate="color",
        object_value="yellow",
        metadata={"confidence": 0.9, "source": "knowledge_base"}
    )
    
    # Query for facts
    print("\nQuerying for fruits:")
    fruits = memory.query_facts("fruit")
    pretty_print(fruits)
    
    print("\nQuerying for apple:")
    apple_facts = memory.query_facts({"subject": "apple"})
    pretty_print(apple_facts)
    
    print("\nQuerying for colors:")
    color_facts = memory.query_facts({"predicate": "color"})
    pretty_print(color_facts)
    
    # Get all entities
    print("\nAll entities:")
    entities = memory.get_entities()
    pretty_print(entities)
    
    # Get facts about an entity
    print("\nAll facts about 'apple':")
    apple_all_facts = memory.get_entity_facts("apple")
    pretty_print(apple_all_facts)
    
    # Get memory stats
    print("\nMemory statistics:")
    stats = memory.get_stats()
    pretty_print(stats)


def demo_vector_memory():
    """Demonstrate the vector database memory."""
    print_separator("Vector Database Memory Demo")
    
    # Create a memory instance
    memory = LongTermMemoryFactory.create_memory("vector", {
        "collection_name": "demo_vector",
        "embedding_model": "all-MiniLM-L6-v2"
    })
    
    # Store text passages
    print("Storing text passages...")
    
    python_text = """Python is a high-level, interpreted programming language that was first released in 1991.
    It emphasizes code readability with its notable use of significant whitespace.
    Python features a dynamic type system and automatic memory management and it supports multiple programming
    paradigms, including object-oriented, imperative, functional, and procedural."""
    
    java_text = """Java is a high-level, class-based, object-oriented programming language that was first released in 1995.
    It is designed to have as few implementation dependencies as possible, allowing 'write once, run anywhere' functionality.
    Java applications are typically compiled to bytecode that can run on any Java virtual machine (JVM)
    regardless of the underlying computer architecture."""
    
    ml_text = """Machine learning is the field of study that gives computers the ability to learn without being explicitly
    programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.
    The process of learning begins with observations or data, such as examples, direct experience, or instruction."""
    
    memory.store_text(python_text, {"topic": "programming", "language": "Python"})
    memory.store_text(java_text, {"topic": "programming", "language": "Java"})
    memory.store_text(ml_text, {"topic": "AI", "subtopic": "machine learning"})
    
    # Query for similar text
    print("\nQuerying for 'Python programming':")
    python_results = memory.query_text("Python programming")
    pretty_print(python_results)
    
    print("\nQuerying for 'Java virtual machine':")
    java_results = memory.query_text("Java virtual machine")
    pretty_print(java_results)
    
    print("\nQuerying for 'Machine learning algorithms':")
    ml_results = memory.query_text("Machine learning algorithms")
    pretty_print(ml_results)
    
    # Get memory stats
    print("\nMemory statistics:")
    stats = memory.get_stats()
    pretty_print(stats)


def demo_combined_memory():
    """Demonstrate the combined memory."""
    print_separator("Combined Memory Demo")
    
    # Create a memory instance
    memory_dir = os.path.join(os.getcwd(), "memory")
    os.makedirs(memory_dir, exist_ok=True)
    
    memory = LongTermMemoryFactory.create_memory("combined", {
        "kg_graph_name": "demo_combined_kg",
        "vector_collection_name": "demo_combined_vector",
        "persistence_path": memory_dir
    })
    
    # Store facts
    print("Storing facts...")
    
    memory.store_fact(
        subject="Python",
        predicate="is_a",
        object_value="programming_language"
    )
    
    memory.store_fact(
        subject="Python",
        predicate="created_by",
        object_value="Guido van Rossum"
    )
    
    memory.store_fact(
        subject="Java",
        predicate="is_a",
        object_value="programming_language"
    )
    
    memory.store_fact(
        subject="Java",
        predicate="created_by",
        object_value="James Gosling"
    )
    
    # Store text
    print("Storing text passages...")
    
    python_info = """Python is a high-level, general-purpose programming language. Its design philosophy 
    emphasizes code readability with the use of significant indentation. Python is dynamically-typed and 
    garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, 
    and functional programming."""
    
    java_info = """Java is a high-level, class-based, object-oriented programming language that is designed 
    to have as few implementation dependencies as possible. It is a general-purpose programming language 
    intended to let programmers write once, run anywhere, meaning that compiled Java code can run on all 
    platforms that support Java without the need for recompilation."""
    
    memory.store_text(python_info, {"subject": "Python", "type": "description"})
    memory.store_text(java_info, {"subject": "Java", "type": "description"})
    
    # Query facts
    print("\nQuerying facts about programming languages:")
    lang_facts = memory.query_facts({"predicate": "is_a", "object": "programming_language"})
    pretty_print(lang_facts)
    
    print("\nQuerying facts about creators:")
    creator_facts = memory.query_facts({"predicate": "created_by"})
    pretty_print(creator_facts)
    
    # Query text
    print("\nQuerying text about Python:")
    python_results = memory.query_text("Python programming language features")
    pretty_print(python_results)
    
    print("\nQuerying text about Java:")
    java_results = memory.query_text("Java platforms and portability")
    pretty_print(java_results)
    
    # Get memory stats
    print("\nMemory statistics:")
    stats = memory.get_stats()
    pretty_print(stats)


def demonstrate_memory_integration_with_agent():
    """Show how to integrate memory with an agent (conceptual example)."""
    print_separator("Memory Integration with Agent")
    
    print("""
    # Create the memory
    memory = LongTermMemoryFactory.create_memory("combined", {
        "persistence_path": "./agent_memory"  
    })

    # Create an agent with long-term memory
    agent = StorageAwareReActAgent(
        long_term_memory=memory
    )

    # When processing user input, fetch relevant context
    user_input = "Tell me more about machine learning"
    
    # Get relevant facts
    facts = memory.query_facts("machine learning")
    
    # Get relevant text
    context = memory.query_text(user_input)
    
    # Inject into agent prompt
    response = agent.respond(user_input, context=context, facts=facts)
    
    # Store new information from conversation
    if "neural networks" in user_input:
        memory.store_fact(
            subject="neural_networks",
            predicate="is_part_of",
            object_value="machine_learning"
        )
        
    memory.store_text(
        f"User asked about: {user_input}",
        metadata={"type": "conversation", "timestamp": "now"}
    )
    """)


def main():
    """Run the demo."""
    print("Long-Term Memory Examples")
    print("------------------------")
    print("This script demonstrates using different long-term memory implementations.")
    
    # Run the demos
    demo_knowledge_graph_memory()
    demo_vector_memory()
    demo_combined_memory()
    demonstrate_memory_integration_with_agent()
    
    print_separator("Demo Complete")
    print("The demo memory is stored in ./memory directory.")


if __name__ == "__main__":
    main() 