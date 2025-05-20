#!/usr/bin/env python
"""
Example script demonstrating the use of knowledge base implementations.

This script shows how to:
1. Create different types of knowledge bases
2. Add entities, relations, and text
3. Query the knowledge base
4. Perform semantic search
"""

import os
import json
from typing import Dict, Any

from knowledge.factory import KnowledgeBaseFactory
from knowledge.base import KnowledgeBase


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


def demo_knowledge_graph():
    """Demonstrate the knowledge graph."""
    print_separator("Knowledge Graph Demo")
    
    # Create a knowledge base instance
    knowledge_dir = os.path.join(os.getcwd(), "knowledge_data")
    os.makedirs(knowledge_dir, exist_ok=True)
    
    kb = KnowledgeBaseFactory.create_knowledge_base("graph", {
        "graph_name": "demo_graph",
        "persist": True,
        "persistence_path": knowledge_dir
    })
    
    # Add entities representing people
    print("Adding entities...")
    
    alice_id = kb.add_entity(
        entity_id="person-1",
        entity_type="person",
        properties={
            "name": "Alice Smith",
            "age": 32,
            "profession": "Data Scientist"
        }
    )
    
    bob_id = kb.add_entity(
        entity_id="person-2",
        entity_type="person",
        properties={
            "name": "Bob Johnson",
            "age": 45,
            "profession": "Software Engineer"
        }
    )
    
    # Add entities representing companies
    acme_id = kb.add_entity(
        entity_id="company-1",
        entity_type="company",
        properties={
            "name": "Acme Corp",
            "industry": "Technology",
            "founded": 1985
        }
    )
    
    globex_id = kb.add_entity(
        entity_id="company-2",
        entity_type="company",
        properties={
            "name": "Globex Inc",
            "industry": "Finance",
            "founded": 1992
        }
    )
    
    # Add relations
    print("Adding relations...")
    
    kb.add_relation(
        source_id="person-1",
        relation_type="works_at",
        target_id="company-1",
        properties={
            "start_date": "2018-05-15",
            "role": "Senior Data Scientist"
        }
    )
    
    kb.add_relation(
        source_id="person-2",
        relation_type="works_at",
        target_id="company-2",
        properties={
            "start_date": "2015-03-01",
            "role": "Engineering Manager"
        }
    )
    
    kb.add_relation(
        source_id="person-1",
        relation_type="knows",
        target_id="person-2",
        properties={
            "connection": "Professional",
            "years": 5
        }
    )
    
    # Query for people
    print("\nQuerying for people:")
    people = kb.query_entities(entity_type="person")
    pretty_print(people)
    
    # Query for relationships
    print("\nWhere does Alice work?")
    work_relations = kb.query_relations(source_id="person-1", relation_type="works_at")
    pretty_print(work_relations)
    
    # Find connections
    print("\nWho knows whom?")
    connections = kb.query_relations(relation_type="knows")
    pretty_print(connections)
    
    # Get stats
    print("\nKnowledge Graph Statistics:")
    stats = kb.get_stats()
    pretty_print(stats)


def demo_vector_database():
    """Demonstrate the vector database."""
    print_separator("Vector Database Demo")
    
    # Create a knowledge base instance
    knowledge_dir = os.path.join(os.getcwd(), "knowledge_data")
    os.makedirs(knowledge_dir, exist_ok=True)
    
    kb = KnowledgeBaseFactory.create_knowledge_base("vector", {
        "collection_name": "demo_vector",
        "embedding_model": "all-MiniLM-L6-v2",
        "persist": True,
        "persistence_path": knowledge_dir
    })
    
    # Add text documents
    print("Adding text documents...")
    
    python_doc = """Python is a high-level, interpreted programming language known for its readability and simplicity.
    It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    Python's design philosophy emphasizes code readability with its use of significant whitespace.
    It provides constructs that enable clear programming on both small and large scales."""
    
    kb.add_text(python_doc, {
        "title": "Python Overview",
        "category": "Programming Languages"
    })
    
    java_doc = """Java is a high-level, class-based, object-oriented programming language.
    It is designed to have as few implementation dependencies as possible.
    Java applications are typically compiled to bytecode that can run on any Java virtual machine (JVM)
    regardless of the underlying computer architecture."""
    
    kb.add_text(java_doc, {
        "title": "Java Overview",
        "category": "Programming Languages"
    })
    
    data_science_doc = """Data science is an interdisciplinary field that uses scientific methods, processes,
    algorithms and systems to extract knowledge and insights from structured and unstructured data.
    It employs techniques and theories drawn from many fields within mathematics, statistics, computer science,
    domain knowledge, and information science."""
    
    kb.add_text(data_science_doc, {
        "title": "Data Science Overview",
        "category": "Data Science"
    })
    
    # Perform semantic search
    print("\nSearching for 'Python programming language':")
    python_results = kb.semantic_search("Python programming language")
    pretty_print(python_results)
    
    print("\nSearching for 'data analysis and machine learning':")
    data_results = kb.semantic_search("data analysis and machine learning")
    pretty_print(data_results)
    
    print("\nSearching for 'object-oriented development':")
    oop_results = kb.semantic_search("object-oriented development")
    pretty_print(oop_results)
    
    # Get stats
    print("\nVector Database Statistics:")
    stats = kb.get_stats()
    pretty_print(stats)


def demo_combined_knowledge_base():
    """Demonstrate the combined knowledge base."""
    print_separator("Combined Knowledge Base Demo")
    
    # Create a knowledge base instance
    knowledge_dir = os.path.join(os.getcwd(), "knowledge_data")
    os.makedirs(knowledge_dir, exist_ok=True)
    
    kb = KnowledgeBaseFactory.create_knowledge_base("combined", {
        "kg_graph_name": "demo_combined_graph",
        "vector_collection_name": "demo_combined_vector",
        "persist": True,
        "persistence_path": knowledge_dir
    })
    
    # Add entities
    print("Adding entities...")
    
    # Add a project entity
    project_id = kb.add_entity(
        entity_id="project-1",
        entity_type="project",
        properties={
            "name": "Knowledge Graph System",
            "start_date": "2023-01-15",
            "status": "In Progress"
        }
    )
    
    # Add a technology entity
    tech_id = kb.add_entity(
        entity_id="tech-1",
        entity_type="technology",
        properties={
            "name": "Qdrant",
            "category": "Vector Database",
            "version": "1.0.0"
        }
    )
    
    # Add a relation
    kb.add_relation(
        source_id="project-1",
        relation_type="uses",
        target_id="tech-1",
        properties={
            "purpose": "Semantic Search"
        }
    )
    
    # Add text documents
    print("Adding text documents...")
    
    kb.add_text("""
    Qdrant is a vector similarity search engine and vector database.
    It provides a production-ready service with a convenient API to store, search, and manage
    vectors with an additional payload. Qdrant is tailored to extended filtering support.
    It makes it useful for all sorts of neural-network or semantic-based matching applications.
    """, {
        "title": "Qdrant Overview",
        "type": "documentation"
    })
    
    kb.add_text("""
    A knowledge graph is a knowledge base that uses a graph-structured data model or topology
    to integrate data. Knowledge graphs are often used to store interlinked descriptions of entities
    with free-form semantics. They have become especially popular since 2012, when Google announced
    their Knowledge Graph project.
    """, {
        "title": "Knowledge Graph Overview",
        "type": "documentation"
    })
    
    # Query entities
    print("\nQuery for projects:")
    projects = kb.query_entities(entity_type="project")
    pretty_print(projects)
    
    # Query relations
    print("\nWhat technologies does the project use?")
    tech_relations = kb.query_relations(source_id="project-1", relation_type="uses")
    pretty_print(tech_relations)
    
    # Semantic search
    print("\nSemantic search for 'vector database for similarity search':")
    search_results = kb.semantic_search("vector database for similarity search")
    pretty_print(search_results)
    
    print("\nSemantic search for 'knowledge representation using graphs':")
    kg_results = kb.semantic_search("knowledge representation using graphs")
    pretty_print(kg_results)
    
    # Get stats
    print("\nCombined Knowledge Base Statistics:")
    stats = kb.get_stats()
    pretty_print(stats)


def demo_knowledge_creation_workflow():
    """Demonstrate a typical knowledge creation workflow."""
    print_separator("Knowledge Creation Workflow")
    
    # Create a combined knowledge base for the demo
    knowledge_dir = os.path.join(os.getcwd(), "knowledge_data")
    os.makedirs(knowledge_dir, exist_ok=True)
    
    kb = KnowledgeBaseFactory.create_knowledge_base("combined", {
        "kg_graph_name": "workflow_demo",
        "vector_collection_name": "workflow_demo_vector",
        "persist": True,
        "persistence_path": knowledge_dir
    })
    
    # Step 1: Create a domain model with entities and relations
    print("Step 1: Creating a domain model")
    
    # Add domain concepts
    concepts = [
        ("concept-1", "Machine Learning", {"definition": "Field of study that gives computers the ability to learn without being explicitly programmed."}),
        ("concept-2", "Neural Network", {"definition": "Computing system inspired by biological neural networks that constitute animal brains."}),
        ("concept-3", "Data Science", {"definition": "Interdisciplinary field that uses scientific methods to extract knowledge from data."}),
        ("concept-4", "Vector Database", {"definition": "Storage system optimized for high-dimensional vectors and similarity search."})
    ]
    
    for concept_id, name, properties in concepts:
        kb.add_entity(
            entity_id=concept_id,
            entity_type="concept",
            properties={**properties, "name": name}
        )
    
    # Add relationships between concepts
    relations = [
        ("concept-1", "related_to", "concept-3", {"strength": 0.8}),
        ("concept-1", "uses", "concept-2", {"context": "For complex learning tasks"}),
        ("concept-3", "uses", "concept-4", {"context": "For semantic search and recommendations"})
    ]
    
    for source, relation, target, properties in relations:
        kb.add_relation(source, relation, target, properties)
    
    # Step 2: Add detailed text documents about each concept
    print("\nStep 2: Adding detailed knowledge")
    
    kb.add_text("""
    Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use
    of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.
    IBM has a rich history with machine learning. One of its own, Arthur Samuel, is credited for coining the term,
    "machine learning" with his research around the game of checkers. Robert Nealey, the self-proclaimed
    checkers master, played the game on an IBM 7094 computer in 1962, and he lost to the computer.
    Compared to what can be done today, this feat seems minor, but it's considered a major milestone in
    the field of artificial intelligence.
    """, {
        "title": "Machine Learning",
        "source": "Introduction to ML",
        "related_concept": "concept-1"
    })
    
    kb.add_text("""
    Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs),
    are a subset of machine learning and are at the heart of deep learning algorithms. Their name and
    structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.
    Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or
    more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has
    an associated weight and threshold. If the output of any individual node is above the specified threshold
    value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed
    along to the next layer of the network.
    """, {
        "title": "Neural Networks",
        "source": "Deep Learning Concepts",
        "related_concept": "concept-2"
    })
    
    # Step 3: Query the knowledge base
    print("\nStep 3: Querying the knowledge base")
    
    print("\nFind all concepts:")
    concepts = kb.query_entities(entity_type="concept")
    pretty_print(concepts)
    
    print("\nFind relationships between concepts:")
    concept_relations = kb.query_relations()
    pretty_print(concept_relations)
    
    print("\nSearch for 'neural networks in deep learning':")
    neural_results = kb.semantic_search("neural networks in deep learning")
    pretty_print(neural_results)
    
    # Step 4: Update and enhance the knowledge
    print("\nStep 4: Updating knowledge")
    
    # Update concept with new information
    kb.update_entity("concept-2", {
        "frameworks": ["TensorFlow", "PyTorch", "Keras"],
        "popularity": "Very High"
    })
    
    # Add a new concept that builds on existing knowledge
    kb.add_entity(
        entity_id="concept-5",
        entity_type="concept",
        properties={
            "name": "Transformer Models",
            "definition": "Neural network architecture using self-attention mechanisms.",
            "examples": ["BERT", "GPT", "T5"]
        }
    )
    
    # Link new concept to existing ones
    kb.add_relation(
        source_id="concept-5",
        relation_type="is_a",
        target_id="concept-2",
        properties={"context": "Advanced architecture"}
    )
    
    # Get updated stats
    print("\nFinal Knowledge Base Statistics:")
    stats = kb.get_stats()
    pretty_print(stats)


def main():
    """Run the knowledge base demonstrations."""
    print("Knowledge Base Examples")
    print("----------------------")
    print("This script demonstrates using different knowledge base implementations.")
    
    # Run the demos
    demo_knowledge_graph()
    demo_vector_database()
    demo_combined_knowledge_base()
    demo_knowledge_creation_workflow()
    
    print_separator("Demo Complete")
    print("The knowledge data is stored in ./knowledge_data directory.")


if __name__ == "__main__":
    main() 