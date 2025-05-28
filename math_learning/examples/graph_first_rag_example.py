"""
Graph-First RAG Integration Example for Math Learning

This example demonstrates the graph-first RAG approach specifically designed 
for the math learning domain. 

IMPORTANT: RAG systems are not inherently "graph-first." They can be:
- Vector-first (e.g., for document search and semantic similarity)
- Graph-first (e.g., for relationship-heavy domains like math learning)  
- Key-value-first (e.g., for session management and fast lookups)

For math learning, we choose graph-first because:
- Math concepts have clear prerequisite relationships
- Learning paths are essentially graph traversals
- Knowledge dependencies are naturally graph-structured
- Student progress tracking benefits from relationship analysis

This example shows how to configure and use the RAG system with graph storage
as the primary mechanism, with optional vector and key-value enhancements.
"""

import asyncio
import logging
from typing import Dict, Any

from math_learning.config.rag_config import (
    RagConfig, 
    get_memory_config, 
    get_local_config,
    create_configured_rag_service
)
from math_learning.knowledge_graph.graph_rag_algebra_graph import (
    GraphRagAlgebraGraph,
    create_graph_rag_algebra_graph
)
from agents.rag.middleware.storage_router import StorageType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_graph_first_approach():
    """Demonstrate the graph-first RAG approach."""
    print("=== Graph-First RAG Math Learning Demo ===\n")
    
    # Configuration 1: Graph storage only (minimal RAG)
    print("1. Graph Storage Only Configuration:")
    graph_only_config = RagConfig(
        primary_storage_type="graph",
        graph_storage_type="memory",
        vector_storage_type="memory",  # Available but not primary
        kv_storage_type="memory",      # Available but not primary
        embedding_provider="dummy"     # Minimal embedding for testing
    )
    
    await demo_configuration(graph_only_config, "Graph-Only")
    
    print("\n" + "="*60 + "\n")
    
    # Configuration 2: Graph + Vector for semantic search
    print("2. Graph + Vector Configuration:")
    graph_vector_config = RagConfig(
        primary_storage_type="graph",
        graph_storage_type="memory",
        vector_storage_type="memory",
        kv_storage_type="memory",
        embedding_provider="sentence-transformer"
    )
    
    await demo_configuration(graph_vector_config, "Graph+Vector")
    
    print("\n" + "="*60 + "\n")
    
    # Configuration 3: Full RAG with all storage types
    print("3. Full RAG Configuration:")
    full_rag_config = get_memory_config()  # All storage types enabled
    
    await demo_configuration(full_rag_config, "Full-RAG")


async def demo_configuration(config: RagConfig, config_name: str):
    """Demo a specific RAG configuration."""
    print(f"Configuration: {config_name}")
    print(f"Primary Storage: {config.primary_storage_type}")
    print(f"Graph Storage: {config.graph_storage_type}")
    print(f"Vector Storage: {config.vector_storage_type}")
    print(f"KV Storage: {config.kv_storage_type}")
    print(f"Embedding Provider: {config.embedding_provider}")
    print()
    
    try:
        # Create RAG service with the configuration
        rag_service = await create_configured_rag_service(config)
        
        # Check which storage types are available
        available_storage = []
        if rag_service.has_storage_adaptor(StorageType.GRAPH):
            available_storage.append("Graph")
        if rag_service.has_storage_adaptor(StorageType.VECTOR):
            available_storage.append("Vector")
        if rag_service.has_storage_adaptor(StorageType.KEY_VALUE):
            available_storage.append("Key-Value")
        
        print(f"Available Storage Types: {', '.join(available_storage)}")
        
        # Create the algebra graph
        algebra_graph = GraphRagAlgebraGraph(rag_service, config)
        await algebra_graph.initialize()
        
        print("✓ Algebra knowledge graph initialized successfully")
        
        # Test basic graph operations
        await test_graph_operations(algebra_graph)
        
        # Test storage-specific features
        await test_storage_features(algebra_graph, rag_service)
        
    except Exception as e:
        print(f"✗ Error with {config_name} configuration: {e}")


async def test_graph_operations(graph: GraphRagAlgebraGraph):
    """Test basic graph operations."""
    print("\nTesting Graph Operations:")
    
    # Test concept retrieval
    concept = await graph.get_concept("linear_equations_one_variable")
    if concept:
        print(f"✓ Retrieved concept: {concept.name}")
    else:
        print("✗ Failed to retrieve concept")
    
    # Test prerequisite relationships
    prereqs = await graph.get_prerequisites("quadratic_functions")
    print(f"✓ Prerequisites for quadratic functions: {len(prereqs)} found")
    for prereq in prereqs[:3]:  # Show first 3
        print(f"  - {prereq.name}")
    
    # Test learning path
    try:
        path = await graph.find_learning_path("integers", "linear_functions")
        print(f"✓ Learning path found: {len(path)} steps")
        path_names = " → ".join([c.name for c in path[:4]])  # Show first 4
        if len(path) > 4:
            path_names += " → ..."
        print(f"  Path: {path_names}")
    except Exception as e:
        print(f"✗ Learning path error: {e}")


async def test_storage_features(graph: GraphRagAlgebraGraph, rag_service):
    """Test storage-specific features."""
    print("\nTesting Storage Features:")
    
    # Test graph storage (always available)
    print("Graph Storage:")
    concepts_by_category = await graph.get_concepts_by_category("Functions")
    print(f"✓ Functions category: {len(concepts_by_category)} concepts")
    
    # Test vector storage (if available)
    if rag_service.has_storage_adaptor(StorageType.VECTOR):
        print("Vector Storage:")
        try:
            search_results = await graph.search_concepts("solving equations")
            print(f"✓ Semantic search: {len(search_results)} results")
        except Exception as e:
            print(f"✗ Semantic search error: {e}")
    else:
        print("Vector Storage: Not configured")
    
    # Test key-value storage (if available)
    if rag_service.has_storage_adaptor(StorageType.KEY_VALUE):
        print("Key-Value Storage:")
        try:
            kv_storage = rag_service.get_storage_adaptor(StorageType.KEY_VALUE)
            # Test storing and retrieving metadata
            await kv_storage.store("test_key", {"test": "value"})
            result = await kv_storage.retrieve("test_key")
            if result:
                print("✓ Key-value operations working")
            else:
                print("✗ Key-value retrieval failed")
        except Exception as e:
            print(f"✗ Key-value error: {e}")
    else:
        print("Key-Value Storage: Not configured")


async def demonstrate_configuration_flags():
    """Demonstrate how configuration flags control storage usage."""
    print("\n=== Configuration Flags Demo ===\n")
    
    configurations = [
        {
            "name": "Graph-Only",
            "config": RagConfig(
                primary_storage_type="graph",
                graph_storage_type="memory",
                vector_storage_type="memory",
                kv_storage_type="memory",
                embedding_provider="dummy"
            ),
            "description": "Uses only graph storage for core operations"
        },
        {
            "name": "Graph+Vector",
            "config": RagConfig(
                primary_storage_type="graph",
                graph_storage_type="memory",
                vector_storage_type="memory",
                kv_storage_type="memory",
                embedding_provider="sentence-transformer"
            ),
            "description": "Graph storage + vector search capabilities"
        },
        {
            "name": "Production-Ready",
            "config": RagConfig(
                primary_storage_type="graph",
                graph_storage_type="memory",  # Would be "neo4j" in production
                vector_storage_type="memory",  # Would be "qdrant" in production
                kv_storage_type="memory",      # Would be "redis" in production
                embedding_provider="sentence-transformer"  # Would be "openai" in production
            ),
            "description": "All storage types with production-ready providers"
        }
    ]
    
    for config_info in configurations:
        print(f"Configuration: {config_info['name']}")
        print(f"Description: {config_info['description']}")
        print(f"Primary Storage: {config_info['config'].primary_storage_type}")
        print(f"Flags: graph={config_info['config'].graph_storage_type}, "
              f"vector={config_info['config'].vector_storage_type}, "
              f"kv={config_info['config'].kv_storage_type}")
        print()


async def main():
    """Main demo function."""
    try:
        await demonstrate_graph_first_approach()
        await demonstrate_configuration_flags()
        
        print("\n=== Summary ===")
        print("✓ Graph-first RAG approach demonstrated")
        print("✓ Configuration flags control storage usage")
        print("✓ RAG system's graph storage used as primary mechanism")
        print("✓ Vector and key-value storage provide additional capabilities")
        print("\nThe math learning system now uses the RAG system's graph storage")
        print("as the foundation, with flags controlling which additional storage")
        print("types are enabled for enhanced functionality.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 