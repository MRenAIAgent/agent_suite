"""
Hybrid Knowledge System Benchmarks

This module implements benchmarks for testing the hybrid knowledge system
that combines graph databases and vector databases.
"""

import os
import time
import json
import logging
import random
from typing import Dict, List, Any, Optional

from agent_suite.graph import GraphManager
from knowledge.graphiti import GraphitiKnowledgeGraphAdapter
from knowledge.vector import QdrantVectorDatabase
from hybrid_knowledge_example import HybridKnowledgeManager

from benchmark_suite.metrics import BenchmarkResult


logger = logging.getLogger("benchmark_suite.hybrid")


def benchmark_hybrid_document_processing(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark document processing in hybrid knowledge system.
    
    This measures the performance of processing documents through
    the complete hybrid pipeline including:
    - Vector embedding and storage
    - Knowledge extraction
    - Graph structure creation
    - Cross-referencing
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load documents from dataset
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Set up a fresh hybrid knowledge manager
    hybrid_manager = HybridKnowledgeManager(
        graph_name=f"benchmark_hybrid_docs_{int(time.time())}",
        vector_collection=f"benchmark_hybrid_docs_{int(time.time())}",
        persist=False  # Don't persist for benchmarks
    )
    
    # Sample documents for benchmarking
    random.shuffle(documents)
    num_documents = min(config.get("num_documents", 20), len(documents))
    sample_documents = documents[:num_documents]
    
    # Measure initial memory
    start_memory = _get_process_memory_usage()
    
    # Benchmark document processing
    processing_times = []
    doc_ids = []
    
    for document in sample_documents:
        start_time = time.time()
        doc_id = hybrid_manager.process_document(
            document_text=document["content"],
            document_metadata=document["properties"]
        )
        end_time = time.time()
        
        processing_times.append((end_time - start_time) * 1000)  # ms
        doc_ids.append(doc_id)
    
    # Measure final memory
    end_memory = _get_process_memory_usage()
    
    # Calculate metrics
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    throughput = num_documents / (sum(processing_times) / 1000) if sum(processing_times) > 0 else 0
    memory_usage = end_memory - start_memory
    
    # Update result
    result.throughput_ops = throughput
    result.memory_mb = memory_usage
    
    # Add custom metrics
    result.custom_metrics["avg_processing_time_ms"] = avg_processing_time
    result.custom_metrics["processing_times_ms"] = processing_times
    
    # Get structure statistics
    graph_stats = hybrid_manager.graph_db.get_stats()
    vector_stats = hybrid_manager.vector_db.get_stats()
    
    result.custom_metrics["graph_entities"] = graph_stats.get("total_entities", 0)
    result.custom_metrics["graph_relations"] = graph_stats.get("total_relations", 0)
    result.custom_metrics["vector_count"] = vector_stats.get("vectors_count", 0)
    
    # Cleanup
    del hybrid_manager


def benchmark_hybrid_search(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark hybrid search across both graph and vector databases.
    
    This measures the performance of combined search queries across
    both knowledge storage types.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load documents from dataset
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Set up and populate hybrid knowledge manager
    hybrid_manager = HybridKnowledgeManager(
        graph_name=f"benchmark_hybrid_search_{int(time.time())}",
        vector_collection=f"benchmark_hybrid_search_{int(time.time())}",
        persist=False  # Don't persist for benchmarks
    )
    
    # Sample documents for loading
    random.shuffle(documents)
    num_documents = min(20, len(documents))
    sample_documents = documents[:num_documents]
    
    # Load documents (not part of the benchmark)
    doc_ids = []
    for document in sample_documents:
        doc_id = hybrid_manager.process_document(
            document_text=document["content"],
            document_metadata=document["properties"]
        )
        doc_ids.append(doc_id)
    
    # Prepare search queries
    search_queries = [
        "cloud architecture and infrastructure",
        "machine learning and data science",
        "project management best practices",
        "software development methodology",
        "team collaboration tools",
        "security and compliance requirements",
        "performance optimization techniques",
        "user experience design principles",
        "database management systems",
        "api integration strategies"
    ]
    
    # Benchmark hybrid search
    search_times = []
    vector_result_counts = []
    graph_result_counts = []
    
    for query in search_queries:
        start_time = time.time()
        results = hybrid_manager.hybrid_search(query=query, limit=10)
        end_time = time.time()
        
        search_time = (end_time - start_time) * 1000  # ms
        search_times.append(search_time)
        
        vector_results = results.get("vector_results", [])
        graph_results = results.get("graph_results", [])
        
        vector_result_counts.append(len(vector_results))
        graph_result_counts.append(len(graph_results))
    
    # Calculate metrics
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0
    avg_vector_count = sum(vector_result_counts) / len(vector_result_counts) if vector_result_counts else 0
    avg_graph_count = sum(graph_result_counts) / len(graph_result_counts) if graph_result_counts else 0
    
    # Add custom metrics
    result.custom_metrics["avg_search_time_ms"] = avg_search_time
    result.custom_metrics["search_times_ms"] = search_times
    result.custom_metrics["avg_vector_results"] = avg_vector_count
    result.custom_metrics["avg_graph_results"] = avg_graph_count
    
    # Cleanup
    del hybrid_manager


def benchmark_hybrid_knowledge_queries(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark specialized knowledge queries in the hybrid system.
    
    This measures the performance of complex knowledge queries
    such as finding related entities, documents mentioning an entity, etc.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load data from dataset
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Set up and populate hybrid knowledge manager
    hybrid_manager = HybridKnowledgeManager(
        graph_name=f"benchmark_hybrid_queries_{int(time.time())}",
        vector_collection=f"benchmark_hybrid_queries_{int(time.time())}",
        persist=False  # Don't persist for benchmarks
    )
    
    # Sample documents for loading
    random.shuffle(documents)
    num_documents = min(20, len(documents))
    sample_documents = documents[:num_documents]
    
    # Load documents (not part of the benchmark)
    doc_ids = []
    for document in sample_documents:
        doc_id = hybrid_manager.process_document(
            document_text=document["content"],
            document_metadata=document["properties"]
        )
        doc_ids.append(doc_id)
    
    # Get a list of entities in the graph
    person_entities = hybrid_manager.graph_db.query_entities(entity_type="person")
    org_entities = hybrid_manager.graph_db.query_entities(entity_type="organization")
    
    entities = person_entities + org_entities
    
    # If no entities found, use doc_ids
    if not entities and doc_ids:
        # Use documents as entities
        entities = [{"id": doc_id} for doc_id in doc_ids]
    
    if not entities:
        logger.warning("No entities found for hybrid query benchmark")
        return
    
    # Prepare different query types
    query_types = [
        "related_entities",
        "document_entities",
        "entity_documents"
    ]
    
    # Benchmark knowledge queries
    query_times = {query_type: [] for query_type in query_types}
    
    # 1. Related entities queries
    for _ in range(min(10, len(entities))):
        entity = random.choice(entities)
        entity_id = entity["id"]
        
        start_time = time.time()
        hybrid_manager.knowledge_query(
            query_type="related_entities",
            params={"entity_id": entity_id}
        )
        end_time = time.time()
        
        query_times["related_entities"].append((end_time - start_time) * 1000)  # ms
    
    # 2. Document entities queries
    for _ in range(min(10, len(doc_ids))):
        doc_id = random.choice(doc_ids)
        
        start_time = time.time()
        hybrid_manager.knowledge_query(
            query_type="document_entities",
            params={"doc_id": doc_id}
        )
        end_time = time.time()
        
        query_times["document_entities"].append((end_time - start_time) * 1000)  # ms
    
    # 3. Entity documents queries
    for _ in range(min(10, len(entities))):
        entity = random.choice(entities)
        entity_id = entity["id"]
        
        start_time = time.time()
        hybrid_manager.knowledge_query(
            query_type="entity_documents",
            params={"entity_id": entity_id}
        )
        end_time = time.time()
        
        query_times["entity_documents"].append((end_time - start_time) * 1000)  # ms
    
    # Calculate average query times
    avg_times = {}
    for query_type, times in query_times.items():
        if times:
            avg_times[query_type] = sum(times) / len(times)
    
    # Add custom metrics
    result.custom_metrics["query_times"] = avg_times
    
    # Cleanup
    del hybrid_manager


def benchmark_hybrid_system_comparison(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Compare performance between hybrid system vs. individual components.
    
    This benchmark compares the performance of:
    1. Hybrid system (combined graph + vector)
    2. Graph database only
    3. Vector database only
    
    for similar operations to quantify the overhead and benefits.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load documents from dataset
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Sample documents for testing
    random.shuffle(documents)
    num_documents = min(10, len(documents))
    sample_documents = documents[:num_documents]
    
    # Timestamp for unique identifiers
    timestamp = int(time.time())
    
    # 1. Set up individual components
    graph_db = GraphitiKnowledgeGraphAdapter(
        graph_name=f"benchmark_comp_graph_{timestamp}",
        persist=False
    )
    
    vector_db = QdrantVectorDatabase(
        collection_name=f"benchmark_comp_vector_{timestamp}",
        persist=False
    )
    
    # 2. Set up hybrid system
    hybrid_manager = HybridKnowledgeManager(
        graph_name=f"benchmark_comp_hybrid_{timestamp}",
        vector_collection=f"benchmark_comp_hybrid_{timestamp}",
        persist=False
    )
    
    # Measure document processing times
    doc_times = {
        "hybrid": [],
        "graph": [],
        "vector": []
    }
    
    # Process documents in all systems
    for document in sample_documents:
        # Hybrid system
        start_time = time.time()
        hybrid_manager.process_document(
            document_text=document["content"],
            document_metadata=document["properties"]
        )
        end_time = time.time()
        doc_times["hybrid"].append((end_time - start_time) * 1000)
        
        # Graph database only
        start_time = time.time()
        doc_id = graph_db.add_text(
            text=document["content"],
            properties=document["properties"]
        )
        end_time = time.time()
        doc_times["graph"].append((end_time - start_time) * 1000)
        
        # Vector database only
        start_time = time.time()
        vector_db.add_text(
            text=document["content"],
            properties=document["properties"]
        )
        end_time = time.time()
        doc_times["vector"].append((end_time - start_time) * 1000)
    
    # Prepare search queries
    search_queries = [
        "cloud architecture",
        "project management",
        "team collaboration",
        "data analytics",
        "software development"
    ]
    
    # Measure search times
    search_times = {
        "hybrid": [],
        "graph": [],
        "vector": []
    }
    
    for query in search_queries:
        # Hybrid search
        start_time = time.time()
        hybrid_manager.hybrid_search(query=query, limit=10)
        end_time = time.time()
        search_times["hybrid"].append((end_time - start_time) * 1000)
        
        # Graph search
        start_time = time.time()
        graph_db.semantic_search(query=query, limit=10)
        end_time = time.time()
        search_times["graph"].append((end_time - start_time) * 1000)
        
        # Vector search
        start_time = time.time()
        vector_db.semantic_search(query=query, limit=10)
        end_time = time.time()
        search_times["vector"].append((end_time - start_time) * 1000)
    
    # Calculate average times
    avg_doc_times = {}
    for system, times in doc_times.items():
        if times:
            avg_doc_times[system] = sum(times) / len(times)
    
    avg_search_times = {}
    for system, times in search_times.items():
        if times:
            avg_search_times[system] = sum(times) / len(times)
    
    # Calculate relative overhead percentages
    doc_overhead = {}
    if "vector" in avg_doc_times and avg_doc_times["vector"] > 0:
        doc_overhead["hybrid_vs_vector"] = (avg_doc_times["hybrid"] / avg_doc_times["vector"] - 1) * 100
    
    if "graph" in avg_doc_times and avg_doc_times["graph"] > 0:
        doc_overhead["hybrid_vs_graph"] = (avg_doc_times["hybrid"] / avg_doc_times["graph"] - 1) * 100
    
    search_overhead = {}
    if "vector" in avg_search_times and avg_search_times["vector"] > 0:
        search_overhead["hybrid_vs_vector"] = (avg_search_times["hybrid"] / avg_search_times["vector"] - 1) * 100
    
    if "graph" in avg_search_times and avg_search_times["graph"] > 0:
        search_overhead["hybrid_vs_graph"] = (avg_search_times["hybrid"] / avg_search_times["graph"] - 1) * 100
    
    # Add custom metrics
    result.custom_metrics["avg_doc_times_ms"] = avg_doc_times
    result.custom_metrics["avg_search_times_ms"] = avg_search_times
    result.custom_metrics["doc_overhead_percent"] = doc_overhead
    result.custom_metrics["search_overhead_percent"] = search_overhead
    
    # Cleanup
    del graph_db
    del vector_db
    del hybrid_manager


def benchmark_hybrid_system_scalability(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark scalability of the hybrid knowledge system.
    
    This measures how the hybrid system scales with 
    increasing data volumes.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load documents from dataset
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Set up a fresh hybrid knowledge manager
    hybrid_manager = HybridKnowledgeManager(
        graph_name=f"benchmark_hybrid_scale_{int(time.time())}",
        vector_collection=f"benchmark_hybrid_scale_{int(time.time())}",
        persist=False  # Don't persist for benchmarks
    )
    
    # Shuffle documents for random sampling
    random.shuffle(documents)
    
    # Define data size increments (count of documents)
    num_docs = min(50, len(documents))
    increments = [int(num_docs * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
    increments = [i for i in increments if i > 0]
    
    # Metrics to track at each increment
    doc_processing_times = []  # (num_docs, avg_time_ms)
    search_times = []  # (num_docs, avg_time_ms)
    entity_counts = []  # (num_docs, count)
    relation_counts = []  # (num_docs, count)
    memory_usage = []  # (num_docs, memory_mb)
    
    # Standard search query
    search_query = "project management and team collaboration"
    
    # Benchmark at increasing data sizes
    docs_processed = 0
    
    for increment in increments:
        docs_to_add = increment - docs_processed
        if docs_to_add <= 0:
            continue
            
        increment_docs = documents[docs_processed:increment]
        
        # Process documents and measure times
        processing_times = []
        for document in increment_docs:
            start_time = time.time()
            hybrid_manager.process_document(
                document_text=document["content"],
                document_metadata=document["properties"]
            )
            end_time = time.time()
            processing_times.append((end_time - start_time) * 1000)  # ms
        
        docs_processed = increment
        
        # Calculate average processing time
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        doc_processing_times.append((increment, avg_processing_time))
        
        # Measure search time
        search_latencies = []
        for _ in range(5):  # Run multiple searches for average
            start_time = time.time()
            hybrid_manager.hybrid_search(query=search_query, limit=10)
            end_time = time.time()
            search_latencies.append((end_time - start_time) * 1000)  # ms
        
        avg_search_time = sum(search_latencies) / len(search_latencies) if search_latencies else 0
        search_times.append((increment, avg_search_time))
        
        # Get stats
        graph_stats = hybrid_manager.graph_db.get_stats()
        graph_entities = graph_stats.get("total_entities", 0)
        graph_relations = graph_stats.get("total_relations", 0)
        
        entity_counts.append((increment, graph_entities))
        relation_counts.append((increment, graph_relations))
        
        # Measure memory usage
        mem_usage = _get_process_memory_usage()
        memory_usage.append((increment, mem_usage))
    
    # Calculate scaling factors
    processing_scaling = _calculate_scaling_factors(doc_processing_times)
    search_scaling = _calculate_scaling_factors(search_times)
    
    # Add custom metrics
    result.custom_metrics["doc_processing_times"] = doc_processing_times
    result.custom_metrics["search_times"] = search_times
    result.custom_metrics["entity_counts"] = entity_counts
    result.custom_metrics["relation_counts"] = relation_counts
    result.custom_metrics["memory_usage"] = memory_usage
    result.custom_metrics["processing_scaling"] = processing_scaling
    result.custom_metrics["search_scaling"] = search_scaling
    
    # Calculate overall scaling factor (average)
    for scaling_name, scaling_data in [
        ("processing", processing_scaling), 
        ("search", search_scaling)
    ]:
        if scaling_data:
            # Filter out inf values
            valid_factors = [f for _, f in scaling_data if f != float('inf')]
            avg_scaling_factor = sum(valid_factors) / len(valid_factors) if valid_factors else float('inf')
            result.custom_metrics[f"avg_{scaling_name}_scaling"] = avg_scaling_factor
    
    # Cleanup
    del hybrid_manager


def _get_process_memory_usage() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    except ImportError:
        return 0.0


def _calculate_scaling_factors(data_points: List[tuple]) -> List[tuple]:
    """
    Calculate scaling factors between data points.
    
    Args:
        data_points: List of (x, y) points where x is data size and y is time
        
    Returns:
        List of (x, scaling_factor) tuples
    """
    scaling_factors = []
    
    for i in range(1, len(data_points)):
        prev_size, prev_value = data_points[i-1]
        curr_size, curr_value = data_points[i]
        
        # Size ratio
        size_ratio = curr_size / prev_size if prev_size > 0 else float('inf')
        
        # Value ratio
        value_ratio = curr_value / prev_value if prev_value > 0 else float('inf')
        
        # Scaling factor (value growth / size growth)
        scaling_factor = value_ratio / size_ratio if size_ratio > 0 else float('inf')
        
        scaling_factors.append((curr_size, scaling_factor))
    
    return scaling_factors 