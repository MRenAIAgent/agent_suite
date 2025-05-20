"""
Graph Database Benchmarks

This module implements benchmarks for testing graph database operations
using the GraphManager and GraphitiKnowledgeGraphAdapter.
"""

import os
import time
import json
import logging
import random
from typing import Dict, List, Any, Optional

from agent_suite.graph import GraphManager
from knowledge.graphiti import GraphitiKnowledgeGraphAdapter

from benchmark_suite.metrics import BenchmarkResult


logger = logging.getLogger("benchmark_suite.graph")


def benchmark_graph_entity_creation(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark entity creation in the graph database.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load entities from dataset
    with open(os.path.join(dataset_path, "entities.json"), 'r') as f:
        entities = json.load(f)
    
    # Set up a fresh graph
    graph_name = f"benchmark_entity_creation_{int(time.time())}"
    graph_manager = GraphManager(
        graph_name=graph_name,
        persist=False  # Don't persist for benchmarks
    )
    
    # Sample entities for benchmarking
    entity_ids = list(entities.keys())
    random.shuffle(entity_ids)
    num_entities = min(config.get("num_entities", 1000), len(entity_ids))
    sample_entity_ids = entity_ids[:num_entities]
    
    # Benchmark entity creation
    start_memory = graph_manager._get_memory_usage() if hasattr(graph_manager, "_get_memory_usage") else 0
    
    # Create entities and measure time
    start_time = time.time()
    for entity_id in sample_entity_ids:
        entity_data = entities[entity_id]
        entity_type = entity_data.get("type", "unknown")
        graph_manager.add_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=entity_data
        )
    
    end_time = time.time()
    
    # Measure final memory
    end_memory = graph_manager._get_memory_usage() if hasattr(graph_manager, "_get_memory_usage") else 0
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    throughput = num_entities / elapsed_time if elapsed_time > 0 else 0
    
    # Update result
    result.throughput_ops = throughput
    if end_memory > 0 and start_memory > 0:
        result.memory_mb = end_memory - start_memory
    
    # Cleanup
    del graph_manager


def benchmark_graph_relation_creation(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark relation creation in the graph database.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load data from dataset
    with open(os.path.join(dataset_path, "entities.json"), 'r') as f:
        entities = json.load(f)
        
    with open(os.path.join(dataset_path, "relations.json"), 'r') as f:
        relations = json.load(f)
    
    # Set up a fresh graph
    graph_name = f"benchmark_relation_creation_{int(time.time())}"
    graph_manager = GraphManager(
        graph_name=graph_name,
        persist=False  # Don't persist for benchmarks
    )
    
    # First add entities (not part of the benchmark)
    for entity_id, entity_data in entities.items():
        entity_type = entity_data.get("type", "unknown")
        graph_manager.add_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=entity_data
        )
    
    # Sample relations for benchmarking
    random.shuffle(relations)
    num_relations = min(config.get("num_relations", 5000), len(relations))
    sample_relations = relations[:num_relations]
    
    # Benchmark relation creation
    start_memory = graph_manager._get_memory_usage() if hasattr(graph_manager, "_get_memory_usage") else 0
    
    # Create relations and measure time
    start_time = time.time()
    for relation in sample_relations:
        try:
            graph_manager.add_relation(
                source_id=relation["source_id"],
                relation_type=relation["relation_type"],
                target_id=relation["target_id"],
                relation_id=relation.get("id"),
                properties=relation.get("properties", {})
            )
        except ValueError:
            # Skip if entities don't exist
            continue
    
    end_time = time.time()
    
    # Measure final memory
    end_memory = graph_manager._get_memory_usage() if hasattr(graph_manager, "_get_memory_usage") else 0
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    throughput = num_relations / elapsed_time if elapsed_time > 0 else 0
    
    # Update result
    result.throughput_ops = throughput
    if end_memory > 0 and start_memory > 0:
        result.memory_mb = end_memory - start_memory
    
    # Cleanup
    del graph_manager


def benchmark_graph_query_performance(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark graph query performance.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load data from dataset
    with open(os.path.join(dataset_path, "entities.json"), 'r') as f:
        entities = json.load(f)
        
    with open(os.path.join(dataset_path, "relations.json"), 'r') as f:
        relations = json.load(f)
    
    # Set up a graph with data
    graph_name = f"benchmark_query_{int(time.time())}"
    graph_manager = GraphManager(
        graph_name=graph_name,
        persist=False  # Don't persist for benchmarks
    )
    
    # Load data (not part of the benchmark)
    for entity_id, entity_data in entities.items():
        entity_type = entity_data.get("type", "unknown")
        graph_manager.add_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=entity_data
        )
    
    for relation in relations:
        try:
            graph_manager.add_relation(
                source_id=relation["source_id"],
                relation_type=relation["relation_type"],
                target_id=relation["target_id"],
                relation_id=relation.get("id"),
                properties=relation.get("properties", {})
            )
        except ValueError:
            continue
    
    # Prepare queries
    entity_types = list(set(entity["type"] for entity in entities.values()))
    relation_types = list(set(relation["relation_type"] for relation in relations))
    entity_ids = list(entities.keys())
    
    # 1. Query entities by type
    entity_type_queries = []
    for _ in range(10):
        if entity_types:
            entity_type = random.choice(entity_types)
            entity_type_queries.append({"type": entity_type})
    
    # 2. Query relations by type
    relation_type_queries = []
    for _ in range(10):
        if relation_types:
            relation_type = random.choice(relation_types)
            relation_type_queries.append({"relation_type": relation_type})
    
    # 3. Query by entity ID
    entity_id_queries = []
    for _ in range(10):
        if entity_ids:
            entity_id = random.choice(entity_ids)
            entity_id_queries.append({"entity_id": entity_id})
    
    # 4. Graph traversal queries
    traversal_queries = []
    for _ in range(10):
        if entity_ids:
            start_id = random.choice(entity_ids)
            max_depth = random.randint(1, 3)
            traversal_queries.append({"start_id": start_id, "max_depth": max_depth})
    
    # Benchmark queries
    query_times = {
        "entity_type": [],
        "relation_type": [],
        "entity_id": [],
        "traversal": []
    }
    
    # 1. Entity type queries
    for query in entity_type_queries:
        start_time = time.time()
        results = graph_manager.query_entities(entity_type=query.get("type"), limit=100)
        end_time = time.time()
        query_times["entity_type"].append((end_time - start_time) * 1000)  # ms
    
    # 2. Relation type queries
    for query in relation_type_queries:
        start_time = time.time()
        results = graph_manager.query_relations(relation_type=query.get("relation_type"), limit=100)
        end_time = time.time()
        query_times["relation_type"].append((end_time - start_time) * 1000)  # ms
    
    # 3. Entity ID queries
    for query in entity_id_queries:
        start_time = time.time()
        entity = graph_manager.get_entity(query.get("entity_id"))
        end_time = time.time()
        query_times["entity_id"].append((end_time - start_time) * 1000)  # ms
    
    # 4. Traversal queries
    for query in traversal_queries:
        start_time = time.time()
        traversal = graph_manager.graph_traversal(
            start_id=query.get("start_id"),
            max_depth=query.get("max_depth")
        )
        end_time = time.time()
        query_times["traversal"].append((end_time - start_time) * 1000)  # ms
    
    # Calculate average query times
    avg_times = {}
    for query_type, times in query_times.items():
        if times:
            avg_times[query_type] = sum(times) / len(times)
    
    # Add to custom metrics
    result.custom_metrics["query_times"] = avg_times
    
    # Cleanup
    del graph_manager


def benchmark_graph_adapter_performance(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark GraphitiKnowledgeGraphAdapter performance.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load data from dataset
    with open(os.path.join(dataset_path, "entities.json"), 'r') as f:
        entities = json.load(f)
        
    with open(os.path.join(dataset_path, "relations.json"), 'r') as f:
        relations = json.load(f)
        
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Set up a fresh adapter
    graph_name = f"benchmark_adapter_{int(time.time())}"
    adapter = GraphitiKnowledgeGraphAdapter(
        graph_name=graph_name,
        persist=False  # Don't persist for benchmarks
    )
    
    # Sample data for benchmarking
    entity_ids = list(entities.keys())
    random.shuffle(entity_ids)
    num_entities = min(100, len(entity_ids))
    sample_entity_ids = entity_ids[:num_entities]
    
    random.shuffle(relations)
    num_relations = min(200, len(relations))
    sample_relations = relations[:num_relations]
    
    random.shuffle(documents)
    num_documents = min(20, len(documents))
    sample_documents = documents[:num_documents]
    
    # Operation times
    operation_times = {
        "add_entity": [],
        "add_relation": [],
        "add_text": [],
        "query_entities": [],
        "query_relations": [],
        "semantic_search": []
    }
    
    # 1. Benchmark add_entity
    for i in range(num_entities):
        entity_id = sample_entity_ids[i]
        entity_data = entities[entity_id]
        entity_type = entity_data.get("type", "unknown")
        
        start_time = time.time()
        adapter.add_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=entity_data
        )
        end_time = time.time()
        
        operation_times["add_entity"].append((end_time - start_time) * 1000)
    
    # 2. Benchmark add_relation
    for i in range(min(num_relations, len(sample_relations))):
        relation = sample_relations[i]
        
        try:
            start_time = time.time()
            adapter.add_relation(
                source_id=relation["source_id"],
                relation_type=relation["relation_type"],
                target_id=relation["target_id"],
                properties=relation.get("properties", {})
            )
            end_time = time.time()
            
            operation_times["add_relation"].append((end_time - start_time) * 1000)
        except Exception:
            pass
    
    # 3. Benchmark add_text
    for i in range(min(num_documents, len(sample_documents))):
        document = sample_documents[i]
        
        start_time = time.time()
        adapter.add_text(
            text=document["content"],
            properties=document["properties"]
        )
        end_time = time.time()
        
        operation_times["add_text"].append((end_time - start_time) * 1000)
    
    # 4. Benchmark query_entities
    entity_types = list(set(entity["type"] for entity in entities.values() if "type" in entity))
    
    for _ in range(10):
        if entity_types:
            entity_type = random.choice(entity_types)
            
            start_time = time.time()
            adapter.query_entities(entity_type=entity_type, limit=50)
            end_time = time.time()
            
            operation_times["query_entities"].append((end_time - start_time) * 1000)
    
    # 5. Benchmark query_relations
    relation_types = list(set(relation["relation_type"] for relation in relations))
    
    for _ in range(10):
        if relation_types:
            relation_type = random.choice(relation_types)
            
            start_time = time.time()
            adapter.query_relations(relation_type=relation_type, limit=50)
            end_time = time.time()
            
            operation_times["query_relations"].append((end_time - start_time) * 1000)
    
    # 6. Benchmark semantic_search
    keywords = ["Project", "Technology", "Team", "System", "Application", "Service"]
    
    for _ in range(10):
        keyword = random.choice(keywords)
        
        start_time = time.time()
        adapter.semantic_search(query=keyword, limit=10)
        end_time = time.time()
        
        operation_times["semantic_search"].append((end_time - start_time) * 1000)
    
    # Calculate average operation times
    avg_times = {}
    for operation, times in operation_times.items():
        if times:
            avg_times[operation] = sum(times) / len(times)
    
    # Add to custom metrics
    result.custom_metrics["operation_times"] = avg_times
    
    # Cleanup
    del adapter


def benchmark_graph_persistence_overhead(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark graph persistence overhead.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load data from dataset
    with open(os.path.join(dataset_path, "entities.json"), 'r') as f:
        entities = json.load(f)
        
    with open(os.path.join(dataset_path, "relations.json"), 'r') as f:
        relations = json.load(f)
    
    # Create a temporary directory for persistence
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Sample data
    entity_ids = list(entities.keys())
    random.shuffle(entity_ids)
    num_entities = min(500, len(entity_ids))
    sample_entity_ids = entity_ids[:num_entities]
    
    random.shuffle(relations)
    num_relations = min(1000, len(relations))
    sample_relations = relations[:num_relations]
    
    # 1. Memory-only graph
    memory_graph = GraphManager(
        graph_name="memory_benchmark",
        persist=False
    )
    
    # 2. Persistent graph
    persistent_graph = GraphManager(
        graph_name="persistence_benchmark",
        persist=True,
        persistence_path=temp_dir
    )
    
    # Load data into both graphs
    for graph in [memory_graph, persistent_graph]:
        # Add entities
        for entity_id in sample_entity_ids:
            entity_data = entities[entity_id]
            entity_type = entity_data.get("type", "unknown")
            graph.add_entity(
                entity_id=entity_id,
                entity_type=entity_type,
                properties=entity_data
            )
        
        # Add relations
        for relation in sample_relations:
            try:
                graph.add_relation(
                    source_id=relation["source_id"],
                    relation_type=relation["relation_type"],
                    target_id=relation["target_id"],
                    relation_id=relation.get("id"),
                    properties=relation.get("properties", {})
                )
            except ValueError:
                continue
    
    # Measure performance difference for common operations
    operations = [
        "query_entities",
        "query_relations",
        "get_entity",
        "graph_traversal"
    ]
    
    performance_ratios = {}
    
    # 1. Query entities by type
    entity_types = list(set(entity["type"] for entity in entities.values() if "type" in entity))
    
    memory_times = []
    persistent_times = []
    
    if entity_types:
        for _ in range(10):
            entity_type = random.choice(entity_types)
            
            # Memory graph
            start_time = time.time()
            memory_graph.query_entities(entity_type=entity_type, limit=50)
            end_time = time.time()
            memory_times.append(end_time - start_time)
            
            # Persistent graph
            start_time = time.time()
            persistent_graph.query_entities(entity_type=entity_type, limit=50)
            end_time = time.time()
            persistent_times.append(end_time - start_time)
    
    if memory_times and persistent_times:
        avg_memory = sum(memory_times) / len(memory_times)
        avg_persistent = sum(persistent_times) / len(persistent_times)
        if avg_memory > 0:
            performance_ratios["query_entities"] = avg_persistent / avg_memory
    
    # 2. Query relations by type
    relation_types = list(set(relation["relation_type"] for relation in relations))
    
    memory_times = []
    persistent_times = []
    
    if relation_types:
        for _ in range(10):
            relation_type = random.choice(relation_types)
            
            # Memory graph
            start_time = time.time()
            memory_graph.query_relations(relation_type=relation_type, limit=50)
            end_time = time.time()
            memory_times.append(end_time - start_time)
            
            # Persistent graph
            start_time = time.time()
            persistent_graph.query_relations(relation_type=relation_type, limit=50)
            end_time = time.time()
            persistent_times.append(end_time - start_time)
    
    if memory_times and persistent_times:
        avg_memory = sum(memory_times) / len(memory_times)
        avg_persistent = sum(persistent_times) / len(persistent_times)
        if avg_memory > 0:
            performance_ratios["query_relations"] = avg_persistent / avg_memory
    
    # 3. Get entity by ID
    memory_times = []
    persistent_times = []
    
    for _ in range(10):
        if sample_entity_ids:
            entity_id = random.choice(sample_entity_ids)
            
            # Memory graph
            start_time = time.time()
            memory_graph.get_entity(entity_id)
            end_time = time.time()
            memory_times.append(end_time - start_time)
            
            # Persistent graph
            start_time = time.time()
            persistent_graph.get_entity(entity_id)
            end_time = time.time()
            persistent_times.append(end_time - start_time)
    
    if memory_times and persistent_times:
        avg_memory = sum(memory_times) / len(memory_times)
        avg_persistent = sum(persistent_times) / len(persistent_times)
        if avg_memory > 0:
            performance_ratios["get_entity"] = avg_persistent / avg_memory
    
    # 4. Graph traversal
    memory_times = []
    persistent_times = []
    
    for _ in range(5):
        if sample_entity_ids:
            start_id = random.choice(sample_entity_ids)
            max_depth = random.randint(1, 2)
            
            # Memory graph
            start_time = time.time()
            memory_graph.graph_traversal(start_id=start_id, max_depth=max_depth)
            end_time = time.time()
            memory_times.append(end_time - start_time)
            
            # Persistent graph
            start_time = time.time()
            persistent_graph.graph_traversal(start_id=start_id, max_depth=max_depth)
            end_time = time.time()
            persistent_times.append(end_time - start_time)
    
    if memory_times and persistent_times:
        avg_memory = sum(memory_times) / len(memory_times)
        avg_persistent = sum(persistent_times) / len(persistent_times)
        if avg_memory > 0:
            performance_ratios["graph_traversal"] = avg_persistent / avg_memory
    
    # Measure storage size
    import os
    
    storage_size = 0
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            storage_size += os.path.getsize(file_path)
    
    # Calculate storage efficiency
    num_nodes = persistent_graph.graph.number_of_nodes()
    num_edges = persistent_graph.graph.number_of_edges()
    if num_nodes + num_edges > 0:
        bytes_per_item = storage_size / (num_nodes + num_edges)
    else:
        bytes_per_item = 0
    
    # Add to custom metrics
    result.custom_metrics["persistence_overhead"] = performance_ratios
    result.custom_metrics["storage_size_bytes"] = storage_size
    result.custom_metrics["bytes_per_item"] = bytes_per_item
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    del memory_graph
    del persistent_graph 