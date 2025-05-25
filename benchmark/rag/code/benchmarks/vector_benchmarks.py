"""
Vector Database Benchmarks

This module implements benchmarks for testing vector database operations
using the QdrantVectorDatabase.
"""

import os
import time
import json
import logging
import random
from typing import Dict, List, Any, Optional

from knowledge.vector import QdrantVectorDatabase
from benchmark_suite.metrics import BenchmarkResult


logger = logging.getLogger("benchmark_suite.vector")


def benchmark_vector_text_embedding(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark text embedding and storage in vector database.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load documents from dataset
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Set up a fresh vector database
    collection_name = f"benchmark_vector_embedding_{int(time.time())}"
    vector_db = QdrantVectorDatabase(
        collection_name=collection_name,
        persist=False  # Don't persist for benchmarks
    )
    
    # Sample documents for benchmarking
    random.shuffle(documents)
    num_documents = min(config.get("num_documents", 100), len(documents))
    sample_documents = documents[:num_documents]
    
    # Measure initial memory
    start_memory = vector_db._get_memory_usage() if hasattr(vector_db, "_get_memory_usage") else 0
    
    # Benchmark text embedding and storage
    embedding_times = []
    
    for document in sample_documents:
        start_time = time.time()
        vector_db.add_text(
            text=document["content"],
            properties=document["properties"]
        )
        end_time = time.time()
        embedding_times.append((end_time - start_time) * 1000)  # ms
    
    # Measure final memory
    end_memory = vector_db._get_memory_usage() if hasattr(vector_db, "_get_memory_usage") else 0
    
    # Calculate metrics
    avg_embedding_time = sum(embedding_times) / len(embedding_times) if embedding_times else 0
    throughput = num_documents / (sum(embedding_times) / 1000) if sum(embedding_times) > 0 else 0
    
    # Update result
    result.throughput_ops = throughput
    if end_memory > 0 and start_memory > 0:
        result.memory_mb = end_memory - start_memory
    
    # Add custom metrics
    result.custom_metrics["avg_embedding_time_ms"] = avg_embedding_time
    result.custom_metrics["embedding_times_ms"] = embedding_times
    
    # Cleanup
    del vector_db


def benchmark_vector_entity_storage(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark entity storage in vector database.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load entities from dataset
    with open(os.path.join(dataset_path, "entities.json"), 'r') as f:
        entities = json.load(f)
    
    # Set up a fresh vector database
    collection_name = f"benchmark_vector_entity_{int(time.time())}"
    vector_db = QdrantVectorDatabase(
        collection_name=collection_name,
        persist=False  # Don't persist for benchmarks
    )
    
    # Sample entities for benchmarking
    entity_ids = list(entities.keys())
    random.shuffle(entity_ids)
    num_entities = min(config.get("num_entities", 1000), len(entity_ids))
    sample_entity_ids = entity_ids[:num_entities]
    
    # Measure initial memory
    start_memory = vector_db._get_memory_usage() if hasattr(vector_db, "_get_memory_usage") else 0
    
    # Benchmark entity storage
    entity_times = []
    
    for entity_id in sample_entity_ids:
        entity_data = entities[entity_id]
        entity_type = entity_data.get("type", "unknown")
        
        start_time = time.time()
        vector_db.add_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=entity_data
        )
        end_time = time.time()
        entity_times.append((end_time - start_time) * 1000)  # ms
    
    # Measure final memory
    end_memory = vector_db._get_memory_usage() if hasattr(vector_db, "_get_memory_usage") else 0
    
    # Calculate metrics
    avg_entity_time = sum(entity_times) / len(entity_times) if entity_times else 0
    throughput = num_entities / (sum(entity_times) / 1000) if sum(entity_times) > 0 else 0
    
    # Update result
    result.throughput_ops = throughput
    if end_memory > 0 and start_memory > 0:
        result.memory_mb = end_memory - start_memory
    
    # Add custom metrics
    result.custom_metrics["avg_entity_time_ms"] = avg_entity_time
    result.custom_metrics["entity_times_ms"] = entity_times
    
    # Cleanup
    del vector_db


def benchmark_vector_semantic_search(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark semantic search in vector database.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load documents from dataset
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Set up a fresh vector database
    collection_name = f"benchmark_vector_search_{int(time.time())}"
    vector_db = QdrantVectorDatabase(
        collection_name=collection_name,
        persist=False  # Don't persist for benchmarks
    )
    
    # Load documents (not part of the benchmark)
    for document in documents:
        vector_db.add_text(
            text=document["content"],
            properties=document["properties"]
        )
    
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
    
    # Benchmark semantic search
    search_times = []
    result_counts = []
    
    # Run search benchmarks
    for query in search_queries:
        start_time = time.time()
        results = vector_db.semantic_search(query=query, limit=10)
        end_time = time.time()
        
        search_time = (end_time - start_time) * 1000  # ms
        search_times.append(search_time)
        result_counts.append(len(results))
    
    # Calculate metrics
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0
    avg_result_count = sum(result_counts) / len(result_counts) if result_counts else 0
    
    # Add custom metrics
    result.custom_metrics["avg_search_time_ms"] = avg_search_time
    result.custom_metrics["search_times_ms"] = search_times
    result.custom_metrics["avg_result_count"] = avg_result_count
    
    # Cleanup
    del vector_db


def benchmark_vector_database_scalability(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark vector database scalability with increasing data sizes.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load documents from dataset
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Set up a fresh vector database
    collection_name = f"benchmark_vector_scalability_{int(time.time())}"
    vector_db = QdrantVectorDatabase(
        collection_name=collection_name,
        persist=False  # Don't persist for benchmarks
    )
    
    # Shuffle documents for random sampling
    random.shuffle(documents)
    
    # Define data size increments (percentage of total documents)
    increments = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Storage metrics
    storage_sizes = []
    
    # Search performance metrics at different data sizes
    search_times = []
    
    # Standard search query
    search_query = "project management and team collaboration"
    
    # Benchmark at increasing data sizes
    for increment in increments:
        # Calculate number of documents for this increment
        num_docs = int(len(documents) * increment)
        if num_docs == 0:
            continue
            
        # Get document subset
        doc_subset = documents[:num_docs]
        
        # Add documents
        for doc in doc_subset:
            vector_db.add_text(
                text=doc["content"],
                properties=doc["properties"]
            )
        
        # Measure storage size (if available)
        storage_size = vector_db.get_storage_size() if hasattr(vector_db, "get_storage_size") else None
        storage_sizes.append((num_docs, storage_size))
        
        # Measure search time
        search_latencies = []
        for _ in range(5):  # Run multiple searches for average
            start_time = time.time()
            vector_db.semantic_search(query=search_query, limit=10)
            end_time = time.time()
            search_latencies.append((end_time - start_time) * 1000)  # ms
        
        avg_search_time = sum(search_latencies) / len(search_latencies)
        search_times.append((num_docs, avg_search_time))
    
    # Calculate scaling factors
    scaling_factors = []
    for i in range(1, len(search_times)):
        prev_docs, prev_time = search_times[i-1]
        curr_docs, curr_time = search_times[i]
        
        # Documents ratio
        docs_ratio = curr_docs / prev_docs if prev_docs > 0 else float('inf')
        
        # Time ratio
        time_ratio = curr_time / prev_time if prev_time > 0 else float('inf')
        
        # Scaling factor (time growth / data growth)
        scaling_factor = time_ratio / docs_ratio if docs_ratio > 0 else float('inf')
        
        scaling_factors.append(scaling_factor)
    
    # Add custom metrics
    result.custom_metrics["storage_sizes"] = storage_sizes
    result.custom_metrics["search_times"] = search_times
    result.custom_metrics["scaling_factors"] = scaling_factors
    
    # Calculate overall scaling factor (average)
    if scaling_factors:
        # Filter out inf values
        valid_factors = [f for f in scaling_factors if f != float('inf')]
        avg_scaling_factor = sum(valid_factors) / len(valid_factors) if valid_factors else float('inf')
        result.custom_metrics["avg_scaling_factor"] = avg_scaling_factor
    
    # Cleanup
    del vector_db


def benchmark_vector_persistence_overhead(
    config: Dict[str, Any],
    dataset_path: str,
    result: BenchmarkResult,
    **kwargs
) -> None:
    """
    Benchmark vector database persistence overhead.
    
    Args:
        config: Benchmark configuration
        dataset_path: Path to dataset
        result: Result container
        **kwargs: Additional arguments
    """
    # Load documents from dataset
    with open(os.path.join(dataset_path, "documents.json"), 'r') as f:
        documents = json.load(f)
    
    # Create a temporary directory for persistence
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Sample documents
    random.shuffle(documents)
    num_documents = min(50, len(documents))
    sample_documents = documents[:num_documents]
    
    # 1. Memory-only vector database
    memory_db = QdrantVectorDatabase(
        collection_name="memory_benchmark",
        persist=False
    )
    
    # 2. Persistent vector database
    persistent_db = QdrantVectorDatabase(
        collection_name="persistence_benchmark",
        persist=True,
        persistence_path=temp_dir
    )
    
    # Load documents into both databases
    for db in [memory_db, persistent_db]:
        for doc in sample_documents:
            db.add_text(
                text=doc["content"],
                properties=doc["properties"]
            )
    
    # Define search queries
    search_queries = [
        "cloud architecture",
        "project management",
        "software development",
        "data analytics",
        "team collaboration"
    ]
    
    # Benchmark search performance
    memory_times = []
    persistent_times = []
    
    for query in search_queries:
        # Memory database
        start_time = time.time()
        memory_db.semantic_search(query=query, limit=10)
        end_time = time.time()
        memory_times.append((end_time - start_time) * 1000)  # ms
        
        # Persistent database
        start_time = time.time()
        persistent_db.semantic_search(query=query, limit=10)
        end_time = time.time()
        persistent_times.append((end_time - start_time) * 1000)  # ms
    
    # Calculate average search times
    avg_memory_time = sum(memory_times) / len(memory_times) if memory_times else 0
    avg_persistent_time = sum(persistent_times) / len(persistent_times) if persistent_times else 0
    
    # Calculate persistence overhead ratio
    overhead_ratio = avg_persistent_time / avg_memory_time if avg_memory_time > 0 else float('inf')
    
    # Measure storage size
    storage_size = 0
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            storage_size += os.path.getsize(file_path)
    
    # Calculate storage per document
    storage_per_document = storage_size / num_documents if num_documents > 0 else 0
    
    # Add custom metrics
    result.custom_metrics["memory_search_time_ms"] = avg_memory_time
    result.custom_metrics["persistent_search_time_ms"] = avg_persistent_time
    result.custom_metrics["persistence_overhead_ratio"] = overhead_ratio
    result.custom_metrics["storage_size_bytes"] = storage_size
    result.custom_metrics["storage_per_document_bytes"] = storage_per_document
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    del memory_db
    del persistent_db 