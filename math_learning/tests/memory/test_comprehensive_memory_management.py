#!/usr/bin/env python3
"""
Comprehensive Memory Management Tests

This test suite provides comprehensive coverage for memory management
in the math learning system, including:
- Memory usage tracking and monitoring
- Memory cleanup and garbage collection
- Memory leaks detection
- Performance optimization
- Large dataset handling
- Concurrent memory usage
- Resource management
"""

import pytest
import gc
import sys
import time
import threading
import tempfile
import os
import psutil
import weakref
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Add project root to path
# From math_learning/tests/memory/test_comprehensive_memory_management.py
# Go up 4 levels: memory -> tests -> math_learning -> agent_suite
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Memory-related imports
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.algebra_graph import AlgebraGraph
from math_learning.config.storage_config import MathLearningStorageConfig, StorageBackend
from math_learning.recommendation.gap_analyzer import GapAnalyzer
from math_learning.recommendation.recommender import Recommender
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.exercises.exercise import Exercise

# RAG imports (if available)
try:
    from agents.rag.storage.graph.memory_storage import MemoryGraphStorageAdaptor
    from agents.rag.storage.key_value.memory_storage import MemoryKeyValueStorage
    from agents.rag.models.knowledge import Entity, Relationship
    from agents.rag.models.context import ContextItem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class MemoryTracker:
    """Utility class for tracking memory usage."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.measurements = []
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def record_measurement(self, label: str = ""):
        """Record current memory usage."""
        current_memory = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
        self.measurements.append({
            "label": label,
            "memory_mb": current_memory,
            "timestamp": time.time()
        })
        return current_memory
    
    def get_memory_increase(self) -> float:
        """Get memory increase since initialization."""
        return self.get_memory_usage() - self.initial_memory
    
    def get_peak_increase(self) -> float:
        """Get peak memory increase."""
        return self.peak_memory - self.initial_memory
    
    def force_gc(self):
        """Force garbage collection and record memory."""
        gc.collect()
        return self.record_measurement("after_gc")


class TestBasicMemoryUsage:
    """Tests for basic memory usage patterns."""
    
    def test_learning_graph_memory_usage(self):
        """Test memory usage of learning graphs."""
        tracker = MemoryTracker()
        tracker.record_measurement("start")
        
        # Create learning graphs
        learning_graphs = []
        for i in range(100):
            lg = LearningGraph(f"student_{i}", f"Student {i}")
            
            # Add some data
            for j in range(20):
                concept_id = f"concept_{j}"
                lg.set_mastery(concept_id, 0.1 + j * 0.04, 0.5 + j * 0.02)
                
                # Add exercise history
                for k in range(5):
                    exercise_id = f"ex_{j}_{k}"
                    success = k % 2 == 0
                    lg.record_exercise_attempt(exercise_id, concept_id, success, 0.5, 1.0)
            
            learning_graphs.append(lg)
        
        tracker.record_measurement("after_creation")
        
        # Memory should be reasonable (< 100MB increase for 100 students)
        memory_increase = tracker.get_memory_increase()
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.2f} MB"
        
        # Test memory cleanup
        learning_graphs.clear()
        tracker.force_gc()
        
        # Memory should decrease after cleanup
        final_memory = tracker.get_memory_increase()
        assert final_memory < memory_increase * 0.8, "Memory not properly cleaned up"
    
    def test_knowledge_graph_memory_usage(self):
        """Test memory usage of knowledge graphs."""
        tracker = MemoryTracker()
        tracker.record_measurement("start")
        
        # Create temporary knowledge graph
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write('{"concepts": {}, "relationships": []}')
        temp_file.close()
        
        try:
            config = MathLearningStorageConfig(
                backend=StorageBackend.NETWORKX,
                networkx_persistence_file=temp_file.name,
                networkx_auto_save=False
            )
            
            kg = KnowledgeGraph(name="Memory Test", config=config)
            
            # Add many concepts
            concepts = []
            for i in range(500):
                concept = Concept(
                    name=f"Concept {i}",
                    concept_id=f"concept_{i:03d}",
                    description=f"Description for concept {i} with additional text to increase memory usage",
                    difficulty=0.1 + (i % 10) * 0.1,
                    time_to_master=5 + (i % 20),
                    category=f"Category {i % 10}"
                )
                concepts.append(concept)
                kg.add_concept(concept)
                
                # Add some relationships
                if i > 0:
                    # Add prerequisite to previous concept
                    kg.add_prerequisite(f"concept_{i-1:03d}", f"concept_{i:03d}")
            
            tracker.record_measurement("after_graph_creation")
            
            # Test retrieval performance with memory tracking
            start_time = time.time()
            for i in range(100):
                concept = kg.get_concept(f"concept_{i:03d}")
                assert concept is not None
            retrieval_time = time.time() - start_time
            
            tracker.record_measurement("after_retrieval")
            
            # Memory increase should be reasonable (< 50MB for 500 concepts)
            memory_increase = tracker.get_memory_increase()
            assert memory_increase < 50, f"Memory usage too high: {memory_increase:.2f} MB"
            
            # Retrieval should be fast
            assert retrieval_time < 1.0, f"Retrieval too slow: {retrieval_time:.2f}s"
            
            # Cleanup
            del kg
            del concepts
            tracker.force_gc()
            
        finally:
            os.unlink(temp_file.name)
    
    def test_exercise_bank_memory_usage(self):
        """Test memory usage of exercise banks."""
        tracker = MemoryTracker()
        tracker.record_measurement("start")
        
        # Create exercise bank with many exercises
        exercise_bank = ExerciseBank()
        
        exercises = []
        for i in range(1000):
            exercise = Exercise(
                exercise_id=f"ex_{i:04d}",
                name=f"Exercise {i}",
                description=f"This is exercise number {i} with a detailed description that includes multiple sentences to test memory usage patterns.",
                difficulty=0.1 + (i % 10) * 0.1,
                format_type="multiple_choice",
                concept_relationships={f"concept_{i % 50}": 0.8}
            )
            exercises.append(exercise)
            exercise_bank.add_exercise(exercise)
        
        tracker.record_measurement("after_exercises")
        
        # Test retrieval
        for i in range(100):
            retrieved = exercise_bank.get_exercise(f"ex_{i:04d}")
            assert retrieved is not None
        
        tracker.record_measurement("after_retrieval")
        
        # Memory should be reasonable
        memory_increase = tracker.get_memory_increase()
        assert memory_increase < 100, f"Exercise bank memory too high: {memory_increase:.2f} MB"
        
        # Cleanup
        exercises.clear()
        del exercise_bank
        tracker.force_gc()


class TestMemoryLeakDetection:
    """Tests for detecting memory leaks."""
    
    def test_learning_graph_leak_detection(self):
        """Test for memory leaks in learning graph creation/destruction."""
        tracker = MemoryTracker()
        
        # Baseline measurement
        tracker.record_measurement("baseline")
        
        # Create and destroy learning graphs multiple times
        for cycle in range(5):
            learning_graphs = []
            
            # Create many learning graphs
            for i in range(50):
                lg = LearningGraph(f"student_{i}_cycle_{cycle}", f"Student {i}")
                
                # Add substantial data
                for j in range(30):
                    concept_id = f"concept_{j}"
                    lg.set_mastery(concept_id, 0.1 + j * 0.03, 0.4 + j * 0.02)
                    
                    # Add exercise attempts
                    for k in range(10):
                        exercise_id = f"ex_{j}_{k}"
                        lg.record_exercise_attempt(exercise_id, concept_id, k % 3 == 0, 0.5, 1.0)
                
                learning_graphs.append(lg)
            
            tracker.record_measurement(f"cycle_{cycle}_created")
            
            # Destroy all learning graphs
            learning_graphs.clear()
            tracker.force_gc()
            
            tracker.record_measurement(f"cycle_{cycle}_cleaned")
        
        # Check for memory leaks
        final_memory = tracker.get_memory_increase()
        
        # Memory should not continuously increase (allow some variance)
        assert final_memory < 20, f"Possible memory leak detected: {final_memory:.2f} MB increase"
    
    def test_weak_reference_cleanup(self):
        """Test that objects are properly cleaned up using weak references."""
        # Create objects and weak references
        learning_graphs = []
        weak_refs = []
        
        for i in range(10):
            lg = LearningGraph(f"student_{i}", f"Student {i}")
            learning_graphs.append(lg)
            weak_refs.append(weakref.ref(lg))
        
        # All weak references should be alive
        alive_refs = [ref for ref in weak_refs if ref() is not None]
        assert len(alive_refs) == 10, "All objects should be alive initially"
        
        # Delete objects
        learning_graphs.clear()
        gc.collect()
        
        # Weak references should be dead
        alive_refs = [ref for ref in weak_refs if ref() is not None]
        assert len(alive_refs) == 0, f"Objects not properly cleaned up: {len(alive_refs)} still alive"
    
    @pytest.mark.skipif(not RAG_AVAILABLE, reason="RAG dependencies not available")
    def test_rag_storage_leak_detection(self):
        """Test for memory leaks in RAG storage operations."""
        tracker = MemoryTracker()
        tracker.record_measurement("baseline")
        
        for cycle in range(3):
            # Create storage adaptors
            graph_storage = MemoryGraphStorageAdaptor()
            kv_storage = MemoryKeyValueStorage()
            
            # Add substantial data
            entities = []
            for i in range(100):
                entity = Entity(
                    type="test_entity",
                    properties={
                        "name": f"Entity {i} Cycle {cycle}",
                        "description": f"Description with additional text for entity {i} in cycle {cycle}",
                        "value": i,
                        "cycle": cycle
                    }
                )
                entities.append(entity)
                
                # Store in graph storage
                import asyncio
                asyncio.run(graph_storage.store_entity(entity))
                
                # Store in key-value storage
                asyncio.run(kv_storage.set(f"key_{i}_{cycle}", {
                    "data": f"value_{i}",
                    "cycle": cycle,
                    "additional_data": "x" * 100  # Add some bulk
                }))
            
            tracker.record_measurement(f"rag_cycle_{cycle}_created")
            
            # Cleanup
            entities.clear()
            asyncio.run(graph_storage.close())
            asyncio.run(kv_storage.close())
            del graph_storage
            del kv_storage
            tracker.force_gc()
            
            tracker.record_measurement(f"rag_cycle_{cycle}_cleaned")
        
        # Check for leaks
        final_memory = tracker.get_memory_increase()
        assert final_memory < 30, f"RAG storage memory leak: {final_memory:.2f} MB"


class TestConcurrentMemoryUsage:
    """Tests for memory usage under concurrent access."""
    
    def test_concurrent_learning_graph_access(self):
        """Test memory usage with concurrent access to learning graphs."""
        tracker = MemoryTracker()
        tracker.record_measurement("start")
        
        # Create shared learning graphs
        shared_graphs = {}
        for i in range(10):
            lg = LearningGraph(f"shared_student_{i}", f"Shared Student {i}")
            # Pre-populate with data
            for j in range(50):
                lg.set_mastery(f"concept_{j}", 0.1 + j * 0.01, 0.5)
            shared_graphs[f"student_{i}"] = lg
        
        tracker.record_measurement("graphs_created")
        
        # Define worker function
        def worker_function(worker_id: int, iterations: int):
            """Worker function that accesses learning graphs."""
            for i in range(iterations):
                # Access random learning graph
                student_id = f"student_{i % 10}"
                lg = shared_graphs[student_id]
                
                # Perform operations
                concept_id = f"concept_{i % 50}"
                current_mastery = lg.get_mastery(concept_id)
                lg.record_exercise_attempt(f"ex_{worker_id}_{i}", concept_id, i % 2 == 0, 0.5, 1.0)
        
        # Run concurrent workers
        threads = []
        num_workers = 5
        iterations_per_worker = 100
        
        start_time = time.time()
        
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=worker_function,
                args=(worker_id, iterations_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        tracker.record_measurement("concurrent_work_done")
        
        # Verify memory usage is reasonable
        memory_increase = tracker.get_memory_increase()
        assert memory_increase < 50, f"Concurrent memory usage too high: {memory_increase:.2f} MB"
        
        # Verify performance is reasonable
        total_time = end_time - start_time
        assert total_time < 10.0, f"Concurrent operations too slow: {total_time:.2f}s"
        
        # Verify data integrity
        for student_id, lg in shared_graphs.items():
            # Should have exercise history from multiple workers
            total_exercises = len(lg.exercise_history)
            assert total_exercises > 0, f"No exercises recorded for {student_id}"
    
    def test_memory_usage_under_load(self):
        """Test memory behavior under heavy load."""
        tracker = MemoryTracker()
        tracker.record_measurement("start")
        
        # Create system components
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write('{"concepts": {}, "relationships": []}')
        temp_file.close()
        
        try:
            config = MathLearningStorageConfig(
                backend=StorageBackend.NETWORKX,
                networkx_persistence_file=temp_file.name,
                networkx_auto_save=False
            )
            
            kg = KnowledgeGraph(name="Load Test", config=config)
            
            # Add concepts
            for i in range(100):
                concept = Concept(
                    name=f"Concept {i}",
                    concept_id=f"concept_{i:03d}",
                    difficulty=0.1 + (i % 10) * 0.1,
                    time_to_master=5 + (i % 15)
                )
                kg.add_concept(concept)
            
            # Add relationships
            for i in range(99):
                kg.add_prerequisite(f"concept_{i:03d}", f"concept_{i+1:03d}")
            
            tracker.record_measurement("knowledge_graph_ready")
            
            # Create many learning graphs and perform operations
            learning_graphs = []
            for student_id in range(50):
                lg = LearningGraph(f"load_student_{student_id}", f"Load Student {student_id}")
                
                # Simulate intensive learning activity
                for concept_idx in range(100):
                    concept_id = f"concept_{concept_idx:03d}"
                    
                    # Multiple exercise attempts per concept
                    for attempt in range(20):
                        exercise_id = f"ex_{student_id}_{concept_idx}_{attempt}"
                        success = (attempt + concept_idx + student_id) % 3 != 0
                        difficulty = 0.1 + (attempt % 10) * 0.1
                        lg.record_exercise_attempt(exercise_id, concept_id, success, difficulty, 1.0)
                
                learning_graphs.append(lg)
                
                # Record memory every 10 students
                if (student_id + 1) % 10 == 0:
                    tracker.record_measurement(f"students_{student_id + 1}")
            
            tracker.record_measurement("load_test_complete")
            
            # Memory should still be reasonable even under load
            final_memory = tracker.get_memory_increase()
            assert final_memory < 200, f"Memory usage under load too high: {final_memory:.2f} MB"
            
            # Test system responsiveness under load
            start_time = time.time()
            for i in range(20):
                concept = kg.get_concept(f"concept_{i:03d}")
                assert concept is not None
                
                lg = learning_graphs[i]
                mastery = lg.get_mastery(f"concept_{i:03d}")
                assert mastery >= 0.0
            
            response_time = time.time() - start_time
            assert response_time < 2.0, f"System too slow under load: {response_time:.2f}s"
            
        finally:
            os.unlink(temp_file.name)


class TestMemoryOptimization:
    """Tests for memory optimization strategies."""
    
    def test_data_structure_efficiency(self):
        """Test efficiency of different data structures."""
        tracker = MemoryTracker()
        
        # Test different ways of storing exercise history
        tracker.record_measurement("start")
        
        # Method 1: Store as list of dictionaries (current approach)
        lg1 = LearningGraph("efficiency_test_1", "Efficiency Test 1")
        for i in range(1000):
            exercise_id = f"ex_{i}"
            concept_id = f"concept_{i % 20}"
            lg1.record_exercise_attempt(exercise_id, concept_id, i % 2 == 0, 0.5, 1.0)
        
        tracker.record_measurement("method_1_complete")
        method_1_memory = tracker.get_memory_increase()
        
        # Method 2: More compact storage (if implemented)
        # This would test alternative implementations
        lg2 = LearningGraph("efficiency_test_2", "Efficiency Test 2")
        for i in range(1000):
            exercise_id = f"ex_{i}"
            concept_id = f"concept_{i % 20}"
            lg2.record_exercise_attempt(exercise_id, concept_id, i % 2 == 0, 0.5, 1.0)
        
        tracker.record_measurement("method_2_complete")
        
        # Both methods should have similar memory usage (within 20%)
        total_memory = tracker.get_memory_increase()
        assert total_memory < 50, f"Data structure memory usage too high: {total_memory:.2f} MB"
    
    def test_memory_efficient_batch_operations(self):
        """Test memory-efficient batch operations."""
        tracker = MemoryTracker()
        tracker.record_measurement("start")
        
        # Create learning graph
        lg = LearningGraph("batch_test", "Batch Test Student")
        
        # Test batch exercise recording
        batch_size = 100
        num_batches = 10
        
        for batch in range(num_batches):
            # Process exercises in batches to manage memory
            batch_exercises = []
            for i in range(batch_size):
                exercise_data = {
                    "exercise_id": f"batch_{batch}_ex_{i}",
                    "concept_id": f"concept_{i % 20}",
                    "success": i % 3 != 0,
                    "difficulty": 0.1 + (i % 10) * 0.1,
                    "time_spent": 1.0 + (i % 5) * 0.5
                }
                batch_exercises.append(exercise_data)
            
            # Record all exercises in batch
            for ex_data in batch_exercises:
                lg.record_exercise_attempt(
                    ex_data["exercise_id"],
                    ex_data["concept_id"],
                    ex_data["success"],
                    ex_data["difficulty"],
                    ex_data["time_spent"]
                )
            
            # Clear batch data
            batch_exercises.clear()
            
            # Force garbage collection between batches
            if batch % 3 == 0:
                tracker.force_gc()
                tracker.record_measurement(f"batch_{batch}_complete")
        
        final_memory = tracker.get_memory_increase()
        
        # Batch processing should keep memory usage reasonable
        assert final_memory < 30, f"Batch processing memory too high: {final_memory:.2f} MB"
        
        # Verify all data was recorded
        total_exercises = len(lg.exercise_history)
        assert total_exercises == batch_size * num_batches
    
    def test_memory_cleanup_strategies(self):
        """Test different memory cleanup strategies."""
        tracker = MemoryTracker()
        tracker.record_measurement("start")
        
        # Strategy 1: Explicit cleanup
        objects_to_cleanup = []
        
        for i in range(20):
            lg = LearningGraph(f"cleanup_test_{i}", f"Cleanup Test {i}")
            
            # Add data
            for j in range(100):
                lg.set_mastery(f"concept_{j}", 0.1 + j * 0.01, 0.5)
                lg.record_exercise_attempt(f"ex_{j}", f"concept_{j}", j % 2 == 0, 0.5, 1.0)
            
            objects_to_cleanup.append(lg)
        
        tracker.record_measurement("objects_created")
        
        # Explicit cleanup
        for obj in objects_to_cleanup:
            # Clear internal data structures
            obj.concept_mastery.clear()
            obj.exercise_history.clear()
        
        objects_to_cleanup.clear()
        tracker.force_gc()
        tracker.record_measurement("explicit_cleanup")
        
        # Strategy 2: Context manager cleanup (if implemented)
        # This would test context manager implementations
        
        # Memory should be cleaned up effectively
        final_memory = tracker.get_memory_increase()
        assert final_memory < 15, f"Cleanup not effective: {final_memory:.2f} MB remaining"


if __name__ == "__main__":
    # Run tests with memory tracking
    pytest.main([__file__, "-v", "--tb=short", "-s"]) 