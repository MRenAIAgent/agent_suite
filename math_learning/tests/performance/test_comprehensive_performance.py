#!/usr/bin/env python3
"""
Comprehensive Performance and Scalability Tests

This test suite provides comprehensive coverage for performance and scalability
of the math learning system, including:
- Load testing with realistic user scenarios
- Stress testing with extreme conditions
- Concurrent operations and thread safety
- Performance benchmarks and regression testing
- Scalability analysis
- Resource utilization monitoring
- Response time analysis
"""

import pytest
import time
import threading
import multiprocessing
import concurrent.futures
import tempfile
import os
import json
import statistics
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import sys

# Add project root to path
# From math_learning/tests/performance/test_comprehensive_performance.py
# Go up 4 levels: performance -> tests -> math_learning -> agent_suite
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Performance testing imports
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.algebra_graph import AlgebraGraph
from math_learning.config.storage_config import MathLearningStorageConfig, StorageBackend
from math_learning.recommendation.gap_analyzer import GapAnalyzer
from math_learning.recommendation.recommender import Recommender
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.exercises.exercise import Exercise


class PerformanceBenchmark:
    """Utility class for performance benchmarking."""
    
    def __init__(self, name: str):
        self.name = name
        self.measurements = []
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing measurement."""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        """Stop timing measurement."""
        if self.start_time is None:
            raise ValueError("Benchmark not started")
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        self.measurements.append(duration)
        return duration
    
    def measure(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        self.start()
        try:
            result = func(*args, **kwargs)
            self.stop()
            return result
        except Exception as e:
            self.stop()
            raise e
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.measurements:
            return {}
        
        return {
            "count": len(self.measurements),
            "total_time": sum(self.measurements),
            "avg_time": statistics.mean(self.measurements),
            "min_time": min(self.measurements),
            "max_time": max(self.measurements),
            "median_time": statistics.median(self.measurements),
            "std_dev": statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0
        }


class TestBasicPerformance:
    """Tests for basic performance characteristics."""
    
    def test_learning_graph_creation_performance(self):
        """Test performance of learning graph creation."""
        benchmark = PerformanceBenchmark("learning_graph_creation")
        
        # Test single creation
        def create_learning_graph():
            lg = LearningGraph("perf_test_001", "Performance Test Student")
            # Add some initial data
            for i in range(50):
                lg.set_mastery(f"concept_{i}", 0.1 + i * 0.01, 0.5)
            return lg
        
        # Measure multiple creations
        learning_graphs = []
        for i in range(100):
            lg = benchmark.measure(create_learning_graph)
            learning_graphs.append(lg)
        
        stats = benchmark.get_stats()
        
        # Performance requirements
        assert stats["avg_time"] < 0.01, f"Learning graph creation too slow: {stats['avg_time']:.4f}s avg"
        assert stats["max_time"] < 0.05, f"Learning graph creation max too slow: {stats['max_time']:.4f}s"
        assert stats["std_dev"] < 0.01, f"Learning graph creation too inconsistent: {stats['std_dev']:.4f}s std dev"
    
    def test_knowledge_graph_query_performance(self):
        """Test performance of knowledge graph queries."""
        # Create knowledge graph
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write('{"concepts": {}, "relationships": []}')
        temp_file.close()
        
        try:
            config = MathLearningStorageConfig(
                backend=StorageBackend.NETWORKX,
                networkx_persistence_file=temp_file.name,
                networkx_auto_save=False
            )
            
            kg = KnowledgeGraph(name="Performance Test", config=config)
            
            # Add concepts and relationships
            concepts = []
            for i in range(200):
                concept = Concept(
                    name=f"Concept {i}",
                    concept_id=f"concept_{i:03d}",
                    difficulty=0.1 + (i % 10) * 0.1,
                    time_to_master=5 + (i % 15)
                )
                concepts.append(concept)
                kg.add_concept(concept)
            
            # Add relationships
            for i in range(199):
                kg.add_prerequisite(f"concept_{i:03d}", f"concept_{i+1:03d}")
            
            # Test query performance
            benchmark = PerformanceBenchmark("concept_retrieval")
            
            # Test concept retrieval
            for i in range(100):
                concept = benchmark.measure(kg.get_concept, f"concept_{i:03d}")
                assert concept is not None
            
            retrieval_stats = benchmark.get_stats()
            
            # Test prerequisite queries
            prereq_benchmark = PerformanceBenchmark("prerequisite_queries")
            
            for i in range(50, 150):  # Test middle concepts
                prereqs = prereq_benchmark.measure(kg.get_prerequisites, f"concept_{i:03d}")
                assert len(prereqs) > 0
            
            prereq_stats = prereq_benchmark.get_stats()
            
            # Performance requirements
            assert retrieval_stats["avg_time"] < 0.001, f"Concept retrieval too slow: {retrieval_stats['avg_time']:.6f}s"
            assert prereq_stats["avg_time"] < 0.01, f"Prerequisite queries too slow: {prereq_stats['avg_time']:.6f}s"
            
        finally:
            os.unlink(temp_file.name)
    
    def test_exercise_recording_performance(self):
        """Test performance of exercise recording."""
        lg = LearningGraph("perf_test_exercises", "Exercise Performance Test")
        
        benchmark = PerformanceBenchmark("exercise_recording")
        
        # Test exercise recording performance
        for i in range(1000):
            exercise_id = f"ex_{i:04d}"
            concept_id = f"concept_{i % 20}"
            success = i % 2 == 0
            difficulty = 0.1 + (i % 10) * 0.1
            
            benchmark.measure(
                lg.record_exercise_attempt,
                exercise_id, concept_id, success, difficulty, 1.0
            )
        
        stats = benchmark.get_stats()
        
        # Performance requirements
        assert stats["avg_time"] < 0.001, f"Exercise recording too slow: {stats['avg_time']:.6f}s avg"
        assert stats["max_time"] < 0.01, f"Exercise recording max too slow: {stats['max_time']:.6f}s"
        
        # Verify all exercises were recorded
        assert len(lg.exercise_history) == 1000
    
    def test_mastery_calculation_performance(self):
        """Test performance of mastery calculations."""
        lg = LearningGraph("perf_test_mastery", "Mastery Performance Test")
        
        # Pre-populate with exercise data
        for i in range(100):
            concept_id = f"concept_{i}"
            for j in range(50):  # 50 exercises per concept
                exercise_id = f"ex_{i}_{j}"
                success = j % 3 != 0  # ~67% success rate
                lg.record_exercise_attempt(exercise_id, concept_id, success, 0.5, 1.0)
        
        benchmark = PerformanceBenchmark("mastery_calculation")
        
        # Test mastery retrieval performance
        for i in range(100):
            concept_id = f"concept_{i}"
            mastery = benchmark.measure(lg.get_mastery, concept_id)
            assert 0.0 <= mastery <= 1.0
        
        stats = benchmark.get_stats()
        
        # Performance requirements
        assert stats["avg_time"] < 0.001, f"Mastery calculation too slow: {stats['avg_time']:.6f}s avg"


class TestLoadTesting:
    """Tests for system behavior under realistic load."""
    
    def test_multiple_students_concurrent_learning(self):
        """Test system with multiple students learning concurrently."""
        num_students = 50
        exercises_per_student = 100
        
        # Create knowledge graph
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
            for i in range(50):
                concept = Concept(
                    name=f"Concept {i}",
                    concept_id=f"concept_{i:02d}",
                    difficulty=0.1 + (i % 10) * 0.1,
                    time_to_master=5 + (i % 15)
                )
                kg.add_concept(concept)
            
            # Create students
            students = {}
            for i in range(num_students):
                student_id = f"student_{i:03d}"
                students[student_id] = LearningGraph(student_id, f"Load Test Student {i}")
            
            # Simulate concurrent learning
            def student_learning_session(student_id: str, iterations: int):
                """Simulate a student's learning session."""
                lg = students[student_id]
                session_stats = {
                    "exercises_completed": 0,
                    "concepts_practiced": set(),
                    "avg_response_time": 0.0,
                    "total_time": 0.0
                }
                
                start_time = time.perf_counter()
                
                for i in range(iterations):
                    concept_id = f"concept_{i % 50:02d}"
                    exercise_id = f"{student_id}_ex_{i:03d}"
                    
                    # Record exercise attempt
                    exercise_start = time.perf_counter()
                    success = (i + hash(student_id)) % 3 != 0  # Deterministic but varied
                    difficulty = 0.1 + (i % 10) * 0.1
                    
                    lg.record_exercise_attempt(exercise_id, concept_id, success, difficulty, 1.0)
                    exercise_time = time.perf_counter() - exercise_start
                    
                    session_stats["exercises_completed"] += 1
                    session_stats["concepts_practiced"].add(concept_id)
                    session_stats["avg_response_time"] += exercise_time
                
                end_time = time.perf_counter()
                session_stats["total_time"] = end_time - start_time
                session_stats["avg_response_time"] /= iterations
                
                return session_stats
            
            # Run concurrent sessions
            start_time = time.perf_counter()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for student_id in students.keys():
                    future = executor.submit(student_learning_session, student_id, exercises_per_student)
                    futures.append((student_id, future))
                
                # Collect results
                session_results = {}
                for student_id, future in futures:
                    session_results[student_id] = future.result()
            
            total_time = time.perf_counter() - start_time
            
            # Analyze results
            total_exercises = sum(stats["exercises_completed"] for stats in session_results.values())
            avg_response_times = [stats["avg_response_time"] for stats in session_results.values()]
            avg_response_time = statistics.mean(avg_response_times)
            
            # Performance assertions
            assert total_time < 30.0, f"Load test took too long: {total_time:.2f}s"
            assert avg_response_time < 0.01, f"Average response time too slow: {avg_response_time:.6f}s"
            assert total_exercises == num_students * exercises_per_student
            
            # Verify data integrity
            for student_id, lg in students.items():
                assert len(lg.exercise_history) == exercises_per_student
                
                # Check that mastery values are reasonable
                for concept_id in [f"concept_{i:02d}" for i in range(50)]:
                    mastery = lg.get_mastery(concept_id)
                    if mastery > 0:  # Only check concepts that were practiced
                        assert 0.0 <= mastery <= 1.0
            
        finally:
            os.unlink(temp_file.name)
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large learning graph
        lg = LearningGraph("large_dataset_test", "Large Dataset Test")
        
        # Add large amount of data
        num_concepts = 500
        exercises_per_concept = 200
        
        print(f"Creating large dataset: {num_concepts} concepts, {exercises_per_concept} exercises each")
        
        start_time = time.perf_counter()
        
        # Add mastery data for all concepts
        for i in range(num_concepts):
            concept_id = f"large_concept_{i:04d}"
            lg.set_mastery(concept_id, 0.1 + (i % 10) * 0.09, 0.5 + (i % 5) * 0.1)
        
        mastery_time = time.perf_counter() - start_time
        
        # Add exercise history
        exercise_start = time.perf_counter()
        
        for concept_idx in range(num_concepts):
            concept_id = f"large_concept_{concept_idx:04d}"
            
            for ex_idx in range(exercises_per_concept):
                exercise_id = f"large_ex_{concept_idx:04d}_{ex_idx:03d}"
                success = (concept_idx + ex_idx) % 3 != 0
                difficulty = 0.1 + (ex_idx % 10) * 0.1
                
                lg.record_exercise_attempt(exercise_id, concept_id, success, difficulty, 1.0)
        
        exercise_time = time.perf_counter() - exercise_start
        
        # Test query performance on large dataset
        query_start = time.perf_counter()
        
        # Test random access performance
        import random
        for _ in range(1000):
            concept_idx = random.randint(0, num_concepts - 1)
            concept_id = f"large_concept_{concept_idx:04d}"
            mastery = lg.get_mastery(concept_id)
            assert 0.0 <= mastery <= 1.0
        
        query_time = time.perf_counter() - query_start
        
        total_time = time.perf_counter() - start_time
        
        # Performance assertions
        total_exercises = num_concepts * exercises_per_concept
        
        print(f"Large dataset performance:")
        print(f"  Mastery setup: {mastery_time:.2f}s")
        print(f"  Exercise recording: {exercise_time:.2f}s ({total_exercises} exercises)")
        print(f"  Query performance: {query_time:.2f}s (1000 queries)")
        print(f"  Total time: {total_time:.2f}s")
        
        assert mastery_time < 5.0, f"Mastery setup too slow: {mastery_time:.2f}s"
        assert exercise_time < 60.0, f"Exercise recording too slow: {exercise_time:.2f}s"
        assert query_time < 2.0, f"Query performance too slow: {query_time:.2f}s"
        assert total_time < 70.0, f"Total time too slow: {total_time:.2f}s"
        
        # Verify data integrity
        assert len(lg.concept_mastery) == num_concepts
        assert len(lg.exercise_history) == total_exercises


class TestStressTesting:
    """Tests for system behavior under extreme conditions."""
    
    def test_extreme_concurrent_access(self):
        """Test system under extreme concurrent access."""
        # Create shared resources
        shared_lg = LearningGraph("stress_test", "Stress Test Student")
        
        # Pre-populate with some data
        for i in range(100):
            shared_lg.set_mastery(f"stress_concept_{i}", 0.5, 0.5)
        
        # Stress test parameters
        num_threads = 20
        operations_per_thread = 500
        
        def stress_worker(worker_id: int, operations: int):
            """Worker function for stress testing."""
            worker_stats = {
                "operations_completed": 0,
                "errors": 0,
                "avg_time": 0.0
            }
            
            total_time = 0.0
            
            for i in range(operations):
                try:
                    op_start = time.perf_counter()
                    
                    # Perform random operations
                    operation = i % 4
                    concept_id = f"stress_concept_{i % 100}"
                    
                    if operation == 0:
                        # Record exercise
                        exercise_id = f"stress_ex_{worker_id}_{i}"
                        shared_lg.record_exercise_attempt(exercise_id, concept_id, i % 2 == 0, 0.5, 1.0)
                    elif operation == 1:
                        # Get mastery
                        mastery = shared_lg.get_mastery(concept_id)
                        assert 0.0 <= mastery <= 1.0
                    elif operation == 2:
                        # Set mastery
                        new_mastery = 0.1 + (i % 9) * 0.1
                        shared_lg.set_mastery(concept_id, new_mastery, 0.5)
                    else:
                        # Get confidence
                        confidence = shared_lg.get_confidence(concept_id)
                        assert 0.0 <= confidence <= 1.0
                    
                    op_time = time.perf_counter() - op_start
                    total_time += op_time
                    worker_stats["operations_completed"] += 1
                    
                except Exception as e:
                    worker_stats["errors"] += 1
                    print(f"Worker {worker_id} error on operation {i}: {e}")
            
            worker_stats["avg_time"] = total_time / operations if operations > 0 else 0.0
            return worker_stats
        
        # Run stress test
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for worker_id in range(num_threads):
                future = executor.submit(stress_worker, worker_id, operations_per_thread)
                futures.append(future)
            
            # Collect results
            worker_results = [future.result() for future in futures]
        
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        total_operations = sum(result["operations_completed"] for result in worker_results)
        total_errors = sum(result["errors"] for result in worker_results)
        avg_operation_times = [result["avg_time"] for result in worker_results]
        overall_avg_time = statistics.mean(avg_operation_times)
        
        print(f"Stress test results:")
        print(f"  Total operations: {total_operations}")
        print(f"  Total errors: {total_errors}")
        print(f"  Error rate: {total_errors/total_operations*100:.2f}%")
        print(f"  Average operation time: {overall_avg_time:.6f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Operations per second: {total_operations/total_time:.2f}")
        
        # Stress test assertions
        expected_operations = num_threads * operations_per_thread
        assert total_operations >= expected_operations * 0.95, f"Too many failed operations: {total_operations}/{expected_operations}"
        assert total_errors < expected_operations * 0.05, f"Error rate too high: {total_errors}/{expected_operations}"
        assert overall_avg_time < 0.01, f"Operations too slow under stress: {overall_avg_time:.6f}s"
    
    def test_memory_pressure_performance(self):
        """Test performance under memory pressure."""
        # Create many learning graphs to create memory pressure
        learning_graphs = []
        
        # Create memory pressure
        for i in range(100):
            lg = LearningGraph(f"memory_pressure_{i}", f"Memory Pressure Test {i}")
            
            # Add substantial data to each
            for j in range(200):
                concept_id = f"concept_{j}"
                lg.set_mastery(concept_id, 0.1 + (j % 10) * 0.09, 0.5)
                
                # Add exercise history
                for k in range(50):
                    exercise_id = f"ex_{j}_{k}"
                    lg.record_exercise_attempt(exercise_id, concept_id, k % 2 == 0, 0.5, 1.0)
            
            learning_graphs.append(lg)
        
        # Test performance under memory pressure
        benchmark = PerformanceBenchmark("memory_pressure_operations")
        
        # Perform operations on random learning graphs
        import random
        for i in range(1000):
            lg = random.choice(learning_graphs)
            concept_id = f"concept_{i % 200}"
            
            # Measure operation time under memory pressure
            mastery = benchmark.measure(lg.get_mastery, concept_id)
            assert 0.0 <= mastery <= 1.0
        
        stats = benchmark.get_stats()
        
        # Performance should still be reasonable under memory pressure
        assert stats["avg_time"] < 0.01, f"Performance degraded under memory pressure: {stats['avg_time']:.6f}s"
        assert stats["max_time"] < 0.1, f"Max time too slow under memory pressure: {stats['max_time']:.6f}s"


class TestScalabilityAnalysis:
    """Tests for scalability analysis and limits."""
    
    def test_scalability_with_increasing_users(self):
        """Test how performance scales with increasing number of users."""
        user_counts = [10, 50, 100, 200]
        exercises_per_user = 100
        
        scalability_results = {}
        
        for num_users in user_counts:
            print(f"Testing scalability with {num_users} users...")
            
            # Create users
            users = {}
            for i in range(num_users):
                user_id = f"scale_user_{i:04d}"
                users[user_id] = LearningGraph(user_id, f"Scale Test User {i}")
            
            # Measure performance
            start_time = time.perf_counter()
            
            def user_session(user_id: str):
                lg = users[user_id]
                for i in range(exercises_per_user):
                    concept_id = f"concept_{i % 50}"
                    exercise_id = f"{user_id}_ex_{i}"
                    lg.record_exercise_attempt(exercise_id, concept_id, i % 2 == 0, 0.5, 1.0)
            
            # Run concurrent user sessions
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_users, 20)) as executor:
                futures = [executor.submit(user_session, user_id) for user_id in users.keys()]
                for future in futures:
                    future.result()
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Calculate metrics
            total_operations = num_users * exercises_per_user
            operations_per_second = total_operations / total_time
            time_per_user = total_time / num_users
            
            scalability_results[num_users] = {
                "total_time": total_time,
                "operations_per_second": operations_per_second,
                "time_per_user": time_per_user,
                "total_operations": total_operations
            }
            
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Operations/sec: {operations_per_second:.2f}")
            print(f"  Time per user: {time_per_user:.4f}s")
        
        # Analyze scalability
        print("\nScalability Analysis:")
        for num_users, results in scalability_results.items():
            print(f"  {num_users} users: {results['operations_per_second']:.2f} ops/sec, {results['time_per_user']:.4f}s/user")
        
        # Check that performance doesn't degrade too much with scale
        base_ops_per_sec = scalability_results[user_counts[0]]["operations_per_second"]
        max_ops_per_sec = scalability_results[user_counts[-1]]["operations_per_second"]
        
        # Allow some degradation but not more than 50%
        performance_ratio = max_ops_per_sec / base_ops_per_sec
        assert performance_ratio > 0.5, f"Performance degraded too much with scale: {performance_ratio:.2f}"
    
    def test_data_size_scalability(self):
        """Test how performance scales with increasing data size."""
        data_sizes = [100, 500, 1000, 2000]  # Number of concepts per user
        
        scalability_results = {}
        
        for num_concepts in data_sizes:
            print(f"Testing with {num_concepts} concepts...")
            
            lg = LearningGraph("data_scale_test", "Data Scale Test")
            
            # Add concepts and exercise data
            setup_start = time.perf_counter()
            
            for i in range(num_concepts):
                concept_id = f"scale_concept_{i:04d}"
                lg.set_mastery(concept_id, 0.1 + (i % 10) * 0.09, 0.5)
                
                # Add some exercise history
                for j in range(20):  # 20 exercises per concept
                    exercise_id = f"scale_ex_{i}_{j}"
                    lg.record_exercise_attempt(exercise_id, concept_id, j % 2 == 0, 0.5, 1.0)
            
            setup_time = time.perf_counter() - setup_start
            
            # Test query performance
            query_start = time.perf_counter()
            
            # Perform random queries
            import random
            for _ in range(1000):
                concept_idx = random.randint(0, num_concepts - 1)
                concept_id = f"scale_concept_{concept_idx:04d}"
                mastery = lg.get_mastery(concept_id)
                assert 0.0 <= mastery <= 1.0
            
            query_time = time.perf_counter() - query_start
            avg_query_time = query_time / 1000
            
            scalability_results[num_concepts] = {
                "setup_time": setup_time,
                "query_time": query_time,
                "avg_query_time": avg_query_time,
                "total_exercises": num_concepts * 20
            }
            
            print(f"  Setup time: {setup_time:.2f}s")
            print(f"  Query time: {query_time:.4f}s (1000 queries)")
            print(f"  Avg query time: {avg_query_time:.6f}s")
        
        # Analyze data size scalability
        print("\nData Size Scalability Analysis:")
        for num_concepts, results in scalability_results.items():
            print(f"  {num_concepts} concepts: {results['avg_query_time']:.6f}s avg query time")
        
        # Check that query time doesn't grow too much with data size
        base_query_time = scalability_results[data_sizes[0]]["avg_query_time"]
        max_query_time = scalability_results[data_sizes[-1]]["avg_query_time"]
        
        # Query time should scale sub-linearly (allow 4x increase for 20x data)
        time_ratio = max_query_time / base_query_time
        data_ratio = data_sizes[-1] / data_sizes[0]
        
        assert time_ratio < data_ratio * 0.5, f"Query time scaled too poorly: {time_ratio:.2f}x time for {data_ratio:.2f}x data"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-s"]) 