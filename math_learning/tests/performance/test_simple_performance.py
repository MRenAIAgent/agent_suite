"""
Simple performance tests for math_learning system.

Tests basic performance characteristics without complex dependencies.
"""

import pytest
import sys
import os
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)


class PerformanceBenchmark:
    """Utility for measuring performance."""
    
    def __init__(self):
        self.measurements = defaultdict(list)
    
    def time_operation(self, operation: Callable, name: str, *args, **kwargs) -> Any:
        """Time an operation and store the result."""
        start_time = time.perf_counter()
        result = operation(*args, **kwargs)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        self.measurements[name].append(duration)
        
        return result
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named operation."""
        if name not in self.measurements:
            return {}
        
        times = self.measurements[name]
        return {
            "count": len(times),
            "total": sum(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0
        }
    
    def compare_operations(self, name1: str, name2: str) -> Dict[str, Any]:
        """Compare two operations."""
        stats1 = self.get_stats(name1)
        stats2 = self.get_stats(name2)
        
        if not stats1 or not stats2:
            return {"error": "Missing data for comparison"}
        
        return {
            "operation1": name1,
            "operation2": name2,
            "mean_ratio": stats1["mean"] / stats2["mean"],
            "median_ratio": stats1["median"] / stats2["median"],
            "faster_operation": name1 if stats1["mean"] < stats2["mean"] else name2
        }


class SimpleLearningSystem:
    """Simple learning system for performance testing."""
    
    def __init__(self):
        self.user_data = {}
        self.concept_graph = {}
        self.learning_records = defaultdict(list)
        self.lock = threading.Lock()
    
    def add_user(self, user_id: str, initial_data: Optional[Dict] = None):
        """Add a user to the system."""
        with self.lock:
            self.user_data[user_id] = initial_data or {}
    
    def record_learning(self, user_id: str, concept: str, score: float, metadata: Optional[Dict] = None):
        """Record a learning event."""
        record = {
            "user_id": user_id,
            "concept": concept,
            "score": score,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        with self.lock:
            self.learning_records[user_id].append(record)
    
    def get_user_mastery(self, user_id: str, concept: str) -> Optional[float]:
        """Get user's mastery level for a concept."""
        with self.lock:
            records = [r for r in self.learning_records[user_id] if r["concept"] == concept]
            if not records:
                return None
            
            # Return the latest score
            return max(records, key=lambda x: x["timestamp"])["score"]
    
    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user progress."""
        with self.lock:
            records = self.learning_records[user_id]
            
            concepts = set(r["concept"] for r in records)
            mastery_levels = {}
            
            for concept in concepts:
                concept_records = [r for r in records if r["concept"] == concept]
                latest_score = max(concept_records, key=lambda x: x["timestamp"])["score"]
                mastery_levels[concept] = latest_score
            
            return {
                "user_id": user_id,
                "total_records": len(records),
                "concepts_attempted": len(concepts),
                "mastery_levels": mastery_levels,
                "average_mastery": statistics.mean(mastery_levels.values()) if mastery_levels else 0.0
            }
    
    def bulk_record_learning(self, records: List[Dict[str, Any]]):
        """Record multiple learning events efficiently."""
        with self.lock:
            for record in records:
                user_id = record["user_id"]
                self.learning_records[user_id].append({
                    "user_id": user_id,
                    "concept": record["concept"],
                    "score": record["score"],
                    "metadata": record.get("metadata", {}),
                    "timestamp": record.get("timestamp", time.time())
                })
    
    def search_records(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search learning records based on criteria."""
        results = []
        
        with self.lock:
            for user_id, records in self.learning_records.items():
                for record in records:
                    match = True
                    
                    for key, value in criteria.items():
                        if key not in record or record[key] != value:
                            match = False
                            break
                    
                    if match:
                        results.append(record.copy())
        
        return results


class TestBasicPerformance:
    """Test basic performance characteristics."""
    
    def test_benchmark_creation(self):
        """Test creating performance benchmarks."""
        benchmark = PerformanceBenchmark()
        assert len(benchmark.measurements) == 0
        
        # Time a simple operation
        def simple_operation():
            return sum(range(1000))
        
        result = benchmark.time_operation(simple_operation, "sum_test")
        assert result == sum(range(1000))
        
        stats = benchmark.get_stats("sum_test")
        assert stats["count"] == 1
        assert stats["mean"] > 0
    
    def test_data_structure_performance(self):
        """Test performance of different data structures."""
        benchmark = PerformanceBenchmark()
        
        # Test list operations
        def list_operations():
            data = []
            for i in range(1000):
                data.append(f"item_{i}")
            return len(data)
        
        # Test dict operations
        def dict_operations():
            data = {}
            for i in range(1000):
                data[f"key_{i}"] = f"item_{i}"
            return len(data)
        
        # Time both operations multiple times
        for _ in range(5):
            benchmark.time_operation(list_operations, "list_ops")
            benchmark.time_operation(dict_operations, "dict_ops")
        
        list_stats = benchmark.get_stats("list_ops")
        dict_stats = benchmark.get_stats("dict_ops")
        
        assert list_stats["count"] == 5
        assert dict_stats["count"] == 5
        
        # Both should be reasonably fast
        assert list_stats["mean"] < 0.1  # Less than 100ms
        assert dict_stats["mean"] < 0.1  # Less than 100ms
    
    def test_learning_system_creation_performance(self):
        """Test performance of creating learning system components."""
        benchmark = PerformanceBenchmark()
        
        def create_system():
            system = SimpleLearningSystem()
            # Add some initial users
            for i in range(100):
                system.add_user(f"user_{i}", {"level": i % 5})
            return system
        
        # Time system creation multiple times
        for _ in range(3):
            system = benchmark.time_operation(create_system, "system_creation")
            assert len(system.user_data) == 100
        
        stats = benchmark.get_stats("system_creation")
        assert stats["count"] == 3
        assert stats["mean"] < 1.0  # Should be less than 1 second
    
    def test_single_record_performance(self):
        """Test performance of recording single learning events."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        # Add a user
        system.add_user("test_user")
        
        def record_single_event():
            system.record_learning("test_user", "algebra", 0.8, {"attempt": 1})
        
        # Time multiple single records
        for i in range(100):
            benchmark.time_operation(record_single_event, "single_record")
        
        stats = benchmark.get_stats("single_record")
        assert stats["count"] == 100
        assert stats["mean"] < 0.01  # Should be very fast (< 10ms)
        
        # Verify records were stored
        progress = system.get_user_progress("test_user")
        assert progress["total_records"] == 100
    
    def test_bulk_record_performance(self):
        """Test performance of bulk recording."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        # Add users
        for i in range(10):
            system.add_user(f"user_{i}")
        
        def create_bulk_records():
            records = []
            for i in range(100):
                records.append({
                    "user_id": f"user_{i % 10}",
                    "concept": f"concept_{i % 20}",
                    "score": 0.5 + (i % 10) * 0.05,
                    "metadata": {"batch": "performance_test"}
                })
            return records
        
        def record_bulk_events(records):
            system.bulk_record_learning(records)
        
        # Create test data
        test_records = create_bulk_records()
        
        # Time bulk recording
        for _ in range(5):
            benchmark.time_operation(record_bulk_events, "bulk_record", test_records)
        
        stats = benchmark.get_stats("bulk_record")
        assert stats["count"] == 5
        assert stats["mean"] < 0.1  # Should be reasonably fast
        
        # Verify records were stored
        total_records = sum(len(records) for records in system.learning_records.values())
        assert total_records == 500  # 5 runs * 100 records each
    
    def test_query_performance(self):
        """Test performance of querying user data."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        # Setup test data
        for i in range(50):
            system.add_user(f"user_{i}")
            for j in range(20):
                system.record_learning(f"user_{i}", f"concept_{j}", 0.5 + j * 0.02)
        
        def get_user_progress(user_id):
            return system.get_user_progress(user_id)
        
        def get_user_mastery(user_id, concept):
            return system.get_user_mastery(user_id, concept)
        
        # Time progress queries
        for i in range(20):
            benchmark.time_operation(get_user_progress, "progress_query", f"user_{i}")
        
        # Time mastery queries
        for i in range(20):
            benchmark.time_operation(get_user_mastery, "mastery_query", f"user_{i}", "concept_10")
        
        progress_stats = benchmark.get_stats("progress_query")
        mastery_stats = benchmark.get_stats("mastery_query")
        
        assert progress_stats["count"] == 20
        assert mastery_stats["count"] == 20
        
        # Queries should be fast
        assert progress_stats["mean"] < 0.01  # < 10ms
        assert mastery_stats["mean"] < 0.005  # < 5ms


class TestLoadTesting:
    """Test system behavior under load."""
    
    def test_concurrent_user_creation(self):
        """Test creating many users concurrently."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        def create_users_batch(start_id, count):
            for i in range(count):
                system.add_user(f"concurrent_user_{start_id + i}")
        
        # Create users concurrently
        def concurrent_user_creation():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(10):
                    futures.append(executor.submit(create_users_batch, i * 100, 100))
                
                for future in as_completed(futures):
                    future.result()
        
        benchmark.time_operation(concurrent_user_creation, "concurrent_users")
        
        stats = benchmark.get_stats("concurrent_users")
        assert stats["count"] == 1
        assert stats["mean"] < 2.0  # Should complete in reasonable time
        
        # Verify all users were created
        assert len(system.user_data) == 1000
    
    def test_concurrent_learning_records(self):
        """Test recording learning events concurrently."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        # Setup users
        for i in range(20):
            system.add_user(f"load_user_{i}")
        
        def record_batch(user_id, batch_size):
            for i in range(batch_size):
                system.record_learning(
                    user_id, 
                    f"concept_{i % 10}", 
                    0.5 + (i % 10) * 0.05,
                    {"batch": "load_test"}
                )
        
        def concurrent_recording():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(20):
                    futures.append(executor.submit(record_batch, f"load_user_{i}", 50))
                
                for future in as_completed(futures):
                    future.result()
        
        benchmark.time_operation(concurrent_recording, "concurrent_records")
        
        stats = benchmark.get_stats("concurrent_records")
        assert stats["count"] == 1
        assert stats["mean"] < 5.0  # Should complete within 5 seconds
        
        # Verify records were created
        total_records = sum(len(records) for records in system.learning_records.values())
        assert total_records == 1000  # 20 users * 50 records each
    
    def test_mixed_workload_performance(self):
        """Test performance under mixed read/write workload."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        # Setup initial data
        for i in range(10):
            system.add_user(f"mixed_user_{i}")
            for j in range(10):
                system.record_learning(f"mixed_user_{i}", f"concept_{j}", 0.6)
        
        def mixed_operations(thread_id):
            operations = 0
            for i in range(100):
                if i % 3 == 0:
                    # Write operation
                    system.record_learning(
                        f"mixed_user_{i % 10}", 
                        f"concept_{i % 15}", 
                        0.7 + (i % 5) * 0.05
                    )
                elif i % 3 == 1:
                    # Read operation - get progress
                    system.get_user_progress(f"mixed_user_{i % 10}")
                else:
                    # Read operation - get mastery
                    system.get_user_mastery(f"mixed_user_{i % 10}", f"concept_{i % 10}")
                operations += 1
            return operations
        
        def run_mixed_workload():
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(mixed_operations, i) for i in range(3)]
                results = [future.result() for future in as_completed(futures)]
            return sum(results)
        
        total_ops = benchmark.time_operation(run_mixed_workload, "mixed_workload")
        
        stats = benchmark.get_stats("mixed_workload")
        assert stats["count"] == 1
        assert stats["mean"] < 3.0  # Should complete within 3 seconds
        assert total_ops == 300  # 3 threads * 100 operations each
    
    def test_search_performance_under_load(self):
        """Test search performance with large datasets."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        # Create large dataset
        for i in range(100):
            system.add_user(f"search_user_{i}")
            for j in range(50):
                system.record_learning(
                    f"search_user_{i}", 
                    f"concept_{j % 20}", 
                    0.5 + (j % 10) * 0.05,
                    {"category": f"cat_{j % 5}", "difficulty": j % 3}
                )
        
        def search_by_concept(concept):
            return system.search_records({"concept": concept})
        
        def search_by_category(category):
            return system.search_records({"metadata.category": category})
        
        def search_by_user(user_id):
            return system.search_records({"user_id": user_id})
        
        # Time different types of searches
        for i in range(10):
            benchmark.time_operation(search_by_concept, "concept_search", f"concept_{i}")
            benchmark.time_operation(search_by_user, "user_search", f"search_user_{i}")
        
        concept_stats = benchmark.get_stats("concept_search")
        user_stats = benchmark.get_stats("user_search")
        
        assert concept_stats["count"] == 10
        assert user_stats["count"] == 10
        
        # Searches should be reasonably fast even with large dataset
        assert concept_stats["mean"] < 0.1  # < 100ms
        assert user_stats["mean"] < 0.05  # < 50ms


class TestStressTesting:
    """Test system behavior under stress conditions."""
    
    def test_memory_stress(self):
        """Test system behavior with large amounts of data."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        def create_large_dataset():
            # Create many users with lots of data
            for i in range(200):
                system.add_user(f"stress_user_{i}")
                
                # Add many learning records
                records = []
                for j in range(100):
                    records.append({
                        "user_id": f"stress_user_{i}",
                        "concept": f"concept_{j % 30}",
                        "score": 0.4 + (j % 12) * 0.05,
                        "metadata": {
                            "session": j // 10,
                            "attempt": j % 10,
                            "data": f"large_string_data_{j}" * 10  # Make it larger
                        }
                    })
                
                system.bulk_record_learning(records)
        
        benchmark.time_operation(create_large_dataset, "large_dataset")
        
        stats = benchmark.get_stats("large_dataset")
        assert stats["count"] == 1
        # Should complete even if it takes longer
        assert stats["mean"] < 30.0  # 30 seconds max
        
        # Verify data was created
        assert len(system.user_data) == 200
        total_records = sum(len(records) for records in system.learning_records.values())
        assert total_records == 20000  # 200 users * 100 records each
    
    def test_high_concurrency_stress(self):
        """Test system under high concurrency."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        # Pre-populate some data
        for i in range(50):
            system.add_user(f"stress_concurrent_user_{i}")
        
        def high_concurrency_operations():
            def worker_operations(worker_id):
                operations = 0
                for i in range(50):
                    # Mix of operations
                    system.record_learning(
                        f"stress_concurrent_user_{i % 50}",
                        f"concept_{i % 20}",
                        0.5 + (i % 10) * 0.05
                    )
                    
                    if i % 5 == 0:
                        system.get_user_progress(f"stress_concurrent_user_{i % 50}")
                    
                    operations += 1
                return operations
            
            # Use many concurrent workers
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(worker_operations, i) for i in range(8)]
                results = [future.result() for future in as_completed(futures)]
            
            return sum(results)
        
        total_ops = benchmark.time_operation(high_concurrency_operations, "high_concurrency")
        
        stats = benchmark.get_stats("high_concurrency")
        assert stats["count"] == 1
        assert stats["mean"] < 10.0  # Should complete within 10 seconds
        assert total_ops == 400  # 8 workers * 50 operations each
    
    def test_rapid_fire_queries(self):
        """Test system with rapid consecutive queries."""
        benchmark = PerformanceBenchmark()
        system = SimpleLearningSystem()
        
        # Setup data
        for i in range(20):
            system.add_user(f"rapid_user_{i}")
            for j in range(30):
                system.record_learning(f"rapid_user_{i}", f"concept_{j}", 0.6)
        
        def rapid_queries():
            results = []
            for i in range(1000):  # Many rapid queries
                if i % 2 == 0:
                    result = system.get_user_progress(f"rapid_user_{i % 20}")
                else:
                    result = system.get_user_mastery(f"rapid_user_{i % 20}", f"concept_{i % 30}")
                results.append(result)
            return len(results)
        
        result_count = benchmark.time_operation(rapid_queries, "rapid_queries")
        
        stats = benchmark.get_stats("rapid_queries")
        assert stats["count"] == 1
        assert stats["mean"] < 5.0  # Should handle rapid queries efficiently
        assert result_count == 1000


class TestScalabilityAnalysis:
    """Test scalability characteristics."""
    
    def test_user_scaling(self):
        """Test how performance scales with number of users."""
        benchmark = PerformanceBenchmark()
        
        def test_with_user_count(user_count):
            system = SimpleLearningSystem()
            
            # Create users
            for i in range(user_count):
                system.add_user(f"scale_user_{i}")
                for j in range(10):  # 10 records per user
                    system.record_learning(f"scale_user_{i}", f"concept_{j}", 0.7)
            
            # Test query performance
            start_time = time.perf_counter()
            for i in range(min(50, user_count)):  # Query up to 50 users
                system.get_user_progress(f"scale_user_{i}")
            end_time = time.perf_counter()
            
            return end_time - start_time
        
        # Test with different user counts
        user_counts = [10, 50, 100, 200]
        times = []
        
        for count in user_counts:
            duration = benchmark.time_operation(test_with_user_count, f"users_{count}", count)
            times.append(duration)
        
        # Performance shouldn't degrade too badly with more users
        # (This is a basic check - real scalability would need more sophisticated analysis)
        assert len(times) == 4
        
        # The increase shouldn't be more than linear
        ratio_50_to_10 = times[1] / times[0] if times[0] > 0 else 1
        ratio_200_to_50 = times[3] / times[1] if times[1] > 0 else 1
        
        # These are loose bounds - real systems would have tighter requirements
        assert ratio_50_to_10 < 10  # 5x users shouldn't be 10x slower
        assert ratio_200_to_50 < 8   # 4x users shouldn't be 8x slower
    
    def test_data_volume_scaling(self):
        """Test how performance scales with data volume per user."""
        benchmark = PerformanceBenchmark()
        
        def test_with_record_count(records_per_user):
            system = SimpleLearningSystem()
            
            # Create 20 users with varying amounts of data
            for i in range(20):
                system.add_user(f"volume_user_{i}")
                for j in range(records_per_user):
                    system.record_learning(
                        f"volume_user_{i}", 
                        f"concept_{j % 50}", 
                        0.5 + (j % 10) * 0.05
                    )
            
            # Test progress query performance
            start_time = time.perf_counter()
            for i in range(10):
                system.get_user_progress(f"volume_user_{i}")
            end_time = time.perf_counter()
            
            return end_time - start_time
        
        # Test with different record counts per user
        record_counts = [10, 50, 100, 200]
        times = []
        
        for count in record_counts:
            duration = benchmark.time_operation(test_with_record_count, f"records_{count}", count)
            times.append(duration)
        
        # Performance should scale reasonably with data volume
        assert len(times) == 4
        
        # Check that performance doesn't degrade too severely
        final_to_initial_ratio = times[-1] / times[0] if times[0] > 0 else 1
        assert final_to_initial_ratio < 20  # 20x data shouldn't be more than 20x slower


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 