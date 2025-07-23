"""
Simple memory management tests for math_learning system.

Tests basic memory usage patterns without complex dependencies.
"""

import pytest
import sys
import os
import gc
import weakref
from typing import Dict, List, Any, Optional
import psutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)


class MemoryTracker:
    """Simple memory tracking utility."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_change(self) -> float:
        """Get memory change since initialization."""
        return self.get_memory_usage() - self.initial_memory
    
    def reset(self):
        """Reset the baseline memory usage."""
        self.initial_memory = self.get_memory_usage()


class LearningDataContainer:
    """Container for learning data that can be monitored for memory usage."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.data = {}
        self.large_data = []
        self.references = set()
    
    def add_learning_record(self, concept: str, score: float, metadata: Optional[Dict] = None):
        """Add a learning record."""
        record = {
            "concept": concept,
            "score": score,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        if concept not in self.data:
            self.data[concept] = []
        self.data[concept].append(record)
    
    def add_large_dataset(self, size: int):
        """Add a large dataset for memory testing."""
        large_list = [f"data_item_{i}" for i in range(size)]
        self.large_data.append(large_list)
    
    def clear_large_data(self):
        """Clear large data to test memory cleanup."""
        self.large_data.clear()
    
    def add_reference(self, obj):
        """Add a reference to track."""
        self.references.add(id(obj))
    
    def cleanup(self):
        """Clean up all data."""
        self.data.clear()
        self.large_data.clear()
        self.references.clear()


class TestBasicMemoryUsage:
    """Test basic memory usage patterns."""
    
    def test_memory_tracker_creation(self):
        """Test creating a memory tracker."""
        tracker = MemoryTracker()
        initial_usage = tracker.get_memory_usage()
        assert initial_usage > 0
        
        change = tracker.get_memory_change()
        assert abs(change) < 1.0  # Should be very small initially
    
    def test_small_data_memory_usage(self):
        """Test memory usage with small datasets."""
        tracker = MemoryTracker()
        tracker.reset()
        
        # Create small learning containers
        containers = []
        for i in range(10):
            container = LearningDataContainer(f"user_{i}")
            for j in range(5):
                container.add_learning_record(f"concept_{j}", 0.8, {"attempt": j})
            containers.append(container)
        
        memory_change = tracker.get_memory_change()
        assert memory_change < 10.0  # Should be less than 10MB for small data
        
        # Cleanup
        containers.clear()
        gc.collect()
    
    def test_large_data_memory_usage(self):
        """Test memory usage with larger datasets."""
        tracker = MemoryTracker()
        tracker.reset()
        
        container = LearningDataContainer("test_user")
        
        # Add progressively larger datasets
        initial_memory = tracker.get_memory_usage()
        
        container.add_large_dataset(1000)  # Small dataset
        small_memory = tracker.get_memory_usage()
        
        container.add_large_dataset(10000)  # Medium dataset
        medium_memory = tracker.get_memory_usage()
        
        container.add_large_dataset(50000)  # Large dataset
        large_memory = tracker.get_memory_usage()
        
        # Memory should increase with dataset size
        assert small_memory >= initial_memory
        assert medium_memory >= small_memory
        assert large_memory >= medium_memory
        
        # Cleanup should reduce memory
        container.clear_large_data()
        gc.collect()
        cleanup_memory = tracker.get_memory_usage()
        
        # Memory should be reduced after cleanup (though may not return to initial)
        assert cleanup_memory <= large_memory
    
    def test_memory_growth_pattern(self):
        """Test memory growth patterns during operations."""
        tracker = MemoryTracker()
        memory_readings = []
        
        container = LearningDataContainer("growth_test")
        
        # Take memory readings at different points
        for i in range(5):
            tracker.reset()
            
            # Add data in batches
            for batch in range(10):
                for record in range(100):
                    container.add_learning_record(
                        f"concept_{record}", 
                        0.5 + (record % 10) * 0.05,
                        {"batch": batch, "iteration": i}
                    )
            
            memory_readings.append(tracker.get_memory_change())
        
        # Memory growth should be relatively consistent
        assert len(memory_readings) == 5
        assert all(reading >= 0 for reading in memory_readings)  # Should not decrease
    
    def test_memory_cleanup_effectiveness(self):
        """Test effectiveness of memory cleanup."""
        tracker = MemoryTracker()
        tracker.reset()
        
        containers = []
        
        # Create and populate containers
        for i in range(20):
            container = LearningDataContainer(f"cleanup_user_{i}")
            container.add_large_dataset(5000)
            containers.append(container)
        
        peak_memory = tracker.get_memory_usage()
        
        # Clear half the containers
        for i in range(10):
            containers[i].cleanup()
        
        partial_cleanup_memory = tracker.get_memory_usage()
        
        # Clear all containers
        for container in containers:
            container.cleanup()
        containers.clear()
        gc.collect()
        
        full_cleanup_memory = tracker.get_memory_usage()
        
        # Memory should decrease after cleanup
        assert partial_cleanup_memory <= peak_memory
        assert full_cleanup_memory <= partial_cleanup_memory


class TestMemoryLeakDetection:
    """Test memory leak detection using weak references."""
    
    def test_weak_reference_cleanup(self):
        """Test that objects are properly garbage collected."""
        references = []
        
        # Create objects and weak references
        for i in range(10):
            container = LearningDataContainer(f"weak_test_{i}")
            container.add_learning_record("test_concept", 0.8)
            
            # Create weak reference
            weak_ref = weakref.ref(container)
            references.append(weak_ref)
        
        # At this point, all containers should still exist
        alive_count = sum(1 for ref in references if ref() is not None)
        assert alive_count == 10
        
        # Force garbage collection
        gc.collect()
        
        # Objects should be garbage collected since we don't hold strong references
        alive_count_after_gc = sum(1 for ref in references if ref() is not None)
        assert alive_count_after_gc == 0  # All should be garbage collected
    
    def test_circular_reference_detection(self):
        """Test detection of circular references."""
        # Create objects with circular references
        container_a = LearningDataContainer("circular_a")
        container_b = LearningDataContainer("circular_b")
        
        # Create circular reference
        container_a.circular_ref = container_b
        container_b.circular_ref = container_a
        
        # Create weak references to detect cleanup
        weak_a = weakref.ref(container_a)
        weak_b = weakref.ref(container_b)
        
        # Remove strong references
        del container_a
        del container_b
        
        # Force garbage collection
        gc.collect()
        
        # Python's garbage collector should handle circular references
        assert weak_a() is None
        assert weak_b() is None
    
    def test_reference_counting(self):
        """Test reference counting behavior."""
        container = LearningDataContainer("ref_count_test")
        
        # Create multiple references
        ref1 = container
        ref2 = container
        ref_list = [container]
        
        weak_ref = weakref.ref(container)
        assert weak_ref() is not None
        
        # Remove references one by one
        del ref1
        assert weak_ref() is not None  # Still alive
        
        del ref2
        assert weak_ref() is not None  # Still alive (ref_list still holds it)
        
        ref_list.clear()
        del container  # Remove original reference
        gc.collect()
        
        # Now should be garbage collected
        assert weak_ref() is None


class TestConcurrentMemoryUsage:
    """Test memory usage in concurrent scenarios."""
    
    def test_concurrent_container_creation(self):
        """Test memory usage during concurrent container creation."""
        tracker = MemoryTracker()
        tracker.reset()
        
        def create_container(user_id):
            container = LearningDataContainer(f"concurrent_{user_id}")
            for i in range(100):
                container.add_learning_record(f"concept_{i}", 0.7)
            return container
        
        # Create containers concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_container, i) for i in range(20)]
            containers = [future.result() for future in as_completed(futures)]
        
        concurrent_memory = tracker.get_memory_change()
        
        # Cleanup
        containers.clear()
        gc.collect()
        
        cleanup_memory = tracker.get_memory_change()
        
        assert len(containers) == 20
        assert concurrent_memory > 0
        assert cleanup_memory <= concurrent_memory
    
    def test_concurrent_data_access(self):
        """Test memory usage during concurrent data access."""
        container = LearningDataContainer("shared_container")
        
        # Pre-populate with data
        for i in range(1000):
            container.add_learning_record(f"concept_{i}", 0.8)
        
        tracker = MemoryTracker()
        tracker.reset()
        
        def access_data(thread_id):
            results = []
            for i in range(100):
                concept = f"concept_{i % 100}"
                if concept in container.data:
                    results.extend(container.data[concept])
            return len(results)
        
        # Access data concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(access_data, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        access_memory = tracker.get_memory_change()
        
        # Memory shouldn't grow significantly just from reading
        assert access_memory < 5.0  # Less than 5MB growth
        assert all(result > 0 for result in results)
    
    def test_memory_under_load(self):
        """Test memory behavior under sustained load."""
        tracker = MemoryTracker()
        memory_samples = []
        
        def sustained_operations():
            containers = []
            for i in range(50):
                container = LearningDataContainer(f"load_test_{i}")
                container.add_large_dataset(1000)
                containers.append(container)
                
                # Sample memory every 10 iterations
                if i % 10 == 0:
                    memory_samples.append(tracker.get_memory_usage())
            
            return containers
        
        # Run sustained operations
        containers = sustained_operations()
        
        # Memory should show growth pattern
        assert len(memory_samples) >= 5
        
        # Generally increasing trend (though may have fluctuations)
        trend_increasing = memory_samples[-1] >= memory_samples[0]
        assert trend_increasing
        
        # Cleanup
        for container in containers:
            container.cleanup()
        containers.clear()
        gc.collect()


class TestMemoryOptimization:
    """Test memory optimization strategies."""
    
    def test_data_structure_efficiency(self):
        """Test efficiency of different data structures."""
        tracker = MemoryTracker()
        
        # Test list vs dict storage
        tracker.reset()
        list_container = []
        for i in range(10000):
            list_container.append({"id": i, "score": 0.8, "concept": f"concept_{i % 100}"})
        list_memory = tracker.get_memory_change()
        
        tracker.reset()
        dict_container = {}
        for i in range(10000):
            concept = f"concept_{i % 100}"
            if concept not in dict_container:
                dict_container[concept] = []
            dict_container[concept].append({"id": i, "score": 0.8})
        dict_memory = tracker.get_memory_change()
        
        # Both should use reasonable memory
        assert list_memory > 0
        assert dict_memory > 0
        
        # The difference shouldn't be extreme (within 50% of each other)
        ratio = max(list_memory, dict_memory) / min(list_memory, dict_memory)
        assert ratio < 1.5
    
    def test_memory_reuse_patterns(self):
        """Test memory reuse patterns."""
        tracker = MemoryTracker()
        tracker.reset()
        
        container = LearningDataContainer("reuse_test")
        
        # Fill and clear repeatedly
        for cycle in range(5):
            # Add data
            for i in range(1000):
                container.add_learning_record(f"concept_{i}", 0.8)
            
            mid_cycle_memory = tracker.get_memory_usage()
            
            # Clear data
            container.cleanup()
            gc.collect()
            
            end_cycle_memory = tracker.get_memory_usage()
            
            # Memory should decrease after cleanup
            assert end_cycle_memory <= mid_cycle_memory
        
        final_memory = tracker.get_memory_change()
        
        # Final memory shouldn't be excessive despite multiple cycles
        assert final_memory < 20.0  # Less than 20MB growth
    
    def test_lazy_loading_simulation(self):
        """Test simulated lazy loading for memory efficiency."""
        tracker = MemoryTracker()
        tracker.reset()
        
        class LazyDataContainer:
            def __init__(self):
                self._data = None
                self._loaded = False
            
            def get_data(self):
                if not self._loaded:
                    # Simulate loading large dataset
                    self._data = [f"item_{i}" for i in range(10000)]
                    self._loaded = True
                return self._data
            
            def clear(self):
                self._data = None
                self._loaded = False
        
        containers = [LazyDataContainer() for _ in range(10)]
        
        # Memory should be low initially
        initial_memory = tracker.get_memory_change()
        assert initial_memory < 1.0
        
        # Access data from some containers
        for i in range(5):
            data = containers[i].get_data()
            assert len(data) == 10000
        
        loaded_memory = tracker.get_memory_change()
        assert loaded_memory > initial_memory
        
        # Clear loaded data
        for container in containers:
            container.clear()
        gc.collect()
        
        cleared_memory = tracker.get_memory_change()
        assert cleared_memory <= loaded_memory
    
    def test_memory_pooling_simulation(self):
        """Test simulated memory pooling for efficiency."""
        tracker = MemoryTracker()
        
        class ObjectPool:
            def __init__(self, size=10):
                self.pool = [LearningDataContainer(f"pooled_{i}") for i in range(size)]
                self.available = list(self.pool)
                self.in_use = []
            
            def acquire(self):
                if self.available:
                    obj = self.available.pop()
                    self.in_use.append(obj)
                    return obj
                return None
            
            def release(self, obj):
                if obj in self.in_use:
                    self.in_use.remove(obj)
                    obj.cleanup()  # Clean the object
                    self.available.append(obj)
        
        tracker.reset()
        pool = ObjectPool(20)
        
        # Use objects from pool repeatedly
        for cycle in range(10):
            # Acquire objects
            objects = []
            for _ in range(15):
                obj = pool.acquire()
                if obj:
                    obj.add_learning_record("test_concept", 0.8)
                    objects.append(obj)
            
            # Release objects back to pool
            for obj in objects:
                pool.release(obj)
        
        pooled_memory = tracker.get_memory_change()
        
        # Memory growth should be limited due to reuse
        assert pooled_memory < 10.0  # Less than 10MB growth


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 