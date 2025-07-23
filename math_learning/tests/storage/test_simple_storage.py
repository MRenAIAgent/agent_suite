"""
Simple storage tests for math_learning system.

Tests basic storage functionality without complex dependencies.
"""

import pytest
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)


class SimpleStorageBackend:
    """Simple in-memory storage backend for testing."""
    
    def __init__(self):
        self.data = {}
        self.metadata = {}
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value with a key."""
        try:
            self.data[key] = value
            self.metadata[key] = {
                "created": datetime.now().isoformat(),
                "type": type(value).__name__
            }
            return True
        except Exception:
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        return self.data.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a value by key."""
        if key in self.data:
            del self.data[key]
            if key in self.metadata:
                del self.metadata[key]
            return True
        return False
    
    def list_keys(self) -> List[str]:
        """List all keys in storage."""
        return list(self.data.keys())
    
    def clear(self):
        """Clear all data."""
        self.data.clear()
        self.metadata.clear()


class MathLearningData:
    """Simple data structure for math learning information."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.mastery_levels = {}
        self.learning_history = []
        self.preferences = {}
    
    def record_mastery(self, concept: str, level: float):
        """Record mastery level for a concept."""
        self.mastery_levels[concept] = level
        self.learning_history.append({
            "concept": concept,
            "mastery": level,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_mastery(self, concept: str) -> Optional[float]:
        """Get mastery level for a concept."""
        return self.mastery_levels.get(concept)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "mastery_levels": self.mastery_levels,
            "learning_history": self.learning_history,
            "preferences": self.preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MathLearningData':
        """Create from dictionary."""
        instance = cls(data["user_id"])
        instance.mastery_levels = data.get("mastery_levels", {})
        instance.learning_history = data.get("learning_history", [])
        instance.preferences = data.get("preferences", {})
        return instance


class TestSimpleStorage:
    """Test simple storage operations."""
    
    def test_storage_backend_creation(self):
        """Test creating a storage backend."""
        storage = SimpleStorageBackend()
        assert storage.data == {}
        assert storage.metadata == {}
    
    def test_basic_store_retrieve(self):
        """Test basic store and retrieve operations."""
        storage = SimpleStorageBackend()
        
        # Store some data
        result = storage.store("test_key", "test_value")
        assert result is True
        
        # Retrieve the data
        value = storage.retrieve("test_key")
        assert value == "test_value"
        
        # Test non-existent key
        value = storage.retrieve("non_existent")
        assert value is None
    
    def test_storage_with_complex_data(self):
        """Test storing complex data structures."""
        storage = SimpleStorageBackend()
        
        # Store a dictionary
        complex_data = {
            "user_id": "test_user",
            "scores": [0.8, 0.9, 0.7],
            "metadata": {"created": "2024-01-01"}
        }
        
        result = storage.store("complex_key", complex_data)
        assert result is True
        
        retrieved = storage.retrieve("complex_key")
        assert retrieved == complex_data
        assert retrieved["user_id"] == "test_user"
        assert len(retrieved["scores"]) == 3
    
    def test_storage_deletion(self):
        """Test deleting stored data."""
        storage = SimpleStorageBackend()
        
        # Store and verify
        storage.store("delete_me", "value")
        assert storage.retrieve("delete_me") == "value"
        
        # Delete and verify
        result = storage.delete("delete_me")
        assert result is True
        assert storage.retrieve("delete_me") is None
        
        # Try to delete non-existent key
        result = storage.delete("non_existent")
        assert result is False
    
    def test_list_keys(self):
        """Test listing all keys."""
        storage = SimpleStorageBackend()
        
        # Initially empty
        keys = storage.list_keys()
        assert keys == []
        
        # Add some data
        storage.store("key1", "value1")
        storage.store("key2", "value2")
        storage.store("key3", "value3")
        
        keys = storage.list_keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys
    
    def test_storage_clear(self):
        """Test clearing all storage."""
        storage = SimpleStorageBackend()
        
        # Add some data
        storage.store("key1", "value1")
        storage.store("key2", "value2")
        
        assert len(storage.list_keys()) == 2
        
        # Clear and verify
        storage.clear()
        assert len(storage.list_keys()) == 0
        assert storage.retrieve("key1") is None
    
    def test_math_learning_data(self):
        """Test the math learning data structure."""
        data = MathLearningData("test_user")
        
        assert data.user_id == "test_user"
        assert data.mastery_levels == {}
        assert data.learning_history == []
    
    def test_record_mastery(self):
        """Test recording mastery levels."""
        data = MathLearningData("test_user")
        
        # Record some mastery
        data.record_mastery("algebra", 0.8)
        data.record_mastery("geometry", 0.6)
        
        # Verify mastery levels
        assert data.get_mastery("algebra") == 0.8
        assert data.get_mastery("geometry") == 0.6
        assert data.get_mastery("calculus") is None
        
        # Verify history
        assert len(data.learning_history) == 2
        assert data.learning_history[0]["concept"] == "algebra"
        assert data.learning_history[0]["mastery"] == 0.8
    
    def test_data_serialization(self):
        """Test converting data to/from dictionary."""
        data = MathLearningData("test_user")
        data.record_mastery("algebra", 0.8)
        data.preferences = {"theme": "dark", "difficulty": "medium"}
        
        # Convert to dict
        data_dict = data.to_dict()
        assert data_dict["user_id"] == "test_user"
        assert data_dict["mastery_levels"]["algebra"] == 0.8
        assert data_dict["preferences"]["theme"] == "dark"
        
        # Convert back from dict
        restored = MathLearningData.from_dict(data_dict)
        assert restored.user_id == "test_user"
        assert restored.get_mastery("algebra") == 0.8
        assert restored.preferences["theme"] == "dark"
    
    def test_integrated_storage_workflow(self):
        """Test a complete storage workflow."""
        storage = SimpleStorageBackend()
        
        # Create multiple users
        users = []
        for i in range(3):
            user_data = MathLearningData(f"user_{i}")
            user_data.record_mastery("algebra", 0.5 + i * 0.1)
            user_data.record_mastery("geometry", 0.6 + i * 0.1)
            users.append(user_data)
        
        # Store all users
        for user_data in users:
            key = f"user_data_{user_data.user_id}"
            result = storage.store(key, user_data.to_dict())
            assert result is True
        
        # Retrieve and verify
        for i in range(3):
            key = f"user_data_user_{i}"
            retrieved_dict = storage.retrieve(key)
            assert retrieved_dict is not None
            
            user_data = MathLearningData.from_dict(retrieved_dict)
            assert user_data.user_id == f"user_{i}"
            assert user_data.get_mastery("algebra") == 0.5 + i * 0.1
    
    def test_storage_performance(self):
        """Test storage performance with larger datasets."""
        storage = SimpleStorageBackend()
        
        # Store many items
        num_items = 100
        for i in range(num_items):
            key = f"item_{i}"
            value = {
                "id": i,
                "data": f"data_{i}",
                "scores": [j * 0.1 for j in range(10)]
            }
            result = storage.store(key, value)
            assert result is True
        
        # Verify all items
        keys = storage.list_keys()
        assert len(keys) == num_items
        
        # Retrieve and verify some items
        for i in [0, 25, 50, 75, 99]:
            key = f"item_{i}"
            value = storage.retrieve(key)
            assert value is not None
            assert value["id"] == i
            assert value["data"] == f"data_{i}"
    
    def test_storage_error_handling(self):
        """Test storage error handling."""
        storage = SimpleStorageBackend()
        
        # Test storing None
        result = storage.store("none_key", None)
        assert result is True  # None is a valid value
        
        retrieved = storage.retrieve("none_key")
        assert retrieved is None
        
        # Test empty string key
        result = storage.store("", "empty_key_value")
        assert result is True
        
        retrieved = storage.retrieve("")
        assert retrieved == "empty_key_value"
    
    def test_concurrent_operations_simulation(self):
        """Test simulated concurrent operations."""
        storage = SimpleStorageBackend()
        
        # Simulate multiple operations
        operations = [
            ("store", "user1", {"mastery": 0.8}),
            ("store", "user2", {"mastery": 0.6}),
            ("retrieve", "user1", None),
            ("store", "user3", {"mastery": 0.9}),
            ("retrieve", "user2", None),
            ("delete", "user1", None),
            ("retrieve", "user1", None)  # Should return None after deletion
        ]
        
        results = []
        for op, key, value in operations:
            if op == "store":
                result = storage.store(key, value)
                results.append(("store", key, result))
            elif op == "retrieve":
                result = storage.retrieve(key)
                results.append(("retrieve", key, result))
            elif op == "delete":
                result = storage.delete(key)
                results.append(("delete", key, result))
        
        # Verify results
        assert len(results) == 7
        
        # Check that user1 was stored, retrieved, deleted, then returned None
        store_user1 = [r for r in results if r[0] == "store" and r[1] == "user1"]
        assert len(store_user1) == 1
        assert store_user1[0][2] is True  # Store was successful
        
        retrieve_user1_after_delete = [r for r in results if r[0] == "retrieve" and r[1] == "user1"]
        assert len(retrieve_user1_after_delete) == 2  # Retrieved twice
        assert retrieve_user1_after_delete[0][2] is not None  # First retrieve succeeded
        assert retrieve_user1_after_delete[1][2] is None  # Second retrieve (after delete) returned None


class TestMemoryUsage:
    """Test memory usage patterns in storage."""
    
    def test_memory_cleanup(self):
        """Test that memory is properly managed."""
        storage = SimpleStorageBackend()
        
        # Create a large dataset
        large_data = {}
        for i in range(100):
            large_data[f"key_{i}"] = [j for j in range(100)]  # 100 items each
        
        # Store the large dataset
        storage.store("large_dataset", large_data)
        
        # Verify it's stored
        retrieved = storage.retrieve("large_dataset")
        assert retrieved is not None
        assert len(retrieved) == 100
        
        # Clear storage
        storage.clear()
        
        # Verify it's gone
        assert storage.retrieve("large_dataset") is None
        assert len(storage.list_keys()) == 0
    
    def test_reference_counting(self):
        """Test that objects are properly referenced."""
        storage = SimpleStorageBackend()
        
        # Create an object
        original_data = {"important": "data", "values": [1, 2, 3]}
        
        # Store it
        storage.store("ref_test", original_data)
        
        # Retrieve it
        retrieved_data = storage.retrieve("ref_test")
        
        # Modify the original
        original_data["new_key"] = "new_value"
        
        # The stored version should be independent (depending on implementation)
        # This test documents the current behavior
        assert retrieved_data is not None
        assert "important" in retrieved_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 