"""
Unit tests for the QdrantManager class.
"""

import os
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
import shutil
from pathlib import Path

import pytest
from qdrant_client.http import models

from agent_suite.vector.qdrant_client import QdrantManager


class TestQdrantManager(unittest.TestCase):
    """Tests for the QdrantManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a patch for the QdrantClient class
        self.client_patcher = patch("agent_suite.vector.qdrant_client.QdrantClient")
        self.mock_client_class = self.client_patcher.start()
        self.mock_client = self.mock_client_class.return_value
        
        # Mock collections response
        collections_response = MagicMock()
        collections_response.collections = []
        self.mock_client.get_collections.return_value = collections_response
        
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patcher.stop()
        shutil.rmtree(self.temp_dir)

    @patch.dict(os.environ, {"QDRANT_API_KEY": "test_api_key"})
    def test_initialize_with_env_api_key(self):
        """Test initialization with API key from environment."""
        manager = QdrantManager()
        self.assertEqual(manager.api_key, "test_api_key")
        self.mock_client_class.assert_called_once_with(":memory:")
        self.assertFalse(manager.is_remote)
    
    def test_initialize_with_param_api_key(self):
        """Test initialization with API key as parameter."""
        manager = QdrantManager(api_key="param_api_key")
        self.assertEqual(manager.api_key, "param_api_key")
    
    def test_initialize_with_url(self):
        """Test initialization with a URL."""
        manager = QdrantManager(connection_url="http://qdrant:6333", api_key="test_key")
        self.mock_client_class.assert_called_once_with(
            url="http://qdrant:6333",
            api_key="test_key"
        )
        self.assertTrue(manager.is_remote)
    
    def test_initialize_with_persist_directory(self):
        """Test initialization with a persistence directory."""
        manager = QdrantManager(persist_directory=self.temp_dir)
        self.mock_client_class.assert_called_once_with(path=str(self.temp_dir))
        self.assertFalse(manager.is_remote)
    
    def test_collection_exists_true(self):
        """Test checking if a collection exists when it does."""
        # Setup the mock response
        collections_response = MagicMock()
        collection = MagicMock()
        collection.name = "test_collection"
        collections_response.collections = [collection]
        self.mock_client.get_collections.return_value = collections_response
        
        manager = QdrantManager()
        exists = manager.collection_exists("test_collection")
        
        self.assertTrue(exists)
        self.assertIn("test_collection", manager._collection_cache)
    
    def test_collection_exists_false(self):
        """Test checking if a collection exists when it doesn't."""
        manager = QdrantManager()
        exists = manager.collection_exists("nonexistent_collection")
        
        self.assertFalse(exists)
        self.assertNotIn("nonexistent_collection", manager._collection_cache)
    
    def test_collection_exists_from_cache(self):
        """Test checking if a collection exists from the cache."""
        manager = QdrantManager()
        manager._collection_cache.add("cached_collection")
        
        exists = manager.collection_exists("cached_collection")
        
        self.assertTrue(exists)
        # Verify that get_collections wasn't called
        self.mock_client.get_collections.assert_not_called()
    
    def test_create_collection_new(self):
        """Test creating a new collection."""
        manager = QdrantManager()
        success = manager.create_collection("new_collection", vector_size=128)
        
        self.assertTrue(success)
        self.assertIn("new_collection", manager._collection_cache)
        self.mock_client.create_collection.assert_called_once_with(
            collection_name="new_collection",
            vectors_config=models.VectorParams(size=128, distance="cosine")
        )
    
    def test_create_collection_existing_no_recreate(self):
        """Test creating a collection that already exists without recreating."""
        # Setup the mock response
        collections_response = MagicMock()
        collection = MagicMock()
        collection.name = "existing_collection"
        collections_response.collections = [collection]
        self.mock_client.get_collections.return_value = collections_response
        
        manager = QdrantManager()
        success = manager.create_collection("existing_collection", vector_size=128)
        
        self.assertTrue(success)
        self.assertIn("existing_collection", manager._collection_cache)
        self.mock_client.delete_collection.assert_not_called()
        self.mock_client.create_collection.assert_not_called()
    
    def test_create_collection_existing_with_recreate(self):
        """Test creating a collection that already exists with recreating."""
        # Setup the mock response
        collections_response = MagicMock()
        collection = MagicMock()
        collection.name = "existing_collection"
        collections_response.collections = [collection]
        self.mock_client.get_collections.return_value = collections_response
        
        manager = QdrantManager()
        success = manager.create_collection(
            "existing_collection", 
            vector_size=128, 
            recreate=True
        )
        
        self.assertTrue(success)
        self.assertIn("existing_collection", manager._collection_cache)
        self.mock_client.delete_collection.assert_called_once_with("existing_collection")
        self.mock_client.create_collection.assert_called_once()
    
    def test_add_points(self):
        """Test adding points to a collection."""
        # Setup collection exists
        collections_response = MagicMock()
        collection = MagicMock()
        collection.name = "test_collection"
        collections_response.collections = [collection]
        self.mock_client.get_collections.return_value = collections_response
        
        manager = QdrantManager()
        
        # Test data
        ids = ["id1", "id2"]
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        payloads = [{"meta": "data1"}, {"meta": "data2"}]
        
        success = manager.add_points(
            "test_collection",
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )
        
        self.assertTrue(success)
        self.mock_client.upsert.assert_called_once()
        
        # Check the points were formatted correctly
        call_args = self.mock_client.upsert.call_args
        self.assertEqual(call_args[1]["collection_name"], "test_collection")
        points = call_args[1]["points"]
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].id, "id1")
        self.assertEqual(points[0].vector, [0.1, 0.2])
        self.assertEqual(points[0].payload, {"meta": "data1"})
        self.assertEqual(points[1].id, "id2")
        self.assertEqual(points[1].vector, [0.3, 0.4])
        self.assertEqual(points[1].payload, {"meta": "data2"})
    
    def test_add_points_collection_not_exists(self):
        """Test adding points to a non-existent collection."""
        manager = QdrantManager()
        
        # Test data
        ids = ["id1"]
        vectors = [[0.1, 0.2]]
        
        # Method should raise ValueError
        with self.assertRaises(ValueError):
            manager.add_points(
                "nonexistent_collection",
                ids=ids,
                vectors=vectors
            )
    
    def test_add_points_default_payload(self):
        """Test adding points without specifying payloads."""
        # Setup collection exists
        collections_response = MagicMock()
        collection = MagicMock()
        collection.name = "test_collection"
        collections_response.collections = [collection]
        self.mock_client.get_collections.return_value = collections_response
        
        manager = QdrantManager()
        
        # Test data
        ids = ["id1", "id2"]
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        
        success = manager.add_points(
            "test_collection",
            ids=ids,
            vectors=vectors
        )
        
        self.assertTrue(success)
        self.mock_client.upsert.assert_called_once()
        
        # Check empty payloads were added
        call_args = self.mock_client.upsert.call_args
        points = call_args[1]["points"]
        self.assertEqual(points[0].payload, {})
        self.assertEqual(points[1].payload, {})
    
    def test_search(self):
        """Test searching for vectors."""
        # Mock search response
        point1 = MagicMock()
        point1.id = "result1"
        point1.score = 0.95
        point1.payload = {"key": "value1"}
        
        point2 = MagicMock()
        point2.id = "result2"
        point2.score = 0.85
        point2.payload = {"key": "value2"}
        
        self.mock_client.search.return_value = [point1, point2]
        
        manager = QdrantManager()
        results = manager.search(
            "test_collection",
            query_vector=[0.1, 0.2],
            limit=5
        )
        
        # Check the results were formatted correctly
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "result1")
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[0]["payload"], {"key": "value1"})
        self.assertEqual(results[1]["id"], "result2")
        self.assertEqual(results[1]["score"], 0.85)
        self.assertEqual(results[1]["payload"], {"key": "value2"})
        
        # Check the search was called correctly
        self.mock_client.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=[0.1, 0.2],
            limit=5,
            query_filter=None
        )
    
    def test_search_with_filter(self):
        """Test searching with a filter condition."""
        self.mock_client.search.return_value = []
        
        manager = QdrantManager()
        filter_condition = {
            "must": [
                {"key": "type", "match": {"value": "document"}}
            ]
        }
        
        results = manager.search(
            "test_collection",
            query_vector=[0.1, 0.2],
            filter_condition=filter_condition
        )
        
        # Ensure filter was passed correctly
        call_args = self.mock_client.search.call_args
        self.assertIsNotNone(call_args[1]["query_filter"])
        # We don't check the exact format as it depends on qdrant-client internals
    
    def test_delete_points(self):
        """Test deleting points from a collection."""
        # Setup collection exists
        collections_response = MagicMock()
        collection = MagicMock()
        collection.name = "test_collection"
        collections_response.collections = [collection]
        self.mock_client.get_collections.return_value = collections_response
        
        manager = QdrantManager()
        success = manager.delete_points(
            "test_collection",
            point_ids=["id1", "id2"]
        )
        
        self.assertTrue(success)
        self.mock_client.delete.assert_called_once()
        
        # Check the points selector was formatted correctly
        call_args = self.mock_client.delete.call_args
        self.assertEqual(call_args[1]["collection_name"], "test_collection")
        self.assertEqual(call_args[1]["points_selector"].points, ["id1", "id2"])
    
    def test_get_collection_info(self):
        """Test getting collection information."""
        # Setup collection exists
        collections_response = MagicMock()
        collection = MagicMock()
        collection.name = "test_collection"
        collections_response.collections = [collection]
        self.mock_client.get_collections.return_value = collections_response
        
        # Mock collection info
        collection_info = MagicMock()
        collection_info.name = "test_collection"
        collection_info.vectors_count = 100
        collection_info.points_count = 50
        collection_info.status = "green"
        self.mock_client.get_collection.return_value = collection_info
        
        manager = QdrantManager()
        info = manager.get_collection_info("test_collection")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "test_collection")
        self.assertEqual(info["vectors_count"], 100)
        self.assertEqual(info["points_count"], 50)
        self.assertEqual(info["status"], "green")
        
        self.mock_client.get_collection.assert_called_once_with("test_collection")
    
    def test_get_collection_info_nonexistent(self):
        """Test getting information for a non-existent collection."""
        manager = QdrantManager()
        info = manager.get_collection_info("nonexistent_collection")
        
        self.assertIsNone(info)
        self.mock_client.get_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main() 