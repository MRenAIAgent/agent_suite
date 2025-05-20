"""
Unit tests for the EmbeddingService class.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from agent_suite.vector.embedding_service import EmbeddingService


class TestEmbeddingService(unittest.TestCase):
    """Tests for the EmbeddingService class."""

    @patch.dict(os.environ, {"COHERE_API_KEY": "test_api_key"})
    @patch("cohere.Client")
    def setUp(self, mock_client):
        """Set up test fixtures."""
        self.mock_client = mock_client
        self.embedding_service = EmbeddingService(model_name="embed-english-v3.0")
        # Reset call counts for mock client
        self.mock_client.reset_mock()

    def test_initialization(self):
        """Test that the embedding service initializes properly."""
        self.assertEqual(self.embedding_service.model_name, "embed-english-v3.0")
        self.assertEqual(self.embedding_service.input_type, "search_document")
        self.assertEqual(self.embedding_service.api_key, "test_api_key")
        self.assertEqual(self.embedding_service.get_vector_size(), 1024)

    @patch.dict(os.environ, {"COHERE_API_KEY": "test_api_key"})
    @patch("cohere.Client")
    def test_initialization_with_custom_model(self, mock_client):
        """Test initialization with a custom model."""
        embedding_service = EmbeddingService(
            model_name="embed-english-light-v3.0",
            input_type="classification"
        )
        self.assertEqual(embedding_service.model_name, "embed-english-light-v3.0")
        self.assertEqual(embedding_service.input_type, "classification")
        self.assertEqual(embedding_service.get_vector_size(), 384)

    @patch.dict(os.environ, {})
    def test_initialization_without_api_key(self):
        """Test initialization fails without an API key."""
        with self.assertRaises(ValueError):
            EmbeddingService()

    @patch.dict(os.environ, {})
    @patch("cohere.Client")
    def test_initialization_with_param_api_key(self, mock_client):
        """Test initialization with API key as a parameter."""
        embedding_service = EmbeddingService(api_key="param_api_key")
        self.assertEqual(embedding_service.api_key, "param_api_key")

    def test_get_vector_size(self):
        """Test getting vector size for different models."""
        sizes = {
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384,
            "unknown-model": 1024  # Default size
        }
        
        for model, size in sizes.items():
            with patch.dict(os.environ, {"COHERE_API_KEY": "test_api_key"}):
                with patch("cohere.Client"):
                    embedding_service = EmbeddingService(model_name=model)
                    self.assertEqual(embedding_service.get_vector_size(), size)

    def test_embed_single(self):
        """Test embedding a single text."""
        # Mock the embed response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        self.mock_client.return_value.embed.return_value = mock_response
        
        # Call the method
        embedding = self.embedding_service._embed_single("test text")
        
        # Verify the result
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        self.mock_client.return_value.embed.assert_called_once_with(
            texts=["test text"],
            model="embed-english-v3.0",
            input_type="search_document"
        )

    def test_embed_single_error(self):
        """Test error handling when embedding a single text."""
        # Make the embed method raise an exception
        self.mock_client.return_value.embed.side_effect = Exception("API error")
        
        # Call the method
        embedding = self.embedding_service._embed_single("test text")
        
        # Verify the result is a zero vector
        self.assertEqual(len(embedding), 1024)
        self.assertEqual(sum(embedding), 0.0)

    def test_embed_single_with_caching(self):
        """Test that the embedding caching works."""
        # Mock the embed response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        self.mock_client.return_value.embed.return_value = mock_response
        
        # Call the method twice with the same input
        embedding1 = self.embedding_service._embed_single("test text")
        embedding2 = self.embedding_service._embed_single("test text")
        
        # Verify the method was only called once
        self.mock_client.return_value.embed.assert_called_once()
        self.assertEqual(embedding1, embedding2)

    def test_embed_list(self):
        """Test embedding a list of texts."""
        # Mock the embed response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.mock_client.return_value.embed.return_value = mock_response
        
        # Call the method
        embeddings = self.embedding_service.embed(["text1", "text2"])
        
        # Verify the result
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.mock_client.return_value.embed.assert_called_once_with(
            texts=["text1", "text2"],
            model="embed-english-v3.0",
            input_type="search_document"
        )

    def test_embed_string(self):
        """Test embedding a single string."""
        # Mock the embed response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        self.mock_client.return_value.embed.return_value = mock_response
        
        # Call the method
        embeddings = self.embedding_service.embed("single text")
        
        # Verify the result
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3]])

    def test_embed_empty_list(self):
        """Test embedding an empty list."""
        embeddings = self.embedding_service.embed([])
        self.assertEqual(embeddings, [])
        self.mock_client.return_value.embed.assert_not_called()

    def test_embed_query(self):
        """Test embedding a query with the appropriate input type."""
        # Mock the embed response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        self.mock_client.return_value.embed.return_value = mock_response
        
        # Call the method
        embedding = self.embedding_service.embed_query("search query")
        
        # Verify the result
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        self.mock_client.return_value.embed.assert_called_once_with(
            texts=["search query"],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        
        # Verify the input type was restored
        self.assertEqual(self.embedding_service.input_type, "search_document")

    def test_embed_documents(self):
        """Test embedding documents with the appropriate input type."""
        # Mock the embed response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.mock_client.return_value.embed.return_value = mock_response
        
        # Set a different input type to check restoration
        self.embedding_service.input_type = "classification"
        
        # Call the method
        embeddings = self.embedding_service.embed_documents(["doc1", "doc2"])
        
        # Verify the result
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.mock_client.return_value.embed.assert_called_once_with(
            texts=["doc1", "doc2"],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        
        # Verify the input type was restored
        self.assertEqual(self.embedding_service.input_type, "classification")


if __name__ == "__main__":
    unittest.main() 