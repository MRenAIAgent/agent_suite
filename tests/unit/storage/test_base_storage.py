"""Unit tests for the BaseStorage interface."""

import pytest
from unittest.mock import AsyncMock, patch

from agents.storage.base_storage import BaseStorage


@pytest.mark.asyncio
async def test_base_storage_methods(mock_base_storage):
    """Test that the base storage methods work as expected."""
    # Test store method
    key = "test-key"
    data = {"value": "test-value"}
    result = await mock_base_storage.store(key, data)
    assert result == "mock-id"
    mock_base_storage.store_mock.assert_called_once_with(key, data)
    
    # Test retrieve method
    result = await mock_base_storage.retrieve(key)
    assert result == {"test": "data"}
    mock_base_storage.retrieve_mock.assert_called_once_with(key)
    
    # Test delete method
    result = await mock_base_storage.delete(key)
    assert result is True
    mock_base_storage.delete_mock.assert_called_once_with(key)
    
    # Test update method
    new_data = {"updated": "value"}
    result = await mock_base_storage.update(key, new_data)
    assert result is True
    mock_base_storage.update_mock.assert_called_once_with(key, new_data)
    
    # Test list method
    pattern = "test-*"
    result = await mock_base_storage.list(pattern)
    assert result == ["id1", "id2"]
    mock_base_storage.list_mock.assert_called_once_with(pattern)
    
    # Test clear method
    result = await mock_base_storage.clear()
    assert result is True
    mock_base_storage.clear_mock.assert_called_once()


@pytest.mark.asyncio
async def test_base_storage_with_kwargs(mock_base_storage):
    """Test that the base storage methods properly pass kwargs."""
    # Test store with kwargs
    await mock_base_storage.store("key", {"data": "value"}, extra_param="value")
    mock_base_storage.store_mock.assert_called_with("key", {"data": "value"}, extra_param="value")
    
    # Test retrieve with kwargs
    await mock_base_storage.retrieve("key", include_metadata=True)
    mock_base_storage.retrieve_mock.assert_called_with("key", include_metadata=True)
    
    # Test delete with kwargs
    await mock_base_storage.delete("key", force=True)
    mock_base_storage.delete_mock.assert_called_with("key", force=True)
    
    # Test update with kwargs
    await mock_base_storage.update("key", {"data": "new"}, upsert=True)
    mock_base_storage.update_mock.assert_called_with("key", {"data": "new"}, upsert=True)
    
    # Test list with kwargs
    await mock_base_storage.list("*", limit=10)
    mock_base_storage.list_mock.assert_called_with("*", limit=10)
    
    # Test clear with kwargs
    await mock_base_storage.clear(type="all")
    mock_base_storage.clear_mock.assert_called_with(type="all") 