"""Memory-based key-value storage adaptor for the RAG system.

This module provides an in-memory implementation of the key-value storage adaptor
interface, suitable for testing and development.
"""

import uuid
from typing import Dict, Any, List, Optional, Union

from agents.rag.storage.key_value.base import KeyValueStorageAdaptor
from agents.rag.models.context import ContextItem


class MemoryKeyValueStorage(KeyValueStorageAdaptor):
    """In-memory implementation of key-value storage adaptor."""
    
    def __init__(self):
        """Initialize the memory key-value storage."""
        self.data: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.context_items: Dict[str, ContextItem] = {}
    
    async def set(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> str:
        """
        Set a key-value pair.
        
        Args:
            key: The key
            value: The value
            metadata: Optional metadata
            
        Returns:
            The ID of the stored item
        """
        self.data[key] = value
        if metadata:
            self.metadata[key] = metadata
        return key
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value by key.
        
        Args:
            key: The key
            
        Returns:
            The value or None if not found
        """
        return self.data.get(key)
    
    async def get_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """
        Get all key-value pairs with keys starting with the given prefix.
        
        Args:
            prefix: The key prefix
            
        Returns:
            Dictionary of matching key-value pairs
        """
        result = {}
        for key, value in self.data.items():
            if key.startswith(prefix):
                result[key] = value
        return result
    
    async def store_context_item(self, item: ContextItem) -> str:
        """
        Store a context item.
        
        Args:
            item: The context item to store
            
        Returns:
            The ID of the stored item
        """
        item_id = item.id or str(uuid.uuid4())
        item.id = item_id
        self.context_items[item_id] = item
        return item_id
    
    async def get_all_context(self) -> List[ContextItem]:
        """
        Get all stored context items.
        
        Returns:
            List of all context items
        """
        return list(self.context_items.values())
    
    # Base StorageAdaptor interface methods
    
    async def store(self, data: Any, metadata: Dict[str, Any] = None) -> str:
        """
        Store data in the backend.
        
        Args:
            data: The data to store
            metadata: Optional metadata
            
        Returns:
            The ID of the stored data
        """
        if isinstance(data, ContextItem):
            return await self.store_context_item(data)
        else:
            # Generate a unique key for the data
            key = str(uuid.uuid4())
            return await self.set(key, data, metadata)
    
    async def retrieve(self, id: str) -> Optional[Any]:
        """
        Retrieve data by ID.
        
        Args:
            id: The ID of the data to retrieve
            
        Returns:
            The retrieved data or None if not found
        """
        # Check context items first
        if id in self.context_items:
            return self.context_items[id]
        
        # Check regular key-value data
        return await self.get(id)
    
    async def update(self, id: str, data: Any) -> bool:
        """
        Update existing data.
        
        Args:
            id: The ID of the data to update
            data: The new data
            
        Returns:
            Success status
        """
        if id in self.data or id in self.context_items:
            if isinstance(data, ContextItem):
                data.id = id
                self.context_items[id] = data
            else:
                self.data[id] = data
            return True
        return False
    
    async def delete(self, id: str) -> bool:
        """
        Delete data by ID.
        
        Args:
            id: The ID of the data to delete
            
        Returns:
            Success status
        """
        deleted = False
        
        if id in self.data:
            del self.data[id]
            if id in self.metadata:
                del self.metadata[id]
            deleted = True
        
        if id in self.context_items:
            del self.context_items[id]
            deleted = True
        
        return deleted
    
    async def search(self, query: Any, **kwargs) -> List[Any]:
        """
        Search for data matching the query.
        
        Args:
            query: The search query
            **kwargs: Additional search parameters
            
        Returns:
            List of matching data items
        """
        results = []
        
        if isinstance(query, str):
            query_lower = query.lower()
            
            # Search in regular data
            for key, value in self.data.items():
                if self._matches_query(key, value, query_lower):
                    results.append(value)
            
            # Search in context items
            for item in self.context_items.values():
                if self._matches_context_item(item, query_lower):
                    results.append(item)
        
        return results
    
    def _matches_query(self, key: str, value: Any, query: str) -> bool:
        """Check if a key-value pair matches the query."""
        # Check key
        if query in key.lower():
            return True
        
        # Check value
        if isinstance(value, str) and query in value.lower():
            return True
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, str) and query in v.lower():
                    return True
                elif isinstance(k, str) and query in k.lower():
                    return True
        
        return False
    
    def _matches_context_item(self, item: ContextItem, query: str) -> bool:
        """Check if a context item matches the query."""
        # Check content
        if hasattr(item, 'content') and isinstance(item.content, str):
            if query in item.content.lower():
                return True
        
        # Check metadata
        if hasattr(item, 'metadata') and isinstance(item.metadata, dict):
            for key, value in item.metadata.items():
                if isinstance(value, str) and query in value.lower():
                    return True
                elif isinstance(key, str) and query in key.lower():
                    return True
        
        return False
    
    async def close(self) -> None:
        """Clean up resources."""
        self.data.clear()
        self.metadata.clear()
        self.context_items.clear() 