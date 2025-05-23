"""Document storage interface.

This module defines specialized interfaces for document storage operations,
extending the base storage interface with document-specific functionality.
"""

from typing import Any, Dict, List, Optional, Union, Tuple

from agents.storage.base_storage import BaseStorage


class DocumentStorage(BaseStorage[Dict[str, Any]]):
    """Interface for document storage operations.
    
    Extends the base storage interface with document-specific operations
    like indexing, querying, and aggregation.
    """
    
    async def create_index(
        self, 
        collection: str,
        fields: Dict[str, str],
        index_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create an index for the specified collection and fields.
        
        Args:
            collection: Collection/table to index
            fields: Fields to index with their index types
            index_name: Optional name for the index
            **kwargs: Additional indexing parameters
            
        Returns:
            Name or identifier of the created index
        """
        raise NotImplementedError("Index creation must be implemented by subclasses")
    
    async def insert_document(
        self, 
        collection: str,
        document: Dict[str, Any],
        **kwargs
    ) -> str:
        """Insert a document into the specified collection.
        
        Args:
            collection: Collection to insert into
            document: Document to insert
            **kwargs: Additional insertion parameters
            
        Returns:
            Identifier for the inserted document
        """
        key = f"{collection}:{document.get('id', None) or document.get('_id', None)}"
        return await self.store(key, document, collection=collection, **kwargs)
    
    async def insert_documents(
        self, 
        collection: str,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """Insert multiple documents in a batch operation.
        
        Args:
            collection: Collection to insert into
            documents: Documents to insert
            **kwargs: Additional batch insertion parameters
            
        Returns:
            List of identifiers for the inserted documents
        """
        raise NotImplementedError("Batch insertion must be implemented by subclasses")
    
    async def find_documents(
        self, 
        collection: str,
        query: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        sort: Optional[Dict[str, int]] = None,
        limit: int = 100,
        skip: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find documents matching the query criteria.
        
        Args:
            collection: Collection to search
            query: Query criteria
            projection: Fields to include in the results
            sort: Sort order specification
            limit: Maximum number of results
            skip: Number of results to skip
            **kwargs: Additional query parameters
            
        Returns:
            List of matching documents
        """
        raise NotImplementedError("Document search must be implemented by subclasses")
    
    async def update_document(
        self, 
        collection: str,
        document_id: str,
        update: Dict[str, Any],
        upsert: bool = False,
        **kwargs
    ) -> bool:
        """Update a document by ID.
        
        Args:
            collection: Collection containing the document
            document_id: Document identifier
            update: Update specification
            upsert: Whether to insert if document doesn't exist
            **kwargs: Additional update parameters
            
        Returns:
            True if update was successful, False otherwise
        """
        key = f"{collection}:{document_id}"
        return await self.update(key, update, collection=collection, upsert=upsert, **kwargs)
    
    async def delete_documents(
        self, 
        collection: str,
        query: Dict[str, Any],
        **kwargs
    ) -> int:
        """Delete documents matching the query criteria.
        
        Args:
            collection: Collection to delete from
            query: Deletion criteria
            **kwargs: Additional deletion parameters
            
        Returns:
            Number of documents deleted
        """
        raise NotImplementedError("Batch deletion must be implemented by subclasses")
    
    async def aggregate(
        self, 
        collection: str,
        pipeline: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute an aggregation pipeline.
        
        Args:
            collection: Collection to aggregate
            pipeline: Aggregation pipeline stages
            **kwargs: Additional aggregation parameters
            
        Returns:
            Aggregation results
        """
        raise NotImplementedError("Aggregation must be implemented by subclasses")
    
    async def text_search(
        self, 
        collection: str,
        query: str,
        fields: Optional[List[str]] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search documents using text search capabilities.
        
        Args:
            collection: Collection to search
            query: Text query
            fields: Fields to search in
            limit: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of matching documents with relevance scores
        """
        raise NotImplementedError("Text search must be implemented by subclasses") 