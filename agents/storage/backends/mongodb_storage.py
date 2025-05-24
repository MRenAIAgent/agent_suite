"""MongoDB document storage implementation.

This module provides a DocumentStorage implementation using MongoDB,
a popular document database for storing flexible JSON-like documents.
"""

import json
from typing import Any, Dict, List, Optional, Union, Tuple

from agents.storage.document_storage import DocumentStorage

try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult
    from bson.objectid import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    # Define dummy types for type hinting when MongoDB isn't available
    class Collection:
        """Dummy Collection class when pymongo isn't available."""
        pass
    class ObjectId:
        """Dummy ObjectId class when pymongo isn't available."""
        pass
    class InsertOneResult:
        """Dummy InsertOneResult class when pymongo isn't available."""
        pass
    class InsertManyResult:
        """Dummy InsertManyResult class when pymongo isn't available."""
        pass
    class UpdateResult:
        """Dummy UpdateResult class when pymongo isn't available."""
        pass
    class DeleteResult:
        """Dummy DeleteResult class when pymongo isn't available."""
        pass


class MongoDBStorage(DocumentStorage):
    """MongoDB implementation of the document storage interface.
    
    Provides document storage and querying capabilities using
    MongoDB as the backend.
    """
    
    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "agent_memory",
        namespace: str = "default",
        **kwargs
    ):
        """Initialize the MongoDB storage.
        
        Args:
            uri: MongoDB connection URI
            database: MongoDB database name
            namespace: Namespace for grouping collections
            **kwargs: Additional connection parameters
            
        Raises:
            ImportError: If MongoDB client is not installed
        """
        if not MONGODB_AVAILABLE:
            raise ImportError(
                "PyMongo is required for MongoDBStorage. "
                "Install it with: pip install pymongo"
            )
        
        self.uri = uri
        self.database_name = database
        self.namespace = namespace
        
        # Initialize client
        self.client = MongoClient(uri, **kwargs)
        self.db = self.client[database]
        
        # Store collection references
        self.collections: Dict[str, Collection] = {}
    
    def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
    
    def _get_collection(self, collection_name: str) -> Collection:
        """Get or create a collection by name.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MongoDB collection
        """
        # Use namespace prefixing for collection names
        full_name = f"{self.namespace}_{collection_name}"
        
        if full_name not in self.collections:
            self.collections[full_name] = self.db[full_name]
        
        return self.collections[full_name]
    
    def _format_document_id(self, document_id: Any) -> str:
        """Format a document ID for consistent usage.
        
        Args:
            document_id: Document ID (could be ObjectId or str)
            
        Returns:
            String representation of the ID
        """
        return str(document_id)
    
    def _prepare_document_for_storage(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a document for storage in MongoDB.
        
        Args:
            document: Document to prepare
            
        Returns:
            Prepared document
        """
        # Create a copy to avoid modifying the original
        doc_copy = document.copy()
        
        # If there's an 'id' field but not '_id', use it as '_id'
        if 'id' in doc_copy and '_id' not in doc_copy:
            doc_copy['_id'] = doc_copy['id']
        
        # Return the prepared document
        return doc_copy
    
    def _prepare_document_for_return(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a document retrieved from MongoDB for return to the caller.
        
        Args:
            document: Document to prepare
            
        Returns:
            Prepared document
        """
        if document is None:
            return None
            
        # Create a copy to avoid modifying the original
        doc_copy = document.copy()
        
        # Convert ObjectId to string
        if '_id' in doc_copy:
            doc_copy['id'] = str(doc_copy['_id'])
            
            # Don't delete _id as it might be needed for future operations
            if 'id' not in document:  # Only if we added it
                doc_copy['id'] = str(doc_copy['_id'])
        
        return doc_copy
    
    async def store(self, key: str, data: Dict[str, Any], **kwargs) -> str:
        """Store data with the given key.
        
        Args:
            key: Unique identifier for the data
            data: The data to store
            **kwargs: Additional storage parameters
            
        Returns:
            Identifier for the stored data
        """
        collection = kwargs.get("collection", "default")
        mongodb_collection = self._get_collection(collection)
        
        # Prepare document for MongoDB
        doc = self._prepare_document_for_storage(data)
        
        # Ensure the document has an ID
        if '_id' not in doc:
            doc['_id'] = key
        
        # Insert or replace the document
        result = mongodb_collection.replace_one(
            {'_id': doc['_id']}, 
            doc, 
            upsert=True
        )
        
        return str(doc['_id'])
    
    async def retrieve(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve data by key.
        
        Args:
            key: Unique identifier for the data
            **kwargs: Additional retrieval parameters
            
        Returns:
            The stored data or None if not found
        """
        collection = kwargs.get("collection", "default")
        mongodb_collection = self._get_collection(collection)
        
        # Handle both string IDs and ObjectId
        doc_id = key
        if len(key) == 24 and all(c in '0123456789abcdef' for c in key):
            try:
                doc_id = ObjectId(key)
            except Exception:
                pass
        
        # Retrieve the document
        document = mongodb_collection.find_one({'_id': doc_id})
        
        # Prepare for return
        return self._prepare_document_for_return(document)
    
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete data by key.
        
        Args:
            key: Unique identifier for the data to delete
            **kwargs: Additional deletion parameters
            
        Returns:
            True if deletion was successful, False otherwise
        """
        collection = kwargs.get("collection", "default")
        mongodb_collection = self._get_collection(collection)
        
        # Handle both string IDs and ObjectId
        doc_id = key
        if len(key) == 24 and all(c in '0123456789abcdef' for c in key):
            try:
                doc_id = ObjectId(key)
            except Exception:
                pass
        
        # Delete the document
        result = mongodb_collection.delete_one({'_id': doc_id})
        
        return result.deleted_count == 1
    
    async def update(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """Update data associated with the key.
        
        Args:
            key: Unique identifier for the data to update
            data: New data to store
            **kwargs: Additional update parameters
            
        Returns:
            True if update was successful, False otherwise
        """
        collection = kwargs.get("collection", "default")
        mongodb_collection = self._get_collection(collection)
        upsert = kwargs.get("upsert", False)
        
        # Handle both string IDs and ObjectId
        doc_id = key
        if len(key) == 24 and all(c in '0123456789abcdef' for c in key):
            try:
                doc_id = ObjectId(key)
            except Exception:
                pass
        
        # Determine update type based on data structure
        if any(k.startswith('$') for k in data.keys()):
            # It's already a MongoDB update operation
            update_spec = data
        else:
            # It's a document - use $set to update fields
            update_spec = {'$set': data}
        
        # Update the document
        result = mongodb_collection.update_one(
            {'_id': doc_id},
            update_spec,
            upsert=upsert
        )
        
        return result.matched_count > 0 or (upsert and result.upserted_id is not None)
    
    async def list(self, pattern: str = "*", **kwargs) -> List[str]:
        """List keys matching a pattern.
        
        Args:
            pattern: Pattern to match keys against
            **kwargs: Additional parameters for the listing operation
            
        Returns:
            List of matching keys
        """
        collection = kwargs.get("collection", "default")
        mongodb_collection = self._get_collection(collection)
        limit = kwargs.get("limit", 100)
        
        # Convert glob pattern to MongoDB regex
        if pattern == "*":
            # Match everything
            query = {}
        else:
            # Convert simple glob syntax to regex
            regex = pattern
            regex = regex.replace(".", "\\.")
            regex = regex.replace("*", ".*")
            regex = regex.replace("?", ".")
            regex = f"^{regex}$"
            
            # MongoDB regex query for _id
            query = {'_id': {'$regex': regex}}
        
        # Get matching document IDs
        cursor = mongodb_collection.find(query, {'_id': 1}).limit(limit)
        
        # Convert ObjectId to string
        return [str(doc['_id']) for doc in cursor]
    
    async def clear(self, **kwargs) -> bool:
        """Clear all data from the storage.
        
        Args:
            **kwargs: Additional parameters for the clear operation
            
        Returns:
            True if clear was successful, False otherwise
        """
        # Clear a specific collection?
        collection = kwargs.get("collection", None)
        
        try:
            if collection:
                # Clear just one collection
                mongodb_collection = self._get_collection(collection)
                mongodb_collection.delete_many({})
            else:
                # Clear all collections in our namespace
                for name, coll in self.collections.items():
                    if name.startswith(f"{self.namespace}_"):
                        coll.delete_many({})
            
            return True
        except Exception:
            return False
    
    async def create_index(
        self, 
        collection: str,
        fields: Dict[str, str],
        index_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create an index for the specified collection and fields.
        
        Args:
            collection: Collection to index
            fields: Fields to index with their index types (1=ascending, -1=descending, "text"=text index)
            index_name: Optional name for the index
            **kwargs: Additional indexing parameters
            
        Returns:
            Name or identifier of the created index
        """
        mongodb_collection = self._get_collection(collection)
        
        # Convert field types to MongoDB index types
        index_fields = []
        for field, type_str in fields.items():
            if type_str == "text":
                index_fields.append((field, pymongo.TEXT))
            elif type_str in (1, "1", "asc", "ascending"):
                index_fields.append((field, pymongo.ASCENDING))
            elif type_str in (-1, "-1", "desc", "descending"):
                index_fields.append((field, pymongo.DESCENDING))
            else:
                # Default to ascending for unknown types
                index_fields.append((field, pymongo.ASCENDING))
        
        # Create the index
        index_options = {k: v for k, v in kwargs.items() if k not in ("collection", "fields")}
        if index_name:
            index_options["name"] = index_name
            
        result = mongodb_collection.create_index(index_fields, **index_options)
        
        return result
    
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
        return await self.store(
            document.get("id", str(ObjectId())),
            document,
            collection=collection,
            **kwargs
        )
    
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
        mongodb_collection = self._get_collection(collection)
        
        # Prepare documents for MongoDB
        docs_to_insert = []
        for doc in documents:
            doc = self._prepare_document_for_storage(doc)
            if '_id' not in doc:
                doc['_id'] = ObjectId()
            docs_to_insert.append(doc)
        
        # Batch insert
        result = mongodb_collection.insert_many(docs_to_insert)
        
        # Return the IDs
        return [str(id) for id in result.inserted_ids]
    
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
        mongodb_collection = self._get_collection(collection)
        
        # Execute the query
        cursor = mongodb_collection.find(query, projection)
        
        # Apply sort if specified
        if sort:
            cursor = cursor.sort(list(sort.items()))
        
        # Apply pagination
        cursor = cursor.skip(skip).limit(limit)
        
        # Prepare documents for return
        return [self._prepare_document_for_return(doc) for doc in cursor]
    
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
        return await self.update(
            document_id,
            update,
            collection=collection,
            upsert=upsert,
            **kwargs
        )
    
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
        mongodb_collection = self._get_collection(collection)
        
        # Execute the deletion
        result = mongodb_collection.delete_many(query)
        
        return result.deleted_count
    
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
        mongodb_collection = self._get_collection(collection)
        
        # Execute the aggregation
        cursor = mongodb_collection.aggregate(pipeline, **kwargs)
        
        # Prepare documents for return
        return [self._prepare_document_for_return(doc) for doc in cursor]
    
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
        mongodb_collection = self._get_collection(collection)
        
        # Check if we have text indexes
        indexes = list(mongodb_collection.list_indexes())
        has_text_index = any(
            'weights' in idx.get('weights', {}) 
            for idx in indexes
        )
        
        # Create a text index if needed
        if not has_text_index and fields:
            text_fields = {field: "text" for field in fields}
            await self.create_index(collection, text_fields)
        
        # Execute the text search
        if fields and not has_text_index:
            # If no text index available, fall back to regex search on specified fields
            or_conditions = []
            for field in fields:
                or_conditions.append({field: {"$regex": query, "$options": "i"}})
            
            search_query = {"$or": or_conditions}
            cursor = mongodb_collection.find(search_query).limit(limit)
            
            # Manual scoring - not as good as MongoDB's text search
            results = []
            for doc in cursor:
                # Simple scoring: count occurrences of query terms in searched fields
                score = 0
                for field in fields:
                    if field in doc and isinstance(doc[field], str):
                        score += doc[field].lower().count(query.lower())
                
                result_doc = self._prepare_document_for_return(doc)
                result_doc['score'] = score
                results.append(result_doc)
            
            # Sort by our manual score
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return results
        else:
            # Use MongoDB's text search
            search_query = {"$text": {"$search": query}}
            projection = {"score": {"$meta": "textScore"}}
            
            cursor = mongodb_collection.find(
                search_query, 
                projection
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            return [self._prepare_document_for_return(doc) for doc in cursor] 