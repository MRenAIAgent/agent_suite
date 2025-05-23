"""Elasticsearch document storage implementation.

This module provides a DocumentStorage implementation using Elasticsearch,
a distributed search and analytics engine designed for full-text search,
structured search, and analytics.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple

from agents.storage.document_storage import DocumentStorage

try:
    from elasticsearch import AsyncElasticsearch, NotFoundError
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False


class ElasticsearchStorage(DocumentStorage):
    """Elasticsearch implementation of the document storage interface.
    
    Provides document storage and powerful search capabilities using
    Elasticsearch as the backend.
    """
    
    def __init__(
        self,
        hosts: Union[str, List[str]] = "http://localhost:9200",
        namespace: str = "default",
        **kwargs
    ):
        """Initialize the Elasticsearch storage.
        
        Args:
            hosts: Elasticsearch host(s) to connect to
            namespace: Namespace for grouping indexes
            **kwargs: Additional connection parameters
            
        Raises:
            ImportError: If Elasticsearch client is not installed
        """
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError(
                "Elasticsearch client is required for ElasticsearchStorage. "
                "Install it with: pip install elasticsearch"
            )
        
        self.hosts = hosts
        self.namespace = namespace
        
        # Initialize client
        self.client = AsyncElasticsearch(hosts, **kwargs)
        
        # Store a record of created indices
        self.created_indices = set()
    
    async def close(self) -> None:
        """Close the Elasticsearch connection."""
        if self.client:
            await self.client.close()
    
    def _get_index_name(self, collection: str) -> str:
        """Get the full index name with namespace prefix.
        
        Args:
            collection: Base index name
            
        Returns:
            Full index name
        """
        # Elasticsearch index names must be lowercase
        return f"{self.namespace}_{collection}".lower()
    
    async def _ensure_index(self, collection: str) -> None:
        """Ensure the index exists.
        
        Args:
            collection: Collection/index to ensure exists
        """
        index_name = self._get_index_name(collection)
        
        # Only check/create once per session
        if index_name in self.created_indices:
            return
            
        # Check if index exists
        exists = await self.client.indices.exists(index=index_name)
        if not exists:
            # Create the index with default settings
            await self.client.indices.create(
                index=index_name,
                body={
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "dynamic": True
                    }
                }
            )
        
        self.created_indices.add(index_name)
    
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
        await self._ensure_index(collection)
        
        # Use the key as the document ID if none is provided
        doc_id = key
        
        # Store in Elasticsearch
        result = await self.client.index(
            index=self._get_index_name(collection),
            body=data,
            id=doc_id,
            refresh=kwargs.get("refresh", True)
        )
        
        return result["_id"]
    
    async def retrieve(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve data by key.
        
        Args:
            key: Unique identifier for the data
            **kwargs: Additional retrieval parameters
            
        Returns:
            The stored data or None if not found
        """
        collection = kwargs.get("collection", "default")
        
        try:
            result = await self.client.get(
                index=self._get_index_name(collection),
                id=key
            )
            
            # Format the response
            document = result["_source"]
            document["id"] = result["_id"]
            
            return document
        except NotFoundError:
            return None
        except Exception as e:
            print(f"Error retrieving from Elasticsearch: {e}")
            return None
    
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete data by key.
        
        Args:
            key: Unique identifier for the data to delete
            **kwargs: Additional deletion parameters
            
        Returns:
            True if deletion was successful, False otherwise
        """
        collection = kwargs.get("collection", "default")
        
        try:
            result = await self.client.delete(
                index=self._get_index_name(collection),
                id=key,
                refresh=kwargs.get("refresh", True)
            )
            
            return result["result"] == "deleted"
        except NotFoundError:
            return False
        except Exception as e:
            print(f"Error deleting from Elasticsearch: {e}")
            return False
    
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
        upsert = kwargs.get("upsert", False)
        
        try:
            # Determine if it's a partial update or full document replacement
            if kwargs.get("partial", True):
                # Partial update
                result = await self.client.update(
                    index=self._get_index_name(collection),
                    id=key,
                    body={"doc": data, "doc_as_upsert": upsert},
                    refresh=kwargs.get("refresh", True)
                )
            else:
                # Full document replacement
                result = await self.client.index(
                    index=self._get_index_name(collection),
                    body=data,
                    id=key,
                    refresh=kwargs.get("refresh", True)
                )
            
            return result["result"] in ("updated", "created", "noop")
        except NotFoundError:
            return False
        except Exception as e:
            print(f"Error updating Elasticsearch: {e}")
            return False
    
    async def list(self, pattern: str = "*", **kwargs) -> List[str]:
        """List keys matching a pattern.
        
        Args:
            pattern: Pattern to match keys against
            **kwargs: Additional parameters for the listing operation
            
        Returns:
            List of matching keys
        """
        collection = kwargs.get("collection", "default")
        limit = kwargs.get("limit", 100)
        
        # Convert glob pattern to Elasticsearch wildcard query
        if pattern == "*":
            query = {"match_all": {}}
        else:
            # Convert glob pattern to Elasticsearch wildcard pattern
            es_pattern = pattern.replace("?", "?").replace("*", "*")
            
            query = {
                "wildcard": {
                    "_id": es_pattern
                }
            }
        
        try:
            result = await self.client.search(
                index=self._get_index_name(collection),
                body={
                    "query": query,
                    "size": limit,
                    "_source": False
                }
            )
            
            return [hit["_id"] for hit in result["hits"]["hits"]]
        except Exception as e:
            print(f"Error listing from Elasticsearch: {e}")
            return []
    
    async def clear(self, **kwargs) -> bool:
        """Clear all data from the storage.
        
        Args:
            **kwargs: Additional parameters for the clear operation
            
        Returns:
            True if clear was successful, False otherwise
        """
        collection = kwargs.get("collection", None)
        
        try:
            if collection:
                # Clear a specific index
                index_name = self._get_index_name(collection)
                
                # Check if index exists before deleting
                exists = await self.client.indices.exists(index=index_name)
                if exists:
                    await self.client.indices.delete(index=index_name)
                    await self._ensure_index(collection)
                    self.created_indices.discard(index_name)
            else:
                # Clear all indices in the namespace
                indices = await self.client.indices.get(index=f"{self.namespace}_*")
                if indices:
                    await self.client.indices.delete(index=f"{self.namespace}_*")
                    self.created_indices.clear()
            
            return True
        except Exception as e:
            print(f"Error clearing Elasticsearch: {e}")
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
            fields: Fields to index with their index types
            index_name: Optional name for the index
            **kwargs: Additional indexing parameters
            
        Returns:
            Name or identifier of the created index
        """
        es_index_name = self._get_index_name(collection)
        
        # Convert field types to Elasticsearch mapping types
        properties = {}
        for field, type_str in fields.items():
            if type_str == "text":
                properties[field] = {"type": "text"}
            elif type_str == "keyword":
                properties[field] = {"type": "keyword"}
            elif type_str in ("int", "integer", "long"):
                properties[field] = {"type": "integer"}
            elif type_str in ("float", "double"):
                properties[field] = {"type": "float"}
            elif type_str == "date":
                properties[field] = {"type": "date"}
            elif type_str == "boolean":
                properties[field] = {"type": "boolean"}
            elif type_str == "object":
                properties[field] = {"type": "object"}
            elif type_str == "nested":
                properties[field] = {"type": "nested"}
            else:
                # Default to text for unknown types
                properties[field] = {"type": "text"}
        
        # Check if index exists
        exists = await self.client.indices.exists(index=es_index_name)
        
        try:
            if exists:
                # Update existing mapping
                await self.client.indices.put_mapping(
                    index=es_index_name,
                    body={"properties": properties}
                )
            else:
                # Create new index
                await self.client.indices.create(
                    index=es_index_name,
                    body={
                        "settings": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0
                        },
                        "mappings": {
                            "properties": properties
                        }
                    }
                )
            
            self.created_indices.add(es_index_name)
            return es_index_name
            
        except Exception as e:
            print(f"Error creating Elasticsearch index: {e}")
            return ""
    
    async def insert_document(
        self, 
        collection: str,
        document: Dict[str, Any],
        **kwargs
    ) -> str:
        """Insert a document into the specified collection.
        
        Args:
            collection: Collection/index to insert into
            document: Document to insert
            **kwargs: Additional insertion parameters
            
        Returns:
            Identifier for the inserted document
        """
        # Generate ID if not provided
        doc_id = document.get("id", str(uuid.uuid4()))
        
        return await self.store(doc_id, document, collection=collection, **kwargs)
    
    async def insert_documents(
        self, 
        collection: str,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """Insert multiple documents in a batch operation.
        
        Args:
            collection: Collection/index to insert into
            documents: Documents to insert
            **kwargs: Additional batch insertion parameters
            
        Returns:
            List of identifiers for the inserted documents
        """
        await self._ensure_index(collection)
        index_name = self._get_index_name(collection)
        
        # Build bulk operation
        operations = []
        doc_ids = []
        
        for document in documents:
            # Generate ID if not provided
            doc_id = document.get("id", str(uuid.uuid4()))
            doc_ids.append(doc_id)
            
            # Add index operation to bulk request
            operations.append({"index": {"_index": index_name, "_id": doc_id}})
            # Remove id from document to avoid duplication
            if "id" in document:
                doc_copy = document.copy()
                del doc_copy["id"]
                operations.append(doc_copy)
            else:
                operations.append(document)
        
        # Execute bulk operation
        if operations:
            result = await self.client.bulk(
                body=operations,
                refresh=kwargs.get("refresh", True)
            )
            
            # Check for errors
            if result.get("errors", False):
                # Some operations failed
                print(f"Errors in bulk operation: {result['items']}")
        
        return doc_ids
    
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
            collection: Collection/index to search
            query: Query criteria
            projection: Fields to include in the results
            sort: Sort order specification
            limit: Maximum number of results
            skip: Number of results to skip
            **kwargs: Additional query parameters
            
        Returns:
            List of matching documents
        """
        # Create source filter from projection if provided
        source = None
        if projection:
            if all(v == 1 for v in projection.values()):
                # Include only these fields
                source = list(projection.keys())
            elif all(v == 0 for v in projection.values()):
                # Exclude these fields
                source = {
                    "excludes": list(projection.keys())
                }
        
        # Convert sort specification to Elasticsearch format
        es_sort = None
        if sort:
            es_sort = []
            for field, direction in sort.items():
                es_sort.append({field: "asc" if direction == 1 else "desc"})
        
        # Build Elasticsearch query
        es_query = {"query": query}
        
        if source:
            es_query["_source"] = source
        
        if es_sort:
            es_query["sort"] = es_sort
        
        # Execute search
        try:
            result = await self.client.search(
                index=self._get_index_name(collection),
                body=es_query,
                from_=skip,
                size=limit
            )
            
            # Format results
            documents = []
            for hit in result["hits"]["hits"]:
                doc = hit["_source"]
                doc["id"] = hit["_id"]
                doc["_score"] = hit["_score"]
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error searching Elasticsearch: {e}")
            return []
    
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
            collection: Collection/index containing the document
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
            collection: Collection/index to delete from
            query: Deletion criteria
            **kwargs: Additional deletion parameters
            
        Returns:
            Number of documents deleted
        """
        try:
            result = await self.client.delete_by_query(
                index=self._get_index_name(collection),
                body={"query": query},
                refresh=kwargs.get("refresh", True)
            )
            
            return result.get("deleted", 0)
        except Exception as e:
            print(f"Error deleting documents from Elasticsearch: {e}")
            return 0
    
    async def aggregate(
        self, 
        collection: str,
        pipeline: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute an aggregation pipeline.
        
        Args:
            collection: Collection/index to aggregate
            pipeline: Aggregation pipeline stages
            **kwargs: Additional aggregation parameters
            
        Returns:
            Aggregation results
        """
        # Convert pipeline to Elasticsearch aggregation format
        # This is a simplified example as full conversion is complex
        es_aggs = {}
        
        # Simple conversion for common patterns
        for stage in pipeline:
            stage_type = list(stage.keys())[0]
            
            if stage_type == "$match":
                # This will be handled in the query part
                continue
                
            elif stage_type == "$group":
                group_spec = stage["$group"]
                group_by = group_spec.get("_id")
                
                if isinstance(group_by, str) and group_by.startswith("$"):
                    field = group_by[1:]  # Remove $ prefix
                    es_aggs["terms"] = {
                        "field": field,
                        "size": kwargs.get("size", 100)
                    }
                
                # Convert aggregations
                for field, agg in group_spec.items():
                    if field == "_id":
                        continue
                        
                    if isinstance(agg, dict):
                        agg_type = list(agg.keys())[0]
                        
                        if agg_type == "$sum":
                            target = agg["$sum"]
                            if isinstance(target, str) and target.startswith("$"):
                                field_name = target[1:]
                                es_aggs[field] = {"sum": {"field": field_name}}
                        
                        elif agg_type == "$avg":
                            target = agg["$avg"]
                            if isinstance(target, str) and target.startswith("$"):
                                field_name = target[1:]
                                es_aggs[field] = {"avg": {"field": field_name}}
                        
                        elif agg_type == "$min":
                            target = agg["$min"]
                            if isinstance(target, str) and target.startswith("$"):
                                field_name = target[1:]
                                es_aggs[field] = {"min": {"field": field_name}}
                        
                        elif agg_type == "$max":
                            target = agg["$max"]
                            if isinstance(target, str) and target.startswith("$"):
                                field_name = target[1:]
                                es_aggs[field] = {"max": {"field": field_name}}
                                
        # Execute aggregation
        try:
            query = {}
            for stage in pipeline:
                if "$match" in stage:
                    query = stage["$match"]
                    break
            
            result = await self.client.search(
                index=self._get_index_name(collection),
                body={
                    "size": 0,  # We only want aggregation results
                    "query": query,
                    "aggs": es_aggs
                }
            )
            
            return result.get("aggregations", {})
        except Exception as e:
            print(f"Error aggregating in Elasticsearch: {e}")
            return []
    
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
            collection: Collection/index to search
            query: Text query
            fields: Fields to search in
            limit: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of matching documents with relevance scores
        """
        # Build Elasticsearch query
        if fields:
            es_query = {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields"
                }
            }
        else:
            es_query = {
                "query_string": {
                    "query": query
                }
            }
        
        # Highlight configuration
        highlight = None
        if kwargs.get("highlight", True):
            highlight = {
                "fields": {}
            }
            
            if fields:
                for field in fields:
                    highlight["fields"][field] = {}
            else:
                highlight["fields"]["*"] = {}
        
        # Execute search
        try:
            body = {
                "query": es_query,
                "size": limit
            }
            
            if highlight:
                body["highlight"] = highlight
                
            result = await self.client.search(
                index=self._get_index_name(collection),
                body=body
            )
            
            # Format results
            documents = []
            for hit in result["hits"]["hits"]:
                doc = hit["_source"]
                doc["id"] = hit["_id"]
                doc["score"] = hit["_score"]
                
                # Add highlights if present
                if "highlight" in hit:
                    doc["highlights"] = hit["highlight"]
                
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error text searching in Elasticsearch: {e}")
            return [] 