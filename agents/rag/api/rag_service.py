"""RagService implementation for the RAG system.

This module provides the main service class that implements both RagStore and
RagRetrieval interfaces, providing a unified API for the RAG system.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Type

from agents.rag.api.rag_store import RagStore
from agents.rag.api.rag_retrieval import RagRetrieval, RetrievalResult
from agents.rag.middleware.storage_router import StorageRouter, StorageType
from agents.rag.middleware.retrieval_orchestrator import RetrievalOrchestrator
from agents.rag.models.document import Document, DocumentChunk
from agents.rag.models.knowledge import Entity, Relationship, KnowledgeGraph
from agents.rag.models.context import ContextItem
from agents.rag.storage.base import StorageAdaptor
from agents.rag.storage.vector.base import VectorStorageAdaptor
from agents.rag.storage.graph.base import GraphStorageAdaptor
from agents.rag.storage.key_value.base import KeyValueStorageAdaptor

logger = logging.getLogger(__name__)


class RagService(RagStore, RagRetrieval):
    """
    Main service for the RAG (Retrieval Augmented Generation) system.
    
    This service implements both RagStore and RagRetrieval interfaces,
    providing a unified API for storing and retrieving information while
    abstracting away the underlying storage technologies.
    """
    
    def __init__(self):
        """Initialize the RAG service."""
        self.storage_router = StorageRouter()
        self.retrieval_orchestrator = RetrievalOrchestrator()
        
        # Storage adaptors by type
        self.storage_adaptors: Dict[StorageType, StorageAdaptor] = {}
    
    def register_storage_adaptor(self, storage_type: StorageType, 
                               adaptor: StorageAdaptor) -> None:
        """
        Register a storage adaptor for a specific storage type.
        
        Args:
            storage_type: The type of storage
            adaptor: The storage adaptor to register
        """
        self.storage_adaptors[storage_type] = adaptor
        
        # Register adaptor with retrieval orchestrator if it can search
        if hasattr(adaptor, "search"):
            self.retrieval_orchestrator.register_retrieval_handler(
                storage_type,
                adaptor
            )
    
    async def store_document(self, document: Document) -> str:
        """
        Store a document in the system.
        
        Args:
            document: The document to store
            
        Returns:
            The ID of the stored document
        """
        # Determine where to store the document
        storage_types = self.storage_router.route(document, document.metadata)
        
        # Store in each determined storage
        doc_id = document.id
        for storage_type in storage_types:
            if storage_type in self.storage_adaptors:
                adaptor = self.storage_adaptors[storage_type]
                try:
                    await adaptor.store(document, document.metadata)
                except Exception as e:
                    logger.error(f"Error storing document in {storage_type}: {e}")
        
        return doc_id
    
    async def store_knowledge(self, knowledge: Union[Entity, Relationship, KnowledgeGraph]) -> str:
        """
        Store knowledge (entity, relationship, or graph) in the system.
        
        Args:
            knowledge: The knowledge to store
            
        Returns:
            The ID of the stored knowledge
        """
        # Extract metadata
        metadata = {}
        if hasattr(knowledge, "metadata"):
            metadata = knowledge.metadata
        
        # Determine where to store the knowledge
        storage_types = self.storage_router.route(knowledge, metadata)
        
        # Store in each determined storage
        knowledge_id = getattr(knowledge, "id", None)
        for storage_type in storage_types:
            if storage_type in self.storage_adaptors:
                adaptor = self.storage_adaptors[storage_type]
                try:
                    await adaptor.store(knowledge, metadata)
                except Exception as e:
                    logger.error(f"Error storing knowledge in {storage_type}: {e}")
        
        return knowledge_id
    
    async def store_context(self, context: ContextItem) -> str:
        """
        Store contextual information in the system.
        
        Args:
            context: The context item to store
            
        Returns:
            The ID of the stored context item
        """
        # Determine where to store the context
        storage_types = self.storage_router.route(context, context.metadata)
        
        # Store in each determined storage
        context_id = context.id
        for storage_type in storage_types:
            if storage_type in self.storage_adaptors:
                adaptor = self.storage_adaptors[storage_type]
                try:
                    await adaptor.store(context, context.metadata)
                except Exception as e:
                    logger.error(f"Error storing context in {storage_type}: {e}")
        
        return context_id
    
    async def update(self, id: str, data: Any) -> bool:
        """
        Update an existing item in the store.
        
        Args:
            id: The ID of the item to update
            data: The new data
            
        Returns:
            Success status
        """
        # Determine where to update based on data type
        storage_types = self.storage_router.route(data, getattr(data, "metadata", {}))
        
        # Update in each determined storage
        success = False
        for storage_type in storage_types:
            if storage_type in self.storage_adaptors:
                adaptor = self.storage_adaptors[storage_type]
                try:
                    result = await adaptor.update(id, data)
                    success = success or result
                except Exception as e:
                    logger.error(f"Error updating item in {storage_type}: {e}")
        
        return success
    
    async def delete(self, id: str) -> bool:
        """
        Delete an item from the store.
        
        Args:
            id: The ID of the item to delete
            
        Returns:
            Success status
        """
        # Try to delete from all storage types
        success = False
        for storage_type, adaptor in self.storage_adaptors.items():
            try:
                result = await adaptor.delete(id)
                success = success or result
            except Exception as e:
                logger.error(f"Error deleting item in {storage_type}: {e}")
        
        return success
    
    async def retrieve(self, query: str, top_k: int = 5, 
                     filter_criteria: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant information based on a query.
        
        Args:
            query: The query string
            top_k: Maximum number of results to return
            filter_criteria: Optional filters to apply
            
        Returns:
            List of retrieval results with relevance scores
        """
        # Use the orchestrator to retrieve from appropriate sources
        metadata = filter_criteria or {}
        results = await self.retrieval_orchestrator.retrieve(query, top_k, metadata)
        return results
    
    async def retrieve_by_id(self, id: str) -> Optional[Any]:
        """
        Retrieve a specific item by ID.
        
        Args:
            id: The ID of the item to retrieve
            
        Returns:
            The retrieved item or None if not found
        """
        # Try to retrieve from all storage types
        for storage_type, adaptor in self.storage_adaptors.items():
            try:
                item = await adaptor.retrieve(id)
                if item is not None:
                    return item
            except Exception as e:
                logger.error(f"Error retrieving item in {storage_type}: {e}")
        
        return None
    
    async def retrieve_documents(self, query: str, top_k: int = 5,
                               filter_criteria: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Retrieve documents based on a query.
        
        Args:
            query: The query string
            top_k: Maximum number of results to return
            filter_criteria: Optional filters to apply
            
        Returns:
            List of document retrieval results with relevance scores
        """
        # Add document type filter
        metadata = filter_criteria.copy() if filter_criteria else {}
        metadata["retrieval_source"] = StorageType.VECTOR.value
        
        results = await self.retrieval_orchestrator.retrieve(query, top_k, metadata)
        
        # Filter to include only Document or DocumentChunk results
        document_results = [
            result for result in results
            if isinstance(result.item, (Document, DocumentChunk))
        ]
        
        return document_results
    
    async def retrieve_knowledge(self, query: str, top_k: int = 5,
                               filter_criteria: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Retrieve knowledge based on a query.
        
        Args:
            query: The query string
            top_k: Maximum number of results to return
            filter_criteria: Optional filters to apply
            
        Returns:
            List of knowledge retrieval results with relevance scores
        """
        # Add knowledge type filter
        metadata = filter_criteria.copy() if filter_criteria else {}
        metadata["retrieval_source"] = StorageType.GRAPH.value
        
        results = await self.retrieval_orchestrator.retrieve(query, top_k, metadata)
        
        # Filter to include only knowledge results (Entity or Relationship)
        knowledge_results = [
            result for result in results
            if isinstance(result.item, (Entity, Relationship))
        ]
        
        return knowledge_results
    
    async def retrieve_context(self, key: Optional[str] = None) -> List[ContextItem]:
        """
        Retrieve context items, optionally filtered by key.
        
        Args:
            key: Optional key to filter by
            
        Returns:
            List of context items
        """
        # Find key-value adaptor
        kv_adaptor = self.storage_adaptors.get(StorageType.KEY_VALUE)
        if not kv_adaptor or not isinstance(kv_adaptor, KeyValueStorageAdaptor):
            return []
        
        try:
            if key:
                # Get specific context item
                item = await kv_adaptor.get(key)
                return [item] if item and isinstance(item, ContextItem) else []
            else:
                # Get all context items
                return await kv_adaptor.get_all_context()
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    async def close(self) -> None:
        """Close all storage adaptors and free resources."""
        for storage_type, adaptor in self.storage_adaptors.items():
            try:
                await adaptor.close()
            except Exception as e:
                logger.error(f"Error closing adaptor for {storage_type}: {e}") 