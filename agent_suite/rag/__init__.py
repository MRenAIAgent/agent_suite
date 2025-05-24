"""RAG (Retrieval Augmented Generation) module.

This module provides a unified API for storing and retrieving information
from multiple storage backends (vector database, graph database, key-value store)
to support retrieval augmented generation.
"""

# Import main interfaces
from agent_suite.rag.api.rag_store import RagStore
from agent_suite.rag.api.rag_retrieval import RagRetrieval, RetrievalResult
from agent_suite.rag.api.rag_service import RagService

# Import models
from agent_suite.rag.models.document import Document, DocumentChunk
from agent_suite.rag.models.knowledge import Entity, Relationship, KnowledgeGraph
from agent_suite.rag.models.context import ContextItem

# Import factory
from agent_suite.rag.factory import create_rag_service

# Import middleware
from agent_suite.rag.middleware.storage_router import StorageType, StorageRouter
from agent_suite.rag.middleware.retrieval_orchestrator import RetrievalOrchestrator

__all__ = [
    # Main interfaces
    "RagStore",
    "RagRetrieval",
    "RetrievalResult",
    "RagService",
    
    # Models
    "Document",
    "DocumentChunk",
    "Entity",
    "Relationship",
    "KnowledgeGraph",
    "ContextItem",
    
    # Factory
    "create_rag_service",
    
    # Middleware
    "StorageType",
    "StorageRouter",
    "RetrievalOrchestrator",
] 