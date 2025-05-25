"""RAG (Retrieval Augmented Generation) module.

This module provides a unified API for storing and retrieving information
from multiple storage backends (vector database, graph database, key-value store)
to support retrieval augmented generation.
"""

# Import main interfaces
from agents.rag.api.rag_store import RagStore
from agents.rag.api.rag_retrieval import RagRetrieval, RetrievalResult
from agents.rag.api.rag_service import RagService

# Import models
from agents.rag.models.document import Document, DocumentChunk
from agents.rag.models.knowledge import Entity, Relationship, KnowledgeGraph
from agents.rag.models.context import ContextItem

# Import factory
from agents.rag.factory import create_rag_service

# Import middleware
from agents.rag.middleware.storage_router import StorageType, StorageRouter
from agents.rag.middleware.retrieval_orchestrator import RetrievalOrchestrator

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