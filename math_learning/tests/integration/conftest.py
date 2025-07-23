"""
Fixtures for integration tests.
"""
import pytest
import asyncio
from typing import AsyncGenerator

from agents.rag.api.rag_service import RagService
from agents.rag.storage.graph.memory_storage import MemoryGraphStorageAdaptor
from agents.rag.storage.key_value.memory_storage import MemoryKeyValueStorage
from agents.rag.middleware.storage_router import StorageRouter
from agents.rag.middleware.retrieval_orchestrator import RetrievalOrchestrator


@pytest.fixture
async def rag_service() -> AsyncGenerator[RagService, None]:
    """Create a RAG service with memory storage for testing."""
    
    # Create memory storage adaptors
    graph_adaptor = MemoryGraphStorageAdaptor()
    kv_adaptor = MemoryKeyValueStorage()
    
    # Create middleware
    storage_router = StorageRouter()
    retrieval_orchestrator = RetrievalOrchestrator()
    
    # Create RAG service
    service = RagService()
    
    # Register storage adaptors
    from agents.rag.middleware.storage_router import StorageType
    service.register_storage_adaptor(StorageType.GRAPH, graph_adaptor)
    service.register_storage_adaptor(StorageType.KEY_VALUE, kv_adaptor)
    
    yield service
    
    # Cleanup
    await service.close()


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close() 