#!/usr/bin/env python3
"""
Simple RAG Test Script

This is a simplified test for the RAG system that doesn't require external services.
It creates simple mocked storage adaptors to verify the basic functionality.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union

from agents.rag.models.document import Document
from agents.rag.api.rag_service import RagService
from agents.rag.middleware.storage_router import StorageType
from agents.rag.storage.vector.base import VectorStorageAdaptor
from agents.rag.api.rag_retrieval import RetrievalResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockVectorStorage(VectorStorageAdaptor):
    """A simple in-memory vector storage for testing."""
    
    def __init__(self):
        """Initialize the mock storage with an empty document store."""
        self.docs = {}  # document_id -> Document
        self.vectors = {}  # document_id -> "vector" (unused in this mock)
    
    async def store(self, data: Any, metadata: Dict[str, Any] = None) -> str:
        """Store a document in the mock storage."""
        if isinstance(data, Document):
            doc_id = data.id
            self.docs[doc_id] = data
            return doc_id
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    async def store_document(self, document: Document, chunk_size: int = 0) -> str:
        """Store a document in the mock storage."""
        doc_id = document.id
        self.docs[doc_id] = document
        return doc_id
    
    async def embed_and_store(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Store text as a document."""
        from uuid import uuid4
        metadata = metadata or {}
        doc_id = metadata.get("id", str(uuid4()))
        doc = Document(content=text, metadata=metadata, id=doc_id)
        self.docs[doc_id] = doc
        return doc_id
    
    async def retrieve(self, query_or_id: Any, top_k: int = 5, metadata: Dict[str, Any] = None) -> Union[Optional[Any], List[RetrievalResult]]:
        """
        Retrieve documents matching the query or a specific document by ID.
        
        Args:
            query_or_id: Either a query string or document ID
            top_k: Maximum number of results to return
            metadata: Optional metadata for filtering
            
        Returns:
            Either a single document (for ID lookup) or a list of retrieval results
        """
        # Check if we're doing retrieval by ID (with no other parameters)
        if isinstance(query_or_id, str) and top_k == 5 and metadata is None and query_or_id in self.docs:
            return self.docs.get(query_or_id)
        
        # Otherwise, treat it as a query
        if isinstance(query_or_id, str):
            return await self.semantic_search(query_or_id, top_k=top_k, filter_criteria=metadata)
        
        return []
    
    async def update(self, id: str, data: Any) -> bool:
        """Update a document."""
        if isinstance(data, Document):
            self.docs[id] = data
            return True
        return False
    
    async def delete(self, id: str) -> bool:
        """Delete a document."""
        if id in self.docs:
            del self.docs[id]
            return True
        return False
    
    async def search(self, query: Any, **kwargs) -> List[Any]:
        """Simple string matching search for testing."""
        if not isinstance(query, str):
            raise ValueError(f"Unsupported query type: {type(query)}")
        
        limit = kwargs.get("limit", 5)
        
        # Use semantic_search for string queries
        if isinstance(query, str):
            return await self.semantic_search(query, top_k=limit)
        
        return []
    
    async def semantic_search(self, query: str, top_k: int = 5, 
                            filter_criteria: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Simple keyword matching for testing."""
        results = []
        
        # Simple keyword matching with scoring based on term frequency
        query_terms = query.lower().split()
        for doc_id, doc in self.docs.items():
            content = doc.content.lower()
            
            # Count matching terms
            matches = sum(1 for term in query_terms if term in content)
            if matches > 0:
                score = min(1.0, 0.5 + (0.1 * matches))
                results.append(RetrievalResult(
                    item=doc,
                    score=score
                ))
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    async def get_embedding(self, text: str) -> List[float]:
        """Return a fake embedding."""
        import random
        # Return a random embedding for testing (10 dimensions)
        return [random.random() for _ in range(10)]
    
    async def close(self) -> None:
        """No resources to clean up."""
        pass


async def run_test():
    """Run a simple test of the RAG service."""
    # Create RAG service with mock vector storage
    logger.info("Creating RAG service with mock vector storage...")
    service = RagService()
    
    # Register vector storage adaptor
    mock_vector_storage = MockVectorStorage()
    service.register_storage_adaptor(StorageType.VECTOR, mock_vector_storage)
    
    # Store some test documents
    logger.info("Storing test documents...")
    docs = [
        Document(
            content="Python is a high-level programming language known for its readability and versatility.",
            metadata={"source": "test", "tags": ["programming", "python"]}
        ),
        Document(
            content="Machine learning is a field of AI that enables computers to learn from data.",
            metadata={"source": "test", "tags": ["AI", "machine learning"]}
        ),
        Document(
            content="Vector databases are optimized for similarity search with embeddings.",
            metadata={"source": "test", "tags": ["database", "vectors"]}
        ),
        Document(
            content="RAG (Retrieval Augmented Generation) combines retrieval with text generation.",
            metadata={"source": "test", "tags": ["AI", "RAG"]}
        )
    ]
    
    for doc in docs:
        doc_id = await service.store_document(doc)
        logger.info(f"Stored document with ID: {doc_id}")
    
    # Test different queries
    test_queries = [
        "What is Python programming?",
        "Explain machine learning",
        "How do vector databases work?",
        "What is RAG in AI?",
    ]
    
    # Run each test query
    for query in test_queries:
        logger.info(f"\n\n==== Query: {query} ====")
        
        # Retrieve documents
        results = await service.retrieve_documents(query, top_k=2)
        
        # Display results
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1} (score: {result.score:.4f}): {result.item.content}")
    
    # Clean up
    await service.close()
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_test()) 