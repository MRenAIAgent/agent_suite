#!/usr/bin/env python3
"""
RAG Reranking Test Script

This script tests the reranking functionality in the RAG system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable

from agent_suite.rag.models.document import Document
from agent_suite.rag.api.rag_service import RagService
from agent_suite.rag.middleware.storage_router import StorageType
from agent_suite.rag.storage.vector.base import VectorStorageAdaptor
from agent_suite.rag.api.rag_retrieval import RetrievalResult
from agent_suite.rag.storage.vector.semantic_search import SemanticSearchLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RerankableMockStorage(VectorStorageAdaptor):
    """A mock storage adaptor with configurable reranking."""
    
    def __init__(self, reranker_fn: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None):
        """
        Initialize the mock storage with reranking capability.
        
        Args:
            reranker_fn: Optional reranker function to use
        """
        self.docs = {}  # document_id -> Document
        
        # Initialize semantic search layer with the provided reranker
        self.semantic_search_layer = SemanticSearchLayer(
            reranker_fn=reranker_fn
        )
    
    async def store(self, data: Any, metadata: Dict[str, Any] = None) -> str:
        """Store a document."""
        if isinstance(data, Document):
            doc_id = data.id
            self.docs[doc_id] = data
            return doc_id
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    async def store_document(self, document: Document, chunk_size: int = 0) -> str:
        """Store a document."""
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
        """Retrieve document(s)."""
        # Check if we're doing retrieval by ID
        if isinstance(query_or_id, str) and top_k == 5 and metadata is None and query_or_id in self.docs:
            return self.docs.get(query_or_id)
        
        # Otherwise, treat it as a query
        if isinstance(query_or_id, str):
            # Define base search function
            async def base_search(query, limit, filters):
                # Simple keyword matching
                results = []
                query_terms = query.lower().split()
                
                for doc_id, doc in self.docs.items():
                    content = doc.content.lower()
                    # Count matching terms
                    matches = sum(1 for term in query_terms if term in content)
                    
                    # Only include if at least one term matches
                    if matches > 0:
                        # Baseline score based on term frequency
                        baseline_score = min(1.0, 0.4 + (0.1 * matches))
                        
                        results.append({
                            "item": doc,
                            "score": baseline_score,
                            "id": doc_id
                        })
                
                return results
            
            # Use the semantic search layer for query processing (including reranking)
            return await self.semantic_search_layer.process_query(
                base_search_fn=base_search,
                query=query_or_id,
                top_k=top_k,
                filter_criteria=metadata
            )
        
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
        """Search using the retrieve method."""
        limit = kwargs.get("limit", 5)
        metadata = kwargs.get("filter_criteria")
        return await self.retrieve(query, top_k=limit, metadata=metadata)
    
    async def semantic_search(self, query: str, top_k: int = 5, 
                            filter_criteria: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Delegate to retrieve."""
        return await self.retrieve(query, top_k=top_k, metadata=filter_criteria)
    
    async def get_embedding(self, text: str) -> List[float]:
        """Return a fake embedding."""
        import random
        return [random.random() for _ in range(10)]
    
    async def close(self) -> None:
        """No resources to clean up."""
        pass


def custom_reranker(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Custom reranker that boosts specific topics based on query content.
    
    Args:
        query: The search query
        results: Original search results
        
    Returns:
        Reranked results
    """
    logger.info(f"Applying custom reranker for query: {query}")
    
    # Lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Boost specific topics based on query
    boost_topics = {}
    
    if "python" in query_lower:
        boost_topics["python"] = 0.3
    if "ai" in query_lower or "machine learning" in query_lower:
        boost_topics["ai"] = 0.3
        boost_topics["machine learning"] = 0.3
    if "database" in query_lower or "vector" in query_lower:
        boost_topics["database"] = 0.3
        boost_topics["vector"] = 0.3
    if "rag" in query_lower or "retrieval" in query_lower:
        boost_topics["rag"] = 0.3
        boost_topics["retrieval"] = 0.3
    
    # Apply boosts
    for result in results:
        item = result.get("item")
        if not item:
            continue
            
        content = getattr(item, "content", "").lower()
        original_score = result.get("score", 0.5)
        
        # Apply topic boosts
        topic_boost = 0
        for topic, boost in boost_topics.items():
            if topic in content:
                topic_boost += boost
        
        # Update score with boost (cap at 1.0)
        result["score"] = min(1.0, original_score + topic_boost)
        
        # Debug info
        if topic_boost > 0:
            logger.debug(f"Boosted score for '{content[:30]}...' from {original_score:.2f} to {result['score']:.2f}")
    
    # Sort by adjusted score
    return sorted(results, key=lambda x: x.get("score", 0), reverse=True)


async def test_with_reranking():
    """Test the RAG system with reranking enabled."""
    logger.info("=== Testing RAG with custom reranking ===")
    
    # Create RAG service with reranking
    service = RagService()
    mock_storage = RerankableMockStorage(reranker_fn=custom_reranker)
    service.register_storage_adaptor(StorageType.VECTOR, mock_storage)
    
    # Store test documents
    docs = [
        Document(
            content="Python is a high-level programming language known for its readability and versatility.",
            metadata={"source": "test", "tags": ["programming", "python"]}
        ),
        Document(
            content="Machine learning is a field of AI that enables computers to learn from data.",
            metadata={"source": "test", "tags": ["ai", "machine learning"]}
        ),
        Document(
            content="Vector databases are optimized for similarity search with embeddings.",
            metadata={"source": "test", "tags": ["database", "vectors"]}
        ),
        Document(
            content="RAG (Retrieval Augmented Generation) combines retrieval with text generation.",
            metadata={"source": "test", "tags": ["ai", "rag"]}
        ),
        Document(
            content="Natural Language Processing is a subfield of AI focused on text understanding.",
            metadata={"source": "test", "tags": ["ai", "nlp"]}
        )
    ]
    
    for doc in docs:
        await service.store_document(doc)
    
    # Test query that should show reranking in action
    query = "AI and machine learning applications"
    
    logger.info(f"\nQuery: {query}")
    results = await service.retrieve_documents(query, top_k=5)
    
    logger.info(f"Found {len(results)} results (with reranking):")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1} (score: {result.score:.4f}): {result.item.content}")
    
    await service.close()


async def test_without_reranking():
    """Test the RAG system without reranking for comparison."""
    logger.info("\n=== Testing RAG without reranking (for comparison) ===")
    
    # Create RAG service without reranking
    service = RagService()
    mock_storage = RerankableMockStorage(reranker_fn=None)
    service.register_storage_adaptor(StorageType.VECTOR, mock_storage)
    
    # Store same test documents
    docs = [
        Document(
            content="Python is a high-level programming language known for its readability and versatility.",
            metadata={"source": "test", "tags": ["programming", "python"]}
        ),
        Document(
            content="Machine learning is a field of AI that enables computers to learn from data.",
            metadata={"source": "test", "tags": ["ai", "machine learning"]}
        ),
        Document(
            content="Vector databases are optimized for similarity search with embeddings.",
            metadata={"source": "test", "tags": ["database", "vectors"]}
        ),
        Document(
            content="RAG (Retrieval Augmented Generation) combines retrieval with text generation.",
            metadata={"source": "test", "tags": ["ai", "rag"]}
        ),
        Document(
            content="Natural Language Processing is a subfield of AI focused on text understanding.",
            metadata={"source": "test", "tags": ["ai", "nlp"]}
        )
    ]
    
    for doc in docs:
        await service.store_document(doc)
    
    # Test same query for comparison
    query = "AI and machine learning applications"
    
    logger.info(f"\nQuery: {query}")
    results = await service.retrieve_documents(query, top_k=5)
    
    logger.info(f"Found {len(results)} results (without reranking):")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1} (score: {result.score:.4f}): {result.item.content}")
    
    await service.close()


async def main():
    """Run the reranking tests."""
    try:
        # Test with and without reranking for comparison
        await test_with_reranking()
        await test_without_reranking()
        logger.info("\nRanking tests completed successfully!")
    
    except Exception as e:
        logger.error(f"Error in reranking test: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main()) 