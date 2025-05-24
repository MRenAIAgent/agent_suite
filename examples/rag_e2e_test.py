#!/usr/bin/env python3
"""
RAG End-to-End Test

This script tests the RAG system with a real Qdrant backend for vector storage.
"""

import asyncio
import logging
import sys
from typing import Dict, Any, List, Optional
import uuid

from agent_suite.rag.models.document import Document
from agent_suite.rag.api.rag_service import RagService
from agent_suite.rag.middleware.storage_router import StorageType
from agent_suite.rag.storage.vector.qdrant_storage import QdrantStorageAdaptor, QDRANT_AVAILABLE
from agent_suite.rag.utils.embedding_provider import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider
from agent_suite.rag.storage.vector.semantic_search import relevance_reranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_rag_service() -> RagService:
    """
    Set up a RAG service with real backend components.
    
    Returns:
        Configured RagService
    """
    logger.info("Setting up RAG service with real Qdrant backend...")
    
    # Create RAG service
    service = RagService()
    
    # Check if Qdrant is available
    if not QDRANT_AVAILABLE:
        logger.error("Qdrant client not available. Please install with 'pip install qdrant-client'")
        sys.exit(1)
    
    # Create a random collection name for testing
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    try:
        # Try to use SentenceTransformer for embeddings
        embedding_provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        logger.info("Using SentenceTransformer for embeddings")
    except (ImportError, Exception) as e:
        logger.warning(f"Could not create SentenceTransformer provider: {e}")
        logger.info("Using DummyEmbeddingProvider as fallback")
        embedding_provider = DummyEmbeddingProvider(vector_size=384)
    
    # Create Qdrant adaptor
    qdrant_adaptor = QdrantStorageAdaptor(
        collection_name=collection_name,
        vector_size=384,
        embedding_provider=embedding_provider,
        host="localhost",
        port=6333
    )
    
    # Register vector storage adaptor
    service.register_storage_adaptor(StorageType.VECTOR, qdrant_adaptor)
    logger.info(f"Registered Qdrant adaptor with collection: {collection_name}")
    
    return service


async def load_test_documents(service: RagService) -> List[str]:
    """
    Load test documents into the RAG service.
    
    Args:
        service: The RAG service to load documents into
        
    Returns:
        List of document IDs
    """
    logger.info("Loading test documents...")
    
    # Sample documents to test with
    docs = [
        Document(
            content="Python is a high-level, interpreted programming language known for its readability and versatility. "
                   "It supports multiple programming paradigms and has a simple, easy-to-learn syntax which reduces the cost of program maintenance.",
            metadata={"source": "test_data", "category": "programming", "tags": ["python", "language"]}
        ),
        Document(
            content="Machine learning is a branch of artificial intelligence that focuses on developing systems that learn from data. "
                   "It encompasses techniques like neural networks, decision trees, and support vector machines to identify patterns and make predictions.",
            metadata={"source": "test_data", "category": "artificial intelligence", "tags": ["ML", "AI", "data science"]}
        ),
        Document(
            content="Vector databases are specialized database systems designed to store and search vector embeddings efficiently. "
                   "They are optimized for similarity search operations using techniques like approximate nearest neighbors (ANN) algorithms.",
            metadata={"source": "test_data", "category": "databases", "tags": ["vectors", "embeddings", "similarity search"]}
        ),
        Document(
            content="Retrieval-Augmented Generation (RAG) is an AI framework that combines retrieval-based and generation-based approaches. "
                   "It first retrieves relevant documents from a knowledge base and then uses them to generate more accurate and contextually relevant responses.",
            metadata={"source": "test_data", "category": "artificial intelligence", "tags": ["RAG", "retrieval", "NLP"]}
        ),
        Document(
            content="Natural Language Processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language. "
                   "NLP technologies power applications like machine translation, sentiment analysis, and question answering systems.",
            metadata={"source": "test_data", "category": "artificial intelligence", "tags": ["NLP", "language", "AI"]}
        ),
        Document(
            content="Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. "
                   "These deep neural networks can automatically learn hierarchical representations of data, making them powerful for tasks like image recognition.",
            metadata={"source": "test_data", "category": "artificial intelligence", "tags": ["deep learning", "neural networks", "AI"]}
        ),
        Document(
            content="Quantum computing leverages quantum mechanics principles to process information in fundamentally different ways than classical computers. "
                   "Quantum computers use qubits that can exist in multiple states simultaneously, potentially solving certain problems exponentially faster.",
            metadata={"source": "test_data", "category": "computing", "tags": ["quantum", "computing", "physics"]}
        )
    ]
    
    # Store documents
    doc_ids = []
    for doc in docs:
        doc_id = await service.store_document(doc)
        doc_ids.append(doc_id)
        logger.info(f"Stored document: {doc.content[:50]}... (ID: {doc_id})")
    
    return doc_ids


async def test_search_with_filters(service: RagService) -> None:
    """
    Test searching with metadata filters.
    
    Args:
        service: Configured RAG service
    """
    logger.info("\n=== Testing search with metadata filters ===")
    
    # Search with category filter
    query = "machine learning techniques"
    metadata = {"category": "artificial intelligence"}
    
    logger.info(f"Query: '{query}' with filter: {metadata}")
    results = await service.retrieve_documents(query, top_k=3, filter_criteria=metadata)
    
    logger.info(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1} (score: {result.score:.4f}): {result.item.content[:100]}...")


async def test_semantic_search(service: RagService) -> None:
    """
    Test semantic search capabilities.
    
    Args:
        service: Configured RAG service
    """
    logger.info("\n=== Testing semantic search capabilities ===")
    
    # Test queries
    test_queries = [
        "How does Python programming work?",
        "Explain machine learning algorithms",
        "What are vector databases used for?",
        "How does RAG improve text generation?",
        "Applications of deep learning in NLP",
        "What is quantum computing?",
    ]
    
    # Run tests
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = await service.retrieve_documents(query, top_k=2)
        
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1} (score: {result.score:.4f}): {result.item.content[:100]}...")


async def test_reranking(service: RagService) -> None:
    """
    Test reranking in the search pipeline.
    
    Args:
        service: Configured RAG service
    """
    logger.info("\n=== Testing reranking functionality ===")
    
    # Get vector storage adaptor
    vector_adaptor = service.storage_adaptors.get(StorageType.VECTOR)
    if not vector_adaptor:
        logger.error("Vector storage adaptor not found")
        return
    
    # Test a broad query to see reranking in action
    query = "AI for natural language"
    
    # First, retrieve without modification to see baseline
    logger.info(f"\nRunning baseline search for: '{query}'")
    baseline_results = await service.retrieve_documents(query, top_k=5)
    
    logger.info(f"Baseline results ({len(baseline_results)} items):")
    for i, result in enumerate(baseline_results):
        logger.info(f"Result {i+1} (score: {result.score:.4f}): {result.item.content[:50]}...")
    
    # Now, ensure a specific reranker is used
    # Save the original semantic search layer if it exists
    original_semantic_search = getattr(vector_adaptor, "semantic_search", None)
    
    # Apply a custom reranker that boosts NLP content
    from agent_suite.rag.storage.vector.semantic_search import SemanticSearchLayer
    
    def nlp_boosting_reranker(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reranker that boosts NLP-related content specifically."""
        logger.info(f"Applying NLP-boosting reranker for query: {query}")
        
        # Keywords to boost
        nlp_keywords = ["nlp", "natural language", "language processing", "text"]
        
        # Apply boosts
        for result in results:
            item = result.get("item")
            if not item:
                continue
                
            content = getattr(item, "content", "").lower()
            original_score = result.get("score", 0.5)
            
            # Count NLP keyword matches
            boost = 0
            for keyword in nlp_keywords:
                if keyword in content:
                    boost += 0.15
            
            # Apply boost
            result["score"] = min(1.0, original_score + boost)
        
        # Sort by score
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    
    # Apply the new reranker
    vector_adaptor.semantic_search = SemanticSearchLayer(
        reranker_fn=nlp_boosting_reranker
    )
    
    logger.info(f"\nRunning search with NLP-boosting reranker for: '{query}'")
    boosted_results = await service.retrieve_documents(query, top_k=5)
    
    logger.info(f"Boosted results ({len(boosted_results)} items):")
    for i, result in enumerate(boosted_results):
        logger.info(f"Result {i+1} (score: {result.score:.4f}): {result.item.content[:50]}...")
    
    # Restore the original semantic search layer if needed
    if original_semantic_search:
        vector_adaptor.semantic_search = original_semantic_search


async def main():
    """Run the end-to-end tests."""
    try:
        # Set up service with real backends
        service = await setup_rag_service()
        
        # Load test documents
        await load_test_documents(service)
        
        # Add a small delay to ensure documents are indexed
        logger.info("Waiting for documents to be indexed...")
        await asyncio.sleep(2)
        
        # Run tests
        await test_semantic_search(service)
        await test_search_with_filters(service)
        await test_reranking(service)
        
        # Clean up
        logger.info("\nCleaning up resources...")
        await service.close()
        
        logger.info("End-to-end test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in end-to-end test: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 