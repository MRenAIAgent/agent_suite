#!/usr/bin/env python3
"""
RAG Semantic Search Example

This example demonstrates how to use the enhanced semantic search capabilities 
in the RAG system with the Qdrant vector database.
"""

import asyncio
import logging
import sys
from typing import List, Dict, Any

from agent_suite.rag import (
    RagService,
    Document,
    Entity,
    Relationship,
    ContextItem,
    create_rag_service
)
from agent_suite.rag.middleware.storage_router import StorageType
from agent_suite.rag.storage.vector.semantic_search import (
    SemanticSearchLayer,
    basic_query_expansion,
    relevance_reranker
)
from agent_suite.rag.utils.embedding_provider import (
    DummyEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def init_rag_service() -> RagService:
    """
    Initialize the RAG service with enhanced semantic search.
    
    Returns:
        Configured RAG service
    """
    # Configure the RAG service with storage backends
    rag_config = {
        "vector": {
            "type": "qdrant",
            "collection_name": "semantic_search_demo",
            "vector_size": 384,
            "embedding": {
                # Use dummy embeddings for demo purposes
                # In production, you would use:
                # "type": "openai",
                # "api_key": "your-api-key",
                # "model": "text-embedding-ada-002"
                # or
                # "type": "sentence-transformer",
                # "model": "all-MiniLM-L6-v2"
                "type": "dummy",
                "vector_size": 384
            }
        }
    }
    
    return await create_rag_service(rag_config)


async def load_sample_documents(rag_service: RagService) -> List[str]:
    """
    Load sample documents into the RAG service.
    
    Args:
        rag_service: The RAG service to load documents into
        
    Returns:
        List of document IDs
    """
    documents = [
        Document(
            content="Python is a high-level, interpreted programming language known for its readability and versatility. "
                   "It supports multiple programming paradigms including procedural, object-oriented, and functional programming. "
                   "Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
            metadata={"source": "wikipedia", "tags": ["programming", "language", "python"]}
        ),
        Document(
            content="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. "
                   "It focuses on developing algorithms that can learn from and make predictions on data. "
                   "Machine learning is employed in a range of computing tasks where designing explicit algorithms is unfeasible.",
            metadata={"source": "textbook", "tags": ["AI", "machine learning", "data science"]}
        ),
        Document(
            content="Vector databases are specialized database systems designed to store, manage, and search vector embeddings efficiently. "
                   "They are optimized for similarity search operations often used in machine learning applications. "
                   "Examples include Qdrant, Pinecone, and Milvus.",
            metadata={"source": "blog", "tags": ["database", "vector", "embeddings"]}
        ),
        Document(
            content="Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. "
                   "The goal of NLP is to enable computers to understand, interpret, and generate human language in a valuable way.",
            metadata={"source": "textbook", "tags": ["AI", "NLP", "language"]}
        ),
        Document(
            content="Retrieval Augmented Generation (RAG) is a technique that combines retrieval-based and generation-based approaches in natural language processing. "
                   "It first retrieves relevant documents from a knowledge base and then uses them to augment the context for text generation, "
                   "resulting in more accurate and informative responses.",
            metadata={"source": "research_paper", "tags": ["AI", "NLP", "RAG"]}
        )
    ]
    
    document_ids = []
    for doc in documents:
        doc_id = await rag_service.store_document(doc)
        document_ids.append(doc_id)
        logger.info(f"Stored document with ID: {doc_id}")
    
    return document_ids


async def semantic_search_demo(rag_service: RagService) -> None:
    """
    Demonstrate semantic search capabilities.
    
    Args:
        rag_service: The RAG service to use for search
    """
    # Define a list of test queries
    test_queries = [
        "How does Python programming work?",
        "What is machine learning?",
        "Tell me about vector databases",
        "How can NLP help with text understanding?",
        "What is RAG in AI?",
        # Test queries that benefit from query expansion
        "coding in Python",
        "algorithms that learn from data",
        "similarity search",
        "understanding text",
        "knowledge retrieval for generation"
    ]
    
    # Run each test query
    for query in test_queries:
        logger.info(f"\n\n==== Query: {query} ====")
        
        # Retrieve documents using the semantic search
        results = await rag_service.retrieve_documents(query, top_k=2)
        
        # Display results
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            content = result.item.content[:100] + "..." if len(result.item.content) > 100 else result.item.content
            logger.info(f"Result {i+1} (score: {result.score:.4f}): {content}")


async def custom_expansion_demo(rag_service: RagService) -> None:
    """
    Demonstrate custom query expansion for semantic search.
    
    Args:
        rag_service: The RAG service to use
    """
    logger.info("\n\n==== Custom Query Expansion Demo ====")
    
    # Define a custom query expansion function
    def domain_specific_expansion(query: str) -> List[str]:
        """Domain-specific query expansion with technical terms."""
        expansions = [query]
        
        # Add technical expansions based on keywords
        if "python" in query.lower():
            expansions.extend([
                f"{query} programming language",
                f"{query} code examples",
                f"{query} syntax"
            ])
        elif "machine learning" in query.lower():
            expansions.extend([
                f"{query} algorithms",
                f"{query} neural networks",
                f"{query} training data"
            ])
        elif "database" in query.lower():
            expansions.extend([
                f"{query} storage",
                f"{query} query performance",
                f"{query} indexing"
            ])
            
        return expansions
    
    # Get the vector storage adaptor
    vector_adaptor = rag_service.storage_adaptors.get(StorageType.VECTOR)
    if not vector_adaptor:
        logger.error("Vector storage adaptor not found")
        return
        
    # Replace the semantic search layer with our custom expansion
    original_semantic_search = getattr(vector_adaptor, "semantic_search", None)
    if original_semantic_search:
        # Create a new semantic search layer with our custom expansion
        vector_adaptor.semantic_search = SemanticSearchLayer(
            query_expansion_fn=domain_specific_expansion,
            reranker_fn=relevance_reranker  # Keep the default reranker
        )
    
    # Test the custom expansion with a query
    test_query = "Python for beginners"
    logger.info(f"Query with custom expansion: {test_query}")
    
    # Retrieve documents
    results = await rag_service.retrieve_documents(test_query, top_k=2)
    
    # Display results
    logger.info(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        content = result.item.content[:100] + "..." if len(result.item.content) > 100 else result.item.content
        logger.info(f"Result {i+1} (score: {result.score:.4f}): {content}")
    
    # Restore the original semantic search layer if needed
    if original_semantic_search:
        vector_adaptor.semantic_search = original_semantic_search


async def main():
    """Main function to run the example."""
    try:
        # Initialize RAG service
        logger.info("Initializing RAG service with semantic search capabilities...")
        rag_service = await init_rag_service()
        
        # Load sample documents
        logger.info("Loading sample documents...")
        await load_sample_documents(rag_service)
        
        # Run semantic search demo
        await semantic_search_demo(rag_service)
        
        # Run custom expansion demo
        await custom_expansion_demo(rag_service)
        
        # Clean up
        await rag_service.close()
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 