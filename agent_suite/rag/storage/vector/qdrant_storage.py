"""Qdrant vector storage adaptor for the RAG system.

This module provides a Qdrant implementation of the vector storage adaptor interface.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Union
import asyncio
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from agent_suite.rag.storage.vector.base import VectorStorageAdaptor
from agent_suite.rag.storage.vector.semantic_search import SemanticSearchLayer, basic_query_expansion, relevance_reranker
from agent_suite.rag.models.document import Document, DocumentChunk
from agent_suite.rag.api.rag_retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class QdrantStorageAdaptor(VectorStorageAdaptor):
    """Qdrant implementation of vector storage adaptor."""
    
    def __init__(self, collection_name: str = "documents", vector_size: int = 384,
                embedding_provider: Any = None, host: str = "localhost", port: int = 6333):
        """
        Initialize the Qdrant storage adaptor.
        
        Args:
            collection_name: Name of the Qdrant collection to use
            vector_size: Size of the embedding vectors
            embedding_provider: Provider for generating embeddings
            host: Qdrant server host
            port: Qdrant server port
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant is not available. Install with 'pip install qdrant-client'")
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_provider = embedding_provider
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port)
        
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name=collection_name)
        except UnexpectedResponse:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
        
        # Initialize semantic search layer with default implementations
        self.semantic_search = SemanticSearchLayer(
            query_expansion_fn=basic_query_expansion,
            reranker_fn=relevance_reranker
        )
    
    async def store(self, data: Any, metadata: Dict[str, Any] = None) -> str:
        """
        Store data in Qdrant.
        
        Args:
            data: The data to store
            metadata: Optional metadata
            
        Returns:
            The ID of the stored data
        """
        if isinstance(data, Document):
            return await self.store_document(data)
        
        # Handle other data types as needed
        raise ValueError(f"Unsupported data type for Qdrant storage: {type(data)}")
    
    async def embed_and_store(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Embed text and store it in Qdrant.
        
        Args:
            text: The text to embed and store
            metadata: Optional metadata
            
        Returns:
            The ID of the stored vector
        """
        if not self.embedding_provider:
            raise ValueError("No embedding provider configured")
        
        metadata = metadata or {}
        
        # Generate embedding
        embedding = await self.get_embedding(text)
        
        # Create ID for the vector
        vector_id = metadata.get("id", str(uuid.uuid4()))
        
        # Prepare payload
        payload = {
            "text": text,
            **metadata
        }
        
        # Store vector in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        return vector_id
    
    async def store_document(self, document: Document, chunk_size: int = 0) -> str:
        """
        Store a document in Qdrant, potentially chunking it.
        
        Args:
            document: The document to store
            chunk_size: Optional chunk size (0 means no chunking)
            
        Returns:
            The ID of the stored document
        """
        if not self.embedding_provider:
            raise ValueError("No embedding provider configured")
        
        doc_id = document.id
        
        # Store whole document if no chunking
        if chunk_size <= 0 or not document.content:
            # Generate embedding for the document
            embedding = await self.get_embedding(document.content)
            
            # Prepare payload
            payload = {
                "document_id": doc_id,
                "content": document.content,
                **document.metadata
            }
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            return doc_id
        
        # Handle chunking
        chunks = document.chunks
        
        # If no chunks, but chunk_size > 0, create chunks
        if not chunks and chunk_size > 0:
            # Simple chunking by character count
            text = document.content
            chunk_texts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            # Create DocumentChunk objects
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=document.metadata.copy(),
                    document_id=doc_id,
                    index=i
                )
                chunks.append(chunk)
        
        # Store each chunk
        for chunk in chunks:
            # Generate embedding for the chunk
            embedding = await self.get_embedding(chunk.content)
            
            # Prepare payload
            payload = {
                "document_id": doc_id,
                "chunk_id": chunk.id,
                "chunk_index": chunk.index,
                "content": chunk.content,
                **chunk.metadata
            }
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=chunk.id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
        
        return doc_id
    
    async def retrieve(self, query_or_id: Any, top_k: int = 5, metadata: Dict[str, Any] = None) -> Union[Optional[Any], List[RetrievalResult]]:
        """
        Retrieve data by ID or perform a search based on a query.
        
        Args:
            query_or_id: The ID of the data to retrieve or a search query
            top_k: Maximum number of results to return (for queries)
            metadata: Optional metadata for filtering (for queries)
            
        Returns:
            The retrieved data, or search results if a query is provided
        """
        # Check if this is an ID lookup (with default parameters)
        if isinstance(query_or_id, str) and top_k == 5 and metadata is None:
            try:
                # Search for point by ID
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[query_or_id]
                )
                
                if not points:
                    return None
                
                point = points[0]
                payload = point.payload
                
                # Check if this is a document
                if "document_id" in payload and "content" in payload:
                    # Create Document object
                    document = Document(
                        content=payload["content"],
                        metadata={k: v for k, v in payload.items() if k not in ["content", "document_id"]},
                        id=payload["document_id"]
                    )
                    return document
                
                # Return raw payload if not a document
                return payload
                
            except Exception as e:
                logger.error(f"Error retrieving from Qdrant: {e}")
                return None
        
        # If it's a query string, perform a search
        if isinstance(query_or_id, str):
            return await self.search(query_or_id, limit=top_k, filter_criteria=metadata)
        
        # Unsupported query type
        raise ValueError(f"Unsupported query type for Qdrant retrieval: {type(query_or_id)}")
    
    async def update(self, id: str, data: Any) -> bool:
        """
        Update existing data in Qdrant.
        
        Args:
            id: The ID of the data to update
            data: The new data
            
        Returns:
            Success status
        """
        try:
            if isinstance(data, Document):
                # Re-store the document
                await self.store_document(data)
                return True
            
            # Handle other update scenarios
            return False
        except Exception as e:
            logger.error(f"Error updating in Qdrant: {e}")
            return False
    
    async def delete(self, id: str) -> bool:
        """
        Delete data by ID from Qdrant.
        
        Args:
            id: The ID of the data to delete
            
        Returns:
            Success status
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[id]
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting from Qdrant: {e}")
            return False
    
    async def semantic_search(self, query: str, top_k: int = 5, 
                            filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search in Qdrant.
        
        Args:
            query: The query text
            top_k: Maximum number of results
            filter_criteria: Optional filters
            
        Returns:
            List of matches with scores
        """
        if not self.embedding_provider:
            raise ValueError("No embedding provider configured")
        
        # Define base search function for the semantic search layer
        async def base_vector_search(query_text, limit, filters):
            # Generate embedding for the query
            query_vector = await self.get_embedding(query_text)
            
            # Convert filter_criteria to Qdrant filter
            qdrant_filter = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                
                qdrant_filter = models.Filter(
                    must=filter_conditions
                )
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter
            )
            
            # Convert to expected format
            results = []
            for hit in search_results:
                payload = hit.payload
                
                # Create document if payload has content
                item = None
                if "content" in payload:
                    if "chunk_id" in payload:
                        # This is a document chunk
                        item = DocumentChunk(
                            content=payload["content"],
                            metadata={k: v for k, v in payload.items() if k not in ["content", "chunk_id", "document_id", "chunk_index"]},
                            id=payload["chunk_id"],
                            document_id=payload.get("document_id"),
                            index=payload.get("chunk_index", 0)
                        )
                    else:
                        # This is a full document
                        item = Document(
                            content=payload["content"],
                            metadata={k: v for k, v in payload.items() if k not in ["content", "document_id"]},
                            id=payload.get("document_id", hit.id)
                        )
                else:
                    # Raw payload
                    item = payload
                
                results.append({
                    "item": item,
                    "score": hit.score,
                    "id": hit.id
                })
            
            return results
        
        # Use the semantic search layer to process the query
        return await self.semantic_search.process_query(
            base_search_fn=base_vector_search,
            query=query,
            top_k=top_k,
            filter_criteria=filter_criteria
        )
    
    async def search(self, query: Any, **kwargs) -> List[Any]:
        """
        Search for data matching the query in Qdrant.
        
        Args:
            query: The search query
            **kwargs: Additional search parameters
            
        Returns:
            List of matching data items
        """
        if isinstance(query, str):
            # Define base search function for semantic search
            async def base_vector_search(query_text, limit, filters):
                query_vector = await self.get_embedding(query_text)
                
                # Convert filter_criteria to Qdrant filter
                qdrant_filter = None
                if filters:
                    filter_conditions = []
                    for key, value in filters.items():
                        if key == "retrieval_source":
                            # Skip this as it's handled by middleware
                            continue
                        if isinstance(value, list):
                            filter_conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchAny(any=value)
                                )
                            )
                        else:
                            filter_conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchValue(value=value)
                                )
                            )
                    
                    if filter_conditions:
                        qdrant_filter = models.Filter(
                            must=filter_conditions
                        )
                
                # Search in Qdrant
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=qdrant_filter
                )
                
                # Convert to expected format
                results = []
                for hit in search_results:
                    payload = hit.payload
                    
                    # Create document if payload has content
                    item = None
                    if "content" in payload:
                        if "chunk_id" in payload:
                            # This is a document chunk
                            item = DocumentChunk(
                                content=payload["content"],
                                metadata={k: v for k, v in payload.items() if k not in ["content", "chunk_id", "document_id", "chunk_index"]},
                                id=payload["chunk_id"],
                                document_id=payload.get("document_id"),
                                index=payload.get("chunk_index", 0)
                            )
                        else:
                            # This is a full document
                            item = Document(
                                content=payload["content"],
                                metadata={k: v for k, v in payload.items() if k not in ["content", "document_id"]},
                                id=payload.get("document_id", hit.id)
                            )
                    else:
                        # Raw payload
                        item = payload
                    
                    results.append({
                        "item": item,
                        "score": hit.score,
                        "id": hit.id
                    })
                
                return results
                
            # Use the semantic search layer to process the query
            # The semantic search layer returns RetrievalResult objects directly
            return await self.semantic_search.process_query(
                base_search_fn=base_vector_search,
                query=query, 
                top_k=kwargs.get("limit", 5),
                filter_criteria=kwargs.get("filter_criteria")
            )
        
        # Handle other query types as needed
        raise ValueError(f"Unsupported query type for Qdrant search: {type(query)}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for text using the configured provider.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        if not self.embedding_provider:
            # Return random embedding for testing/demo purposes
            return list(np.random.rand(self.vector_size))
        
        try:
            return await self.embedding_provider.embed(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return random embedding as fallback
            return list(np.random.rand(self.vector_size))
    
    async def close(self) -> None:
        """Clean up Qdrant resources."""
        if hasattr(self, 'client'):
            # Qdrant client does not have an async close method
            self.client.close() 