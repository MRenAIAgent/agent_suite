import uuid
import json
import datetime
from typing import Any, Dict, List, Optional, Union, Set, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from agent_suite.agents.storage.long_term.base import LongTermMemoryBase


class VectorDatabaseMemory(LongTermMemoryBase):
    """
    Long-term memory implementation using vector database for unstructured data.
    
    This implementation uses Sentence Transformers to generate embeddings
    and Qdrant as the vector database for storage and similarity search.
    """
    
    def __init__(self, 
                collection_name: str = "agent_memory",
                embedding_model: str = "all-MiniLM-L6-v2",
                connection_url: Optional[str] = None,
                vector_size: int = 384):
        """
        Initialize the vector database memory.
        
        Args:
            collection_name: Name for the memory collection
            embedding_model: Name of the sentence-transformers model to use
            connection_url: Optional URL for the Qdrant server
            vector_size: Size of vector embeddings (must match the model)
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.vector_size = vector_size
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize vector DB client
        if connection_url:
            self.client = QdrantClient(url=connection_url)
        else:
            # Use in-memory storage if no connection URL provided
            self.client = QdrantClient(":memory:")
        
        # Create collection if it doesn't exist
        self._setup_collection()
        
        # Track stored facts (not using vector DB for structured data)
        self.facts = {}
    
    def _setup_collection(self) -> None:
        """Set up the vector collection."""
        collections = self.client.get_collections().collections
        if not any(collection.name == self.collection_name for collection in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def store_fact(self, subject: str, predicate: str, object_value: Any, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a single fact.
        
        Note: This implementation stores facts in a simple dictionary,
        not in the vector database, as facts are structured data.
        
        Args:
            subject: The subject entity of the fact
            predicate: The relationship type
            object_value: The object value or entity
            metadata: Optional metadata about this fact
            
        Returns:
            fact_id: A unique identifier for the stored fact
        """
        fact_id = str(uuid.uuid4())
        
        # Process metadata
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
        
        # Store the fact
        self.facts[fact_id] = {
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
            "metadata": metadata
        }
        
        return fact_id
    
    def store_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store unstructured text in the vector database.
        
        Args:
            text: The text content to store
            metadata: Optional metadata about this text
            
        Returns:
            text_id: A unique identifier for the stored text
        """
        text_id = str(uuid.uuid4())
        
        # Process metadata
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            
        # Add an ID to the metadata
        metadata["id"] = text_id
        
        # Generate embedding
        embedding = self._generate_embedding(text)
        
        # Store in vector DB
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=text_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "metadata": metadata
                    }
                )
            ]
        )
        
        return text_id
    
    def query_facts(self, query: Union[str, Dict[str, Any]], 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the stored facts.
        
        Args:
            query: Either a natural language query or a structured query dict
            limit: Maximum number of results to return
            
        Returns:
            List of matching facts with their metadata
        """
        results = []
        
        if isinstance(query, str):
            # Natural language query - use simple keyword matching
            keywords = query.lower().split()
            
            for fact_id, fact in self.facts.items():
                # Check if any keyword is in the subject, predicate, or object
                subject_match = any(keyword in str(fact["subject"]).lower() for keyword in keywords)
                predicate_match = any(keyword in fact["predicate"].lower() for keyword in keywords)
                object_match = any(keyword in str(fact["object"]).lower() for keyword in keywords)
                
                if subject_match or predicate_match or object_match:
                    confidence = 0.0
                    if subject_match:
                        confidence += 0.4
                    if predicate_match:
                        confidence += 0.4
                    if object_match:
                        confidence += 0.4
                    
                    results.append({
                        **fact,
                        "fact_id": fact_id,
                        "confidence": min(confidence, 1.0)
                    })
        else:
            # Structured query
            subject = query.get("subject")
            predicate = query.get("predicate")
            object_value = query.get("object")
            
            for fact_id, fact in self.facts.items():
                match = True
                
                if subject and fact["subject"] != subject:
                    match = False
                if predicate and fact["predicate"] != predicate:
                    match = False
                if object_value is not None and fact["object"] != object_value:
                    match = False
                
                if match:
                    results.append({
                        **fact,
                        "fact_id": fact_id,
                        "confidence": 1.0  # Exact match
                    })
        
        # Sort by confidence
        results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
        
        return results[:limit]
    
    def query_text(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for relevant text passages.
        
        Args:
            query: The query text
            limit: Maximum number of results to return
            
        Returns:
            List of matching text passages with their metadata and relevance scores
        """
        # Generate embedding for the query
        query_embedding = self._generate_embedding(query)
        
        # Search in vector DB
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "text": result.payload["text"],
                "score": result.score,
                "metadata": result.payload["metadata"]
            })
        
        return results
    
    def get_entities(self, entity_type: Optional[str] = None) -> List[str]:
        """
        Get all entities or entities of a specific type.
        
        This implementation doesn't track entity types, so all unique
        entities are returned regardless of the entity_type parameter.
        
        Args:
            entity_type: Optional type to filter entities (ignored)
            
        Returns:
            List of entity identifiers
        """
        entities = set()
        
        for fact in self.facts.values():
            entities.add(fact["subject"])
            if isinstance(fact["object"], str):
                entities.add(fact["object"])
        
        return list(entities)
    
    def get_entity_facts(self, entity: str) -> List[Dict[str, Any]]:
        """
        Get all facts related to a specific entity.
        
        Args:
            entity: Entity identifier
            
        Returns:
            List of facts where the entity is either subject or object
        """
        results = []
        
        for fact_id, fact in self.facts.items():
            if fact["subject"] == entity or fact["object"] == entity:
                results.append({
                    **fact,
                    "fact_id": fact_id
                })
        
        return results
    
    def delete_fact(self, fact_id: str) -> bool:
        """
        Delete a specific fact.
        
        Args:
            fact_id: The unique identifier of the fact
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if fact_id in self.facts:
            del self.facts[fact_id]
            return True
        return False
    
    def delete_entity(self, entity: str) -> int:
        """
        Delete an entity and all its related facts.
        
        Args:
            entity: Entity identifier
            
        Returns:
            Number of facts deleted
        """
        facts_to_delete = [
            fact_id for fact_id, fact in self.facts.items()
            if fact["subject"] == entity or fact["object"] == entity
        ]
        
        for fact_id in facts_to_delete:
            del self.facts[fact_id]
        
        return len(facts_to_delete)
    
    def delete_text(self, text_id: str) -> bool:
        """
        Delete a specific text passage.
        
        Args:
            text_id: The unique identifier of the text
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[text_id]
                )
            )
            return True
        except Exception:
            return False
    
    def clear(self) -> None:
        """
        Clear all stored data (facts and text).
        """
        # Clear facts
        self.facts = {}
        
        # Recreate collection (clear vector DB)
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
        self._setup_collection()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored data.
        
        Returns:
            Dictionary with statistics
        """
        # Count facts
        fact_count = len(self.facts)
        
        # Count predicates
        predicates = {}
        for fact in self.facts.values():
            predicate = fact["predicate"]
            if predicate not in predicates:
                predicates[predicate] = 0
            predicates[predicate] += 1
        
        # Count unique entities
        entities = set()
        for fact in self.facts.values():
            entities.add(fact["subject"])
            if isinstance(fact["object"], str):
                entities.add(fact["object"])
        
        # Get vector DB stats
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            vector_count = collection_info.vectors_count
        except Exception:
            vector_count = 0
        
        return {
            "fact_count": fact_count,
            "vector_count": vector_count,
            "entity_count": len(entities),
            "predicates": predicates
        } 