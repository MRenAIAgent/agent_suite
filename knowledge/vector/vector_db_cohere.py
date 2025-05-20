"""
Qdrant vector database implementation with Cohere embeddings.

This module provides a QdrantVectorDatabase implementation that leverages
the centralized Qdrant client and Cohere embedding service.
"""

import uuid
import json
import datetime
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from knowledge.base import KnowledgeBase
from agent_suite.vector.qdrant_client import QdrantManager
from agent_suite.vector.embedding_service import EmbeddingService


class QdrantVectorDatabase(KnowledgeBase):
    """
    Knowledge base implementation using Qdrant vector database with Cohere embeddings.
    
    This implementation stores text and entities as vectors for
    semantic search capabilities using Cohere's advanced embedding models.
    """
    
    def __init__(
        self, 
        collection_name: str = "knowledge_base",
        embedding_model: str = "embed-english-v3.0",
        connection_url: Optional[str] = None,
        persist: bool = True,
        persistence_path: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None
    ):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name for the vector collection
            embedding_model: Name of the Cohere embedding model
            connection_url: Optional URL for the Qdrant server
            persist: Whether to persist data locally
            persistence_path: Path for local Qdrant storage
            qdrant_api_key: Optional Qdrant API key
            cohere_api_key: Optional Cohere API key
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.persist = persist
        
        # Set up persistence path
        if persist and persistence_path is None:
            self.persistence_path = os.path.join(os.getcwd(), "knowledge_data", "vector")
            os.makedirs(self.persistence_path, exist_ok=True)
        else:
            self.persistence_path = persistence_path
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            api_key=cohere_api_key
        )
        
        # Get vector size from the embedding model
        self.vector_size = self.embedding_service.get_vector_size()
        
        # Initialize Qdrant client manager
        if connection_url:
            # Use remote server
            self.vector_db = QdrantManager(
                connection_url=connection_url,
                api_key=qdrant_api_key
            )
        elif persist and self.persistence_path:
            # Use local persistent storage
            self.vector_db = QdrantManager(
                persist_directory=self.persistence_path
            )
        else:
            # Use in-memory storage
            self.vector_db = QdrantManager()
        
        # Create collection if it doesn't exist
        self._setup_collection()
        
        # Entity and relation tracking (for structured data)
        self.entities = {}  # entity_id -> {"type": type, "properties": props}
        self.relations = {}  # relation_id -> {"source": src, "relation": rel, "target": tgt, "properties": props}
        
        # Load persistent entity and relation data if available
        if persist and self.persistence_path:
            self._load_metadata()
    
    def _setup_collection(self) -> None:
        """Set up the vector collection."""
        self.vector_db.create_collection(
            collection_name=self.collection_name,
            vector_size=self.vector_size,
            distance="cosine"
        )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        embeddings = self.embedding_service.embed_documents([text])
        return embeddings[0]
    
    def _save_metadata(self) -> None:
        """Save entity and relation data to disk."""
        if not self.persist or not self.persistence_path:
            return
            
        metadata_path = os.path.join(self.persistence_path, f"{self.collection_name}_metadata.json")
        
        metadata = {
            "entities": self.entities,
            "relations": self.relations
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def _load_metadata(self) -> None:
        """Load entity and relation data from disk."""
        if not self.persist or not self.persistence_path:
            return
            
        metadata_path = os.path.join(self.persistence_path, f"{self.collection_name}_metadata.json")
        
        if not os.path.exists(metadata_path):
            return
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.entities = metadata.get("entities", {})
                self.relations = metadata.get("relations", {})
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]) -> str:
        """
        Add an entity to the vector database.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            properties: Dictionary of properties for the entity
            
        Returns:
            The entity ID
        """
        # Prepare entity data
        properties_copy = properties.copy()
        
        # Add creation timestamp if not present
        if "created_at" not in properties_copy:
            properties_copy["created_at"] = datetime.datetime.now().isoformat()
        
        # Store entity metadata
        self.entities[entity_id] = {
            "type": entity_type,
            "properties": properties_copy
        }
        
        # Create a text representation for embedding
        entity_text = f"Entity of type {entity_type}: "
        entity_text += " ".join(f"{key}: {value}" for key, value in properties_copy.items()
                              if key not in ['created_at', 'updated_at', 'id'])
        
        # Generate embedding
        embedding = self._generate_embedding(entity_text)
        
        # Store in vector DB
        payload = {
            "type": "entity",
            "entity_type": entity_type,
            "properties": properties_copy
        }
        
        self.vector_db.add_points(
            collection_name=self.collection_name,
            ids=[entity_id],
            vectors=[embedding],
            payloads=[payload]
        )
        
        # Save metadata
        self._save_metadata()
        
        return entity_id
    
    def add_relation(self, source_id: str, relation_type: str, target_id: str, 
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relation between two entities.
        
        Args:
            source_id: ID of the source entity
            relation_type: Type of the relation
            target_id: ID of the target entity
            properties: Optional properties for the relation
            
        Returns:
            Unique identifier for the relation
        """
        # Verify that both entities exist
        if source_id not in self.entities:
            raise ValueError(f"Source entity {source_id} does not exist")
        if target_id not in self.entities:
            raise ValueError(f"Target entity {target_id} does not exist")
        
        # Initialize properties
        if properties is None:
            properties = {}
        
        properties_copy = properties.copy()
        
        # Add creation timestamp
        if "created_at" not in properties_copy:
            properties_copy["created_at"] = datetime.datetime.now().isoformat()
        
        # Generate a relation ID
        relation_id = f"rel-{uuid.uuid4()}"
        
        # Store relation metadata
        self.relations[relation_id] = {
            "source": source_id,
            "relation": relation_type,
            "target": target_id,
            "properties": properties_copy
        }
        
        # Generate text representation
        source_type = self.entities[source_id]["type"]
        target_type = self.entities[target_id]["type"]
        
        relation_text = f"Relation: {source_type} {relation_type} {target_type}. "
        relation_text += f"Source: {source_id}, Target: {target_id}. "
        relation_text += " ".join(f"{key}: {value}" for key, value in properties_copy.items()
                                if key not in ['created_at', 'updated_at', 'id'])
        
        # Generate embedding
        embedding = self._generate_embedding(relation_text)
        
        # Store in vector DB
        payload = {
            "type": "relation",
            "relation_type": relation_type,
            "source_id": source_id,
            "target_id": target_id,
            "properties": properties_copy,
            "relation_id": relation_id
        }
        
        self.vector_db.add_points(
            collection_name=self.collection_name,
            ids=[relation_id],
            vectors=[embedding],
            payloads=[payload]
        )
        
        # Save metadata
        self._save_metadata()
        
        return relation_id
    
    def add_text(self, text: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add unstructured text to the vector database.
        
        Args:
            text: The text content
            properties: Optional metadata for the text
            
        Returns:
            Unique identifier for the text
        """
        # Generate ID
        text_id = f"text-{uuid.uuid4()}"
        
        # Initialize properties
        if properties is None:
            properties = {}
            
        properties_copy = properties.copy()
        
        # Add creation timestamp
        if "created_at" not in properties_copy:
            properties_copy["created_at"] = datetime.datetime.now().isoformat()
        
        # Generate embedding
        embedding = self._generate_embedding(text)
        
        # Store in vector DB
        payload = {
            "type": "text",
            "content": text,
            "properties": properties_copy
        }
        
        self.vector_db.add_points(
            collection_name=self.collection_name,
            ids=[text_id],
            vectors=[embedding],
            payloads=[payload]
        )
        
        return text_id
    
    def query_entities(self, entity_type: Optional[str] = None, 
                      properties: Optional[Dict[str, Any]] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query entities in the vector database.
        
        Args:
            entity_type: Optional type to filter entities
            properties: Optional properties to filter entities
            limit: Maximum number of results
            
        Returns:
            List of matching entities with their properties
        """
        results = []
        
        # Build filter condition
        filter_condition = {"must": [{"key": "type", "match": {"value": "entity"}}]}
        
        if entity_type:
            filter_condition["must"].append({"key": "entity_type", "match": {"value": entity_type}})
            
        if properties:
            for key, value in properties.items():
                filter_condition["must"].append(
                    {"key": f"properties.{key}", "match": {"value": value}}
                )
        
        # In a real implementation, we would use the Qdrant filter API
        # For simplicity, we'll scan through our cached entities
        for entity_id, entity_data in self.entities.items():
            if entity_type and entity_data["type"] != entity_type:
                continue
                
            if properties:
                match = True
                for key, value in properties.items():
                    if key not in entity_data["properties"] or entity_data["properties"][key] != value:
                        match = False
                        break
                        
                if not match:
                    continue
            
            # Add to results
            results.append({
                "id": entity_id,
                "entity_type": entity_data["type"],
                "properties": entity_data["properties"]
            })
            
            if len(results) >= limit:
                break
                
        return results
    
    def query_relations(self, 
                       source_id: Optional[str] = None,
                       relation_type: Optional[str] = None,
                       target_id: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query relations in the vector database.
        
        Args:
            source_id: Optional source entity ID
            relation_type: Optional relation type
            target_id: Optional target entity ID
            limit: Maximum number of results
            
        Returns:
            List of matching relations with their properties
        """
        results = []
        
        # For simplicity, we'll scan through our cached relations
        for relation_id, relation_data in self.relations.items():
            if source_id and relation_data["source"] != source_id:
                continue
                
            if relation_type and relation_data["relation"] != relation_type:
                continue
                
            if target_id and relation_data["target"] != target_id:
                continue
            
            # Add to results
            results.append({
                "id": relation_id,
                "source_id": relation_data["source"],
                "relation_type": relation_data["relation"],
                "target_id": relation_data["target"],
                "properties": relation_data["properties"]
            })
            
            if len(results) >= limit:
                break
                
        return results
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the vector database.
        
        Args:
            query: Natural language query
            limit: Maximum number of results
            
        Returns:
            List of relevant items with relevance scores
        """
        # Generate query embedding using query-optimized embedding
        query_embedding = self.embedding_service.embed_query(query)
        
        # Search in vector DB
        search_results = self.vector_db.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Format results
        results = []
        for result in search_results:
            payload = result["payload"]
            item_type = payload.get("type")
            
            if item_type == "text":
                # Format text result
                results.append({
                    "id": result["id"],
                    "type": "text",
                    "content": payload.get("content", ""),
                    "properties": payload.get("properties", {}),
                    "score": result["score"]
                })
            elif item_type == "entity":
                # Format entity result
                results.append({
                    "id": result["id"],
                    "type": "entity",
                    "entity_type": payload.get("entity_type"),
                    "properties": payload.get("properties", {}),
                    "score": result["score"]
                })
            elif item_type == "relation":
                # Format relation result
                results.append({
                    "id": result["id"],
                    "type": "relation",
                    "relation_type": payload.get("relation_type"),
                    "source_id": payload.get("source_id"),
                    "target_id": payload.get("target_id"),
                    "properties": payload.get("properties", {}),
                    "score": result["score"]
                })
                
        return results
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Entity data if found, None otherwise
        """
        if entity_id in self.entities:
            entity_data = self.entities[entity_id]
            return {
                "id": entity_id,
                "entity_type": entity_data["type"],
                "properties": entity_data["properties"]
            }
        return None
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update an entity's properties.
        
        Args:
            entity_id: The entity ID
            properties: Properties to update
            
        Returns:
            True if successful, False otherwise
        """
        if entity_id not in self.entities:
            return False
            
        # Update local metadata
        current_properties = self.entities[entity_id]["properties"]
        entity_type = self.entities[entity_id]["type"]
        
        # Update properties
        updated_properties = current_properties.copy()
        updated_properties.update(properties)
        
        # Add updated timestamp
        updated_properties["updated_at"] = datetime.datetime.now().isoformat()
        
        # Update entity data
        self.entities[entity_id]["properties"] = updated_properties
        
        # Re-create the entity embedding
        entity_text = f"Entity of type {entity_type}: "
        entity_text += " ".join(f"{key}: {value}" for key, value in updated_properties.items()
                              if key not in ['created_at', 'updated_at', 'id'])
        
        # Generate embedding
        embedding = self._generate_embedding(entity_text)
        
        # Update in vector DB
        payload = {
            "type": "entity",
            "entity_type": entity_type,
            "properties": updated_properties
        }
        
        self.vector_db.add_points(
            collection_name=self.collection_name,
            ids=[entity_id],
            vectors=[embedding],
            payloads=[payload]
        )
        
        # Save metadata
        self._save_metadata()
        
        return True
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from the vector database.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            True if successful, False otherwise
        """
        if entity_id not in self.entities:
            return False
            
        # Remove from local metadata
        del self.entities[entity_id]
        
        # Also delete all relations involving this entity
        relation_ids_to_delete = []
        for relation_id, relation_data in self.relations.items():
            if relation_data["source"] == entity_id or relation_data["target"] == entity_id:
                relation_ids_to_delete.append(relation_id)
                
        for relation_id in relation_ids_to_delete:
            del self.relations[relation_id]
            
        # Delete from vector DB
        all_ids_to_delete = [entity_id] + relation_ids_to_delete
        
        self.vector_db.delete_points(
            collection_name=self.collection_name,
            point_ids=all_ids_to_delete
        )
        
        # Save metadata
        self._save_metadata()
        
        return True
    
    def delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation from the vector database.
        
        Args:
            relation_id: The relation ID
            
        Returns:
            True if successful, False otherwise
        """
        if relation_id not in self.relations:
            return False
            
        # Remove from local metadata
        del self.relations[relation_id]
        
        # Delete from vector DB
        self.vector_db.delete_points(
            collection_name=self.collection_name,
            point_ids=[relation_id]
        )
        
        # Save metadata
        self._save_metadata()
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with statistics
        """
        entity_types = {}
        for entity_id, entity_data in self.entities.items():
            entity_type = entity_data["type"]
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
            
        relation_types = {}
        for relation_id, relation_data in self.relations.items():
            relation_type = relation_data["relation"]
            if relation_type not in relation_types:
                relation_types[relation_type] = 0
            relation_types[relation_type] += 1
            
        # Get collection info from Qdrant
        collection_info = self.vector_db.get_collection_info(self.collection_name)
        
        return {
            "vector_db_type": "Qdrant with Cohere embeddings",
            "embedding_model": self.embedding_model_name,
            "collection_name": self.collection_name,
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "vector_count": collection_info.get("vectors_count", 0) if collection_info else 0
        } 