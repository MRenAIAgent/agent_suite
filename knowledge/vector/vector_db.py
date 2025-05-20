import uuid
import json
import datetime
import os
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Required libraries not found. Install with 'pip install qdrant-client sentence-transformers numpy'")

from knowledge.base import KnowledgeBase


class QdrantVectorDatabase(KnowledgeBase):
    """
    Knowledge base implementation using Qdrant vector database.
    
    This implementation stores text and entities as vectors for
    semantic search capabilities.
    """
    
    def __init__(self, 
                collection_name: str = "knowledge_base",
                embedding_model: str = "all-MiniLM-L6-v2",
                connection_url: Optional[str] = None,
                persist: bool = True,
                persistence_path: Optional[str] = None,
                vector_size: int = 384):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name for the vector collection
            embedding_model: Name of the sentence-transformers model
            connection_url: Optional URL for the Qdrant server
            persist: Whether to persist data locally
            persistence_path: Path for local Qdrant storage
            vector_size: Size of the embeddings from the model
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.vector_size = vector_size
        self.persist = persist
        
        # Set up persistence path
        if persist and persistence_path is None:
            self.persistence_path = os.path.join(os.getcwd(), "knowledge_data", "vector")
            os.makedirs(self.persistence_path, exist_ok=True)
        else:
            self.persistence_path = persistence_path
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize vector DB client
        if connection_url:
            # Use remote server
            self.client = QdrantClient(url=connection_url)
        elif persist and self.persistence_path:
            # Use local persistent storage
            self.client = QdrantClient(path=self.persistence_path)
        else:
            # Use in-memory storage
            self.client = QdrantClient(":memory:")
        
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
        collections = self.client.get_collections().collections
        collection_exists = any(collection.name == self.collection_name for collection in collections)
        
        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=5000  # Build index after 5000 vectors
                )
            )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
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
        entity_text += " ".join(f"{key}: {value}" for key, value in properties_copy.items())
        
        # Generate embedding
        embedding = self._generate_embedding(entity_text)
        
        # Store in vector DB
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=entity_id,
                    vector=embedding,
                    payload={
                        "type": "entity",
                        "entity_type": entity_type,
                        "properties": properties_copy
                    }
                )
            ]
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
        # Ensure both entities exist
        if source_id not in self.entities:
            raise ValueError(f"Source entity {source_id} does not exist")
        
        if target_id not in self.entities:
            raise ValueError(f"Target entity {target_id} does not exist")
        
        # Generate a unique ID for the relation
        relation_id = str(uuid.uuid4())
        
        # Process properties
        if properties is None:
            properties = {}
            
        properties_copy = properties.copy()
        
        # Add timestamp if not present
        if "created_at" not in properties_copy:
            properties_copy["created_at"] = datetime.datetime.now().isoformat()
        
        # Store relation metadata
        self.relations[relation_id] = {
            "source_id": source_id,
            "relation_type": relation_type,
            "target_id": target_id,
            "properties": properties_copy
        }
        
        # Note: We don't store relations in the vector database directly
        # They are kept in the metadata structure
        
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
        # Generate a unique ID for the text
        text_id = str(uuid.uuid4())
        
        # Process properties
        if properties is None:
            properties = {}
            
        properties_copy = properties.copy()
        
        # Add timestamp if not present
        if "created_at" not in properties_copy:
            properties_copy["created_at"] = datetime.datetime.now().isoformat()
        
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
                        "type": "text",
                        "content": text,
                        "properties": properties_copy
                    }
                )
            ]
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
        
        # Filter entities
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
            
            results.append({
                "id": entity_id,
                "type": entity_data["type"],
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
        
        # Filter relations
        for relation_id, relation_data in self.relations.items():
            if source_id and relation_data["source_id"] != source_id:
                continue
                
            if relation_type and relation_data["relation_type"] != relation_type:
                continue
                
            if target_id and relation_data["target_id"] != target_id:
                continue
            
            results.append({
                "relation_id": relation_id,
                "source_id": relation_data["source_id"],
                "relation_type": relation_data["relation_type"],
                "target_id": relation_data["target_id"],
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
            payload = result.payload
            item_type = payload.get("type", "unknown")
            
            if item_type == "text":
                results.append({
                    "id": result.id,
                    "type": "text",
                    "content": payload.get("content", ""),
                    "score": result.score,
                    "properties": payload.get("properties", {})
                })
            elif item_type == "entity":
                results.append({
                    "id": result.id,
                    "type": "entity",
                    "entity_type": payload.get("entity_type", "unknown"),
                    "score": result.score,
                    "properties": payload.get("properties", {})
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
        if entity_id not in self.entities:
            return None
            
        entity_data = self.entities[entity_id]
        
        return {
            "id": entity_id,
            "type": entity_data["type"],
            "properties": entity_data["properties"]
        }
    
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
            
        # Update metadata
        for key, value in properties.items():
            self.entities[entity_id]["properties"][key] = value
            
        # Add update timestamp
        self.entities[entity_id]["properties"]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Create a text representation for embedding
        entity_type = self.entities[entity_id]["type"]
        entity_props = self.entities[entity_id]["properties"]
        
        entity_text = f"Entity of type {entity_type}: "
        entity_text += " ".join(f"{key}: {value}" for key, value in entity_props.items())
        
        # Generate new embedding
        embedding = self._generate_embedding(entity_text)
        
        # Update in vector DB
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=entity_id,
                    vector=embedding,
                    payload={
                        "type": "entity",
                        "entity_type": entity_type,
                        "properties": entity_props
                    }
                )
            ]
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
            
        # Remove from entities
        del self.entities[entity_id]
        
        # Find and remove related relations
        relation_ids_to_remove = []
        for relation_id, relation_data in self.relations.items():
            if relation_data["source_id"] == entity_id or relation_data["target_id"] == entity_id:
                relation_ids_to_remove.append(relation_id)
                
        for relation_id in relation_ids_to_remove:
            del self.relations[relation_id]
            
        # Remove from vector DB
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[entity_id])
            )
        except Exception as e:
            print(f"Error deleting entity from vector DB: {e}")
            
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
            
        # Remove from relations
        del self.relations[relation_id]
        
        # Save metadata
        self._save_metadata()
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with statistics
        """
        # Count entities by type
        entity_types = {}
        for entity_data in self.entities.values():
            entity_type = entity_data["type"]
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
            
        # Count relations by type
        relation_types = {}
        for relation_data in self.relations.values():
            relation_type = relation_data["relation_type"]
            if relation_type not in relation_types:
                relation_types[relation_type] = 0
            relation_types[relation_type] += 1
            
        # Get vector DB stats
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            vector_count = collection_info.vectors_count
        except Exception:
            vector_count = 0
            
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "vectors_count": vector_count
        } 