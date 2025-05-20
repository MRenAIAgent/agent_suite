from typing import Any, Dict, List, Optional, Union
import uuid
import datetime

from knowledge.base import KnowledgeBase
from knowledge.graphiti.knowledge_graph import GraphitiKnowledgeGraph
from knowledge.vector.vector_db_cohere import QdrantVectorDatabase


class CombinedKnowledgeBase(KnowledgeBase):
    """
    Combined knowledge base implementation using both Graphiti and Qdrant.
    
    This provides the benefits of both implementations:
    - Knowledge graph for structured data and relationship queries
    - Vector database for semantic search over unstructured data
    """
    
    def __init__(self,
                kg_graph_name: str = "knowledge_graph",
                vector_collection_name: str = "knowledge_vectors",
                embedding_model: str = "embed-english-v3.0",
                persist: bool = True,
                persistence_path: Optional[str] = None,
                vector_connection_url: Optional[str] = None,
                qdrant_api_key: Optional[str] = None,
                cohere_api_key: Optional[str] = None):
        """
        Initialize the combined knowledge base.
        
        Args:
            kg_graph_name: Name for the knowledge graph
            vector_collection_name: Name for the vector collection
            embedding_model: Model for generating embeddings
            persist: Whether to persist data
            persistence_path: Path for persistence
            vector_connection_url: Optional URL for Qdrant server
            qdrant_api_key: Optional Qdrant API key
            cohere_api_key: Optional Cohere API key
        """
        self.persist = persist
        self.persistence_path = persistence_path
        
        # Initialize knowledge graph
        self.kg = GraphitiKnowledgeGraph(
            graph_name=kg_graph_name,
            persist=persist,
            persistence_path=persistence_path
        )
        
        # Initialize vector database
        self.vector_db = QdrantVectorDatabase(
            collection_name=vector_collection_name,
            embedding_model=embedding_model,
            connection_url=vector_connection_url,
            persist=persist,
            persistence_path=persistence_path,
            qdrant_api_key=qdrant_api_key,
            cohere_api_key=cohere_api_key
        )
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]) -> str:
        """
        Add an entity to both knowledge graph and vector database.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            properties: Dictionary of properties for the entity
            
        Returns:
            The entity ID
        """
        # Add to knowledge graph
        self.kg.add_entity(entity_id, entity_type, properties)
        
        # Add to vector database
        self.vector_db.add_entity(entity_id, entity_type, properties)
        
        return entity_id
    
    def add_relation(self, source_id: str, relation_type: str, target_id: str, 
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relation between two entities.
        
        We store relations in both systems to ensure consistency,
        though the knowledge graph is the primary store for relationships.
        
        Args:
            source_id: ID of the source entity
            relation_type: Type of the relation
            target_id: ID of the target entity
            properties: Optional properties for the relation
            
        Returns:
            Unique identifier for the relation
        """
        # Add to knowledge graph
        relation_id = self.kg.add_relation(source_id, relation_type, target_id, properties)
        
        # Add to vector database
        try:
            self.vector_db.add_relation(source_id, relation_type, target_id, properties)
        except ValueError:
            # Entity might not exist in vector DB (shouldn't happen, but just in case)
            pass
        
        return relation_id
    
    def add_text(self, text: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add unstructured text to both systems.
        
        The vector database is primary for text storage and retrieval,
        but we also add a reference in the knowledge graph.
        
        Args:
            text: The text content
            properties: Optional metadata for the text
            
        Returns:
            Unique identifier for the text
        """
        # Add to vector database (primary for text)
        text_id = self.vector_db.add_text(text, properties)
        
        # Also add to knowledge graph for relationship tracking
        try:
            if properties is None:
                properties = {}
            
            # Copy properties and add text_id
            props = properties.copy()
            props["text_id"] = text_id
            props["vector_stored"] = True
            
            # Store in knowledge graph
            self.kg.add_entity(text_id, "text", {**props, "content": text})
        except Exception as e:
            print(f"Error adding text to knowledge graph: {e}")
        
        return text_id
    
    def query_entities(self, entity_type: Optional[str] = None, 
                      properties: Optional[Dict[str, Any]] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query entities, primarily from the knowledge graph.
        
        Args:
            entity_type: Optional type to filter entities
            properties: Optional properties to filter entities
            limit: Maximum number of results
            
        Returns:
            List of matching entities with their properties
        """
        # Knowledge graph is primary for entity querying
        return self.kg.query_entities(entity_type, properties, limit)
    
    def query_relations(self, 
                       source_id: Optional[str] = None,
                       relation_type: Optional[str] = None,
                       target_id: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query relations, primarily from the knowledge graph.
        
        Args:
            source_id: Optional source entity ID
            relation_type: Optional relation type
            target_id: Optional target entity ID
            limit: Maximum number of results
            
        Returns:
            List of matching relations with their properties
        """
        # Knowledge graph is primary for relation querying
        return self.kg.query_relations(source_id, relation_type, target_id, limit)
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search using the vector database.
        
        Args:
            query: Natural language query
            limit: Maximum number of results
            
        Returns:
            List of relevant items with relevance scores
        """
        # Vector database is primary for semantic search
        return self.vector_db.semantic_search(query, limit)
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID, using the knowledge graph.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Entity data if found, None otherwise
        """
        # Knowledge graph is primary for entity retrieval
        return self.kg.get_entity(entity_id)
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update an entity's properties in both systems.
        
        Args:
            entity_id: The entity ID
            properties: Properties to update
            
        Returns:
            True if successful, False otherwise
        """
        # Update in knowledge graph
        kg_success = self.kg.update_entity(entity_id, properties)
        
        # Update in vector database
        vector_success = self.vector_db.update_entity(entity_id, properties)
        
        return kg_success and vector_success
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from both systems.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            True if successful, False otherwise
        """
        # Delete from knowledge graph
        kg_success = self.kg.delete_entity(entity_id)
        
        # Delete from vector database
        vector_success = self.vector_db.delete_entity(entity_id)
        
        return kg_success and vector_success
    
    def delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation from both systems.
        
        Args:
            relation_id: The relation ID
            
        Returns:
            True if successful, False otherwise
        """
        # Delete from knowledge graph
        kg_success = self.kg.delete_relation(relation_id)
        
        # Delete from vector database (if stored there)
        try:
            self.vector_db.delete_relation(relation_id)
        except:
            # Not all relations may be in the vector DB
            pass
        
        return kg_success
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        kg_stats = self.kg.get_stats()
        vector_stats = self.vector_db.get_stats()
        
        return {
            "type": "combined",
            "knowledge_graph": kg_stats,
            "vector_database": vector_stats,
            "total_entities": kg_stats.get("total_entities", 0),
            "total_relations": kg_stats.get("total_relations", 0)
        } 