from typing import Any, Dict, List, Optional, Union, Set, Tuple
import uuid
import datetime

from agent_suite.agents.storage.long_term.base import LongTermMemoryBase
from agent_suite.agents.storage.long_term.kg_memory import GraphitiKnowledgeGraphMemory
from agent_suite.agents.storage.long_term.vector_memory import VectorDatabaseMemory


class CombinedLongTermMemory(LongTermMemoryBase):
    """
    Combined long-term memory implementation using both knowledge graph and vector database.
    
    This implementation uses:
    - Graphiti knowledge graph for structured data (facts)
    - Vector database for unstructured data (text)
    
    This provides the best of both worlds:
    - Relationship-based querying and reasoning through the knowledge graph
    - Semantic similarity search through the vector database
    """
    
    def __init__(self, 
                kg_graph_name: str = "agent_kg_memory",
                vector_collection_name: str = "agent_vector_memory",
                embedding_model: str = "all-MiniLM-L6-v2",
                persistence_path: Optional[str] = None,
                vector_connection_url: Optional[str] = None):
        """
        Initialize the combined memory.
        
        Args:
            kg_graph_name: Name for the knowledge graph
            vector_collection_name: Name for the vector database collection
            embedding_model: Name of the embedding model for vector DB
            persistence_path: Path for persisting the knowledge graph
            vector_connection_url: Optional URL for the vector database server
        """
        # Initialize knowledge graph memory
        self.kg_memory = GraphitiKnowledgeGraphMemory(
            graph_name=kg_graph_name,
            persist=persistence_path is not None,
            persistence_path=persistence_path
        )
        
        # Initialize vector database memory
        self.vector_memory = VectorDatabaseMemory(
            collection_name=vector_collection_name,
            embedding_model=embedding_model,
            connection_url=vector_connection_url
        )
    
    def store_fact(self, subject: str, predicate: str, object_value: Any, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a single fact in the knowledge graph.
        
        Args:
            subject: The subject entity of the fact
            predicate: The relationship type
            object_value: The object value or entity
            metadata: Optional metadata about this fact
            
        Returns:
            fact_id: A unique identifier for the stored fact
        """
        return self.kg_memory.store_fact(subject, predicate, object_value, metadata)
    
    def store_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store unstructured text in the vector database.
        
        Args:
            text: The text content to store
            metadata: Optional metadata about this text
            
        Returns:
            text_id: A unique identifier for the stored text
        """
        # Store the text in the vector database for semantic search
        text_id = self.vector_memory.store_text(text, metadata)
        
        # Also store a reference in the knowledge graph for graph traversal
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "text_id": text_id,
            "vector_stored": True,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Create a text node in the knowledge graph
        self.kg_memory.store_text(text, metadata)
        
        return text_id
    
    def query_facts(self, query: Union[str, Dict[str, Any]], 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph for facts.
        
        Args:
            query: Either a natural language query or a structured query dict
            limit: Maximum number of results to return
            
        Returns:
            List of matching facts with their metadata
        """
        return self.kg_memory.query_facts(query, limit)
    
    def query_text(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query both vector database and knowledge graph for text content.
        
        This implementation prioritizes vector search results but may
        supplement with keyword-based results from the knowledge graph.
        
        Args:
            query: The query text
            limit: Maximum number of results to return
            
        Returns:
            List of matching text passages with their metadata and relevance scores
        """
        # Get results from vector search (semantic similarity)
        vector_results = self.vector_memory.query_text(query, limit)
        
        # Get results from knowledge graph (keyword matching)
        kg_results = self.kg_memory.query_text(query, limit)
        
        # Combine and deduplicate results
        combined_results = []
        seen_ids = set()
        
        # Add vector results first (typically higher quality)
        for result in vector_results:
            combined_results.append(result)
            seen_ids.add(result["id"])
        
        # Add KG results if there's room and they're not duplicates
        remaining_slots = limit - len(combined_results)
        if remaining_slots > 0:
            for result in kg_results:
                if result["id"] not in seen_ids and len(combined_results) < limit:
                    combined_results.append(result)
                    seen_ids.add(result["id"])
        
        return combined_results
    
    def get_entities(self, entity_type: Optional[str] = None) -> List[str]:
        """
        Get all entities or entities of a specific type from the knowledge graph.
        
        Args:
            entity_type: Optional type to filter entities
            
        Returns:
            List of entity identifiers
        """
        return self.kg_memory.get_entities(entity_type)
    
    def get_entity_facts(self, entity: str) -> List[Dict[str, Any]]:
        """
        Get all facts related to a specific entity from the knowledge graph.
        
        Args:
            entity: Entity identifier
            
        Returns:
            List of facts where the entity is either subject or object
        """
        return self.kg_memory.get_entity_facts(entity)
    
    def delete_fact(self, fact_id: str) -> bool:
        """
        Delete a specific fact from the knowledge graph.
        
        Args:
            fact_id: The unique identifier of the fact
            
        Returns:
            True if deleted successfully, False otherwise
        """
        return self.kg_memory.delete_fact(fact_id)
    
    def delete_entity(self, entity: str) -> int:
        """
        Delete an entity and all its related facts from the knowledge graph.
        
        Args:
            entity: Entity identifier
            
        Returns:
            Number of facts deleted
        """
        return self.kg_memory.delete_entity(entity)
    
    def delete_text(self, text_id: str) -> bool:
        """
        Delete a specific text passage from both vector database and knowledge graph.
        
        Args:
            text_id: The unique identifier of the text
            
        Returns:
            True if deleted successfully, False otherwise
        """
        # Delete from vector database
        vector_success = self.vector_memory.delete_text(text_id)
        
        # Delete from knowledge graph
        kg_success = self.kg_memory.delete_text(text_id)
        
        # Return true only if both deletions were successful
        return vector_success and kg_success
    
    def clear(self) -> None:
        """
        Clear all stored data from both knowledge graph and vector database.
        """
        self.kg_memory.clear()
        self.vector_memory.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get combined statistics about the stored data.
        
        Returns:
            Dictionary with statistics from both memory systems
        """
        kg_stats = self.kg_memory.get_stats()
        vector_stats = self.vector_memory.get_stats()
        
        combined_stats = {
            **kg_stats,
            "vector_database": vector_stats
        }
        
        return combined_stats 