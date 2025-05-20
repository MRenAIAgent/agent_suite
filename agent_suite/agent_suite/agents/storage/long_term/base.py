from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Set, Tuple


class LongTermMemoryBase(ABC):
    """
    Abstract base class for long-term memory implementations.
    
    This defines the interface that all long-term memory implementations
    must adhere to, providing methods for storing, retrieving, and
    querying persistent memory.
    """
    
    @abstractmethod
    def store_fact(self, subject: str, predicate: str, object_value: Any, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a single fact in the knowledge graph.
        
        Args:
            subject: The subject entity of the fact
            predicate: The relationship type
            object_value: The object value or entity
            metadata: Optional metadata about this fact (confidence, source, etc.)
            
        Returns:
            fact_id: A unique identifier for the stored fact
        """
        pass
    
    @abstractmethod
    def store_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store unstructured text in the vector database.
        
        Args:
            text: The text content to store
            metadata: Optional metadata about this text (source, timestamp, etc.)
            
        Returns:
            text_id: A unique identifier for the stored text
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def query_text(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for relevant text passages.
        
        Args:
            query: The query text
            limit: Maximum number of results to return
            
        Returns:
            List of matching text passages with their metadata and relevance scores
        """
        pass
    
    @abstractmethod
    def get_entities(self, entity_type: Optional[str] = None) -> List[str]:
        """
        Get all entities or entities of a specific type.
        
        Args:
            entity_type: Optional type to filter entities
            
        Returns:
            List of entity identifiers
        """
        pass
    
    @abstractmethod
    def get_entity_facts(self, entity: str) -> List[Dict[str, Any]]:
        """
        Get all facts related to a specific entity.
        
        Args:
            entity: Entity identifier
            
        Returns:
            List of facts where the entity is either subject or object
        """
        pass
    
    @abstractmethod
    def delete_fact(self, fact_id: str) -> bool:
        """
        Delete a specific fact.
        
        Args:
            fact_id: The unique identifier of the fact
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_entity(self, entity: str) -> int:
        """
        Delete an entity and all its related facts.
        
        Args:
            entity: Entity identifier
            
        Returns:
            Number of facts deleted
        """
        pass
    
    @abstractmethod
    def delete_text(self, text_id: str) -> bool:
        """
        Delete a specific text passage.
        
        Args:
            text_id: The unique identifier of the text
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all stored data (facts and text).
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored data.
        
        Returns:
            Dictionary with statistics like entity count, fact count, etc.
        """
        pass 