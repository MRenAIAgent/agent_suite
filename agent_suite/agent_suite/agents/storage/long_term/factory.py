from typing import Optional, Dict, Any

from agent_suite.agents.storage.long_term.base import LongTermMemoryBase
from agent_suite.agents.storage.long_term.kg_memory import GraphitiKnowledgeGraphMemory
from agent_suite.agents.storage.long_term.vector_memory import VectorDatabaseMemory
from agent_suite.agents.storage.long_term.combined_memory import CombinedLongTermMemory


class LongTermMemoryFactory:
    """
    Factory class for creating long-term memory instances.
    
    This class provides a simple interface for creating different types
    of long-term memory implementations based on configuration.
    """
    
    @staticmethod
    def create_memory(memory_type: str, config: Optional[Dict[str, Any]] = None) -> LongTermMemoryBase:
        """
        Create a long-term memory instance.
        
        Args:
            memory_type: Type of memory to create ('kg', 'vector', or 'combined')
            config: Optional configuration for the memory
            
        Returns:
            Instance of LongTermMemoryBase
            
        Raises:
            ValueError: If memory_type is invalid
        """
        if config is None:
            config = {}
            
        if memory_type == "kg":
            return GraphitiKnowledgeGraphMemory(
                graph_name=config.get("graph_name", "agent_memory"),
                persist=config.get("persist", True),
                persistence_path=config.get("persistence_path")
            )
        elif memory_type == "vector":
            return VectorDatabaseMemory(
                collection_name=config.get("collection_name", "agent_memory"),
                embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
                connection_url=config.get("connection_url"),
                vector_size=config.get("vector_size", 384)
            )
        elif memory_type == "combined":
            return CombinedLongTermMemory(
                kg_graph_name=config.get("kg_graph_name", "agent_kg_memory"),
                vector_collection_name=config.get("vector_collection_name", "agent_vector_memory"),
                embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
                persistence_path=config.get("persistence_path"),
                vector_connection_url=config.get("vector_connection_url")
            )
        else:
            raise ValueError(f"Invalid memory type: {memory_type}. Must be 'kg', 'vector', or 'combined'.") 