from typing import Optional, Dict, Any

from knowledge.base import KnowledgeBase
from knowledge.graphiti.knowledge_graph import GraphitiKnowledgeGraph
from knowledge.vector.vector_db_cohere import QdrantVectorDatabase
from knowledge.combined import CombinedKnowledgeBase


class KnowledgeBaseFactory:
    """
    Factory class for creating knowledge base instances.
    
    This provides a simple interface for creating different types
    of knowledge base implementations based on configuration.
    """
    
    @staticmethod
    def create_knowledge_base(kb_type: str, config: Optional[Dict[str, Any]] = None) -> KnowledgeBase:
        """
        Create a knowledge base instance.
        
        Args:
            kb_type: Type of knowledge base ('graph', 'vector', or 'combined')
            config: Optional configuration dictionary
            
        Returns:
            A KnowledgeBase implementation
            
        Raises:
            ValueError: If kb_type is invalid
        """
        if config is None:
            config = {}
            
        if kb_type == "graph":
            return GraphitiKnowledgeGraph(
                graph_name=config.get("graph_name", "knowledge_graph"),
                persist=config.get("persist", True),
                persistence_path=config.get("persistence_path")
            )
        elif kb_type == "vector":
            return QdrantVectorDatabase(
                collection_name=config.get("collection_name", "knowledge_base"),
                embedding_model=config.get("embedding_model", "embed-english-v3.0"),
                connection_url=config.get("connection_url"),
                persist=config.get("persist", True),
                persistence_path=config.get("persistence_path"),
                qdrant_api_key=config.get("qdrant_api_key"),
                cohere_api_key=config.get("cohere_api_key")
            )
        elif kb_type == "combined":
            return CombinedKnowledgeBase(
                kg_graph_name=config.get("kg_graph_name", "knowledge_graph"),
                vector_collection_name=config.get("vector_collection_name", "knowledge_vectors"),
                embedding_model=config.get("embedding_model", "embed-english-v3.0"),
                persist=config.get("persist", True),
                persistence_path=config.get("persistence_path"),
                vector_connection_url=config.get("connection_url"),
                qdrant_api_key=config.get("qdrant_api_key"),
                cohere_api_key=config.get("cohere_api_key")
            )
        else:
            raise ValueError(f"Invalid knowledge base type: {kb_type}. Must be 'graph', 'vector', or 'combined'") 