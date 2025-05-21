"""
Factory class for creating memory components.

This module provides a factory to easily create memory components
with different configurations and backends.
"""

from typing import Dict, Any, Optional, Union

from agent_suite.memory.memory_manager import MemoryManager
from agent_suite.memory.context.base import ContextMemory
from agent_suite.memory.context.in_memory import InMemoryContextMemory
from agent_suite.memory.context.redis import RedisContextMemory

from agent_suite.memory.long_term.base import LongTermMemory
from agent_suite.memory.long_term.vector.redis import RedisVectorMemory
from agent_suite.memory.long_term.graph.graphiti import GraphitiMemory

# For vector memory
try:
    from agent_suite.vector.embedding_service import EmbeddingService
except ImportError:
    # Fallback for when embedding_service is not available
    EmbeddingService = None


class MemoryFactory:
    """
    Factory for creating memory components and managers.
    
    This class provides static methods to create context memory,
    long-term memory, and memory managers with different configurations.
    """
    
    @staticmethod
    def create_context_memory(config: Dict[str, Any]) -> ContextMemory:
        """
        Create a context memory implementation based on configuration.
        
        Args:
            config: Configuration dictionary with the following fields:
                - type: Type of context memory ('in_memory' or 'redis')
                - redis_url: Redis connection URL (for Redis type)
                - namespace: Redis key namespace (for Redis type)
                - ttl: Optional time-to-live for Redis keys (for Redis type)
                
        Returns:
            Implementation of ContextMemory
            
        Raises:
            ValueError: If the context memory type is not supported
        """
        memory_type = config.get("type", "in_memory")
        
        if memory_type == "in_memory":
            return InMemoryContextMemory()
        
        elif memory_type == "redis":
            redis_url = config.get("redis_url", "redis://localhost:6379")
            namespace = config.get("namespace", "agent_context:")
            ttl = config.get("ttl")
            
            return RedisContextMemory(redis_url=redis_url, namespace=namespace, ttl=ttl)
        
        else:
            raise ValueError(f"Unsupported context memory type: {memory_type}")
    
    @staticmethod
    def create_long_term_memory(config: Dict[str, Any]) -> LongTermMemory:
        """
        Create a long-term memory implementation based on configuration.
        
        Args:
            config: Configuration dictionary with the following fields:
                - type: Type of long-term memory ('redis_vector' or 'graphiti')
                - embedding_service: Embedding service instance (for vector memory)
                - redis_url: Redis connection URL (for Redis type)
                - index_name: Name of the Redis search index (for Redis type)
                - prefix: Prefix for Redis keys (for Redis type)
                - vector_dimensions: Size of embedding vectors (for vector type)
                - graphiti_client: Graphiti client instance (for Graphiti type)
                - endpoint: Graphiti API endpoint URL (for Graphiti type)
                - api_key: Graphiti API key (for Graphiti type)
                - user_id: Optional user ID for user-specific memory
                
        Returns:
            Implementation of LongTermMemory
            
        Raises:
            ValueError: If the long-term memory type is not supported or requirements not met
        """
        memory_type = config.get("type", "redis_vector")
        user_id = config.get("user_id")
        
        if memory_type == "redis_vector":
            # Check for embedding service
            embedding_service = config.get("embedding_service")
            if embedding_service is None:
                if EmbeddingService is None:
                    raise ValueError("Embedding service is required for vector memory but not available")
                
                # Default embedding service - in real usage, this should be configured
                embedding_service = EmbeddingService()
            
            # Create Redis vector memory
            redis_url = config.get("redis_url", "redis://localhost:6379")
            index_name = config.get("index_name", "agent_memory")
            prefix = config.get("prefix", "agent:")
            vector_dimensions = config.get("vector_dimensions", 384)
            distance_metric = config.get("distance_metric", "COSINE")
            
            return RedisVectorMemory(
                embedding_service=embedding_service,
                redis_url=redis_url,
                index_name=index_name,
                prefix=prefix,
                vector_dimensions=vector_dimensions,
                distance_metric=distance_metric,
                user_id=user_id
            )
        
        elif memory_type == "graphiti":
            # Check for Graphiti client or credentials
            graphiti_client = config.get("graphiti_client")
            endpoint = config.get("endpoint")
            api_key = config.get("api_key")
            
            return GraphitiMemory(
                graphiti_client=graphiti_client,
                endpoint=endpoint,
                api_key=api_key,
                user_id=user_id
            )
        
        else:
            raise ValueError(f"Unsupported long-term memory type: {memory_type}")
    
    @staticmethod
    def create_memory_manager(
        context_config: Optional[Dict[str, Any]] = None,
        long_term_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> MemoryManager:
        """
        Create a complete memory manager with configured memory components.
        
        Args:
            context_config: Configuration for context memory (if None, uses in-memory)
            long_term_config: Configuration for long-term memory (if None, uses Redis vector)
            user_id: Optional user ID for user-specific memory
            
        Returns:
            Configured MemoryManager instance
        """
        # Use default configs if not provided
        if context_config is None:
            context_config = {"type": "in_memory"}
        
        if long_term_config is None:
            long_term_config = {"type": "redis_vector"}
        
        # Add user_id to configs if provided
        if user_id:
            long_term_config = {**long_term_config, "user_id": user_id}
        
        # Create memory components
        context_memory = MemoryFactory.create_context_memory(context_config)
        long_term_memory = MemoryFactory.create_long_term_memory(long_term_config)
        
        # Create and return memory manager
        return MemoryManager(
            context_memory=context_memory,
            long_term_memory=long_term_memory,
            user_id=user_id
        ) 