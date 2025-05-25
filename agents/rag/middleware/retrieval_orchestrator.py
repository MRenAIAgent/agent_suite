"""RetrievalOrchestrator middleware for the RAG system.

This module provides the orchestrator that coordinates retrieval from multiple
storage backends and combines the results.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from agents.rag.api.rag_retrieval import RagRetrieval, RetrievalResult
from agents.rag.middleware.storage_router import StorageType


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    
    @abstractmethod
    def determine_sources(self, query: str, metadata: Dict[str, Any]) -> Set[StorageType]:
        """
        Determine which storage sources to query based on the query and metadata.
        
        Args:
            query: The query string
            metadata: Additional query metadata
            
        Returns:
            Set of storage types to query
        """
        pass


class DefaultRetrievalStrategy(RetrievalStrategy):
    """Default strategy for determining retrieval sources."""
    
    def determine_sources(self, query: str, metadata: Dict[str, Any]) -> Set[StorageType]:
        """
        Determine sources based on query characteristics.
        
        Default behavior:
        - For questions about relationships or connections -> GRAPH
        - For semantic/meaning questions -> VECTOR
        - For exact lookups with specific keys -> KEY_VALUE
        - If unsure -> ALL
        """
        # Check for explicit source directive in metadata
        if "retrieval_source" in metadata:
            source = metadata["retrieval_source"]
            if isinstance(source, str):
                try:
                    return {StorageType(source)}
                except ValueError:
                    pass
            elif isinstance(source, list):
                try:
                    return {StorageType(s) for s in source}
                except ValueError:
                    pass
        
        sources = set()
        
        # Check for relationship keywords
        relationship_keywords = {"connection", "relationship", "linked", "connected", "between"}
        if any(keyword in query.lower() for keyword in relationship_keywords):
            sources.add(StorageType.GRAPH)
        
        # Check for key-value lookups
        kv_patterns = ["what is the", "value of", "setting for", "configuration"]
        if any(pattern in query.lower() for pattern in kv_patterns):
            sources.add(StorageType.KEY_VALUE)
        
        # By default, include vector search for semantic understanding
        sources.add(StorageType.VECTOR)
        
        return sources if sources else {StorageType.ALL}


class RetrievalOrchestrator:
    """
    Orchestrates retrieval from multiple sources and combines results.
    """
    
    def __init__(self):
        """Initialize the retrieval orchestrator."""
        self.strategies = []
        self.retrieval_handlers = {}
        # Register default strategy
        self.register_strategy(DefaultRetrievalStrategy())
    
    def register_strategy(self, strategy: RetrievalStrategy) -> None:
        """
        Register a new retrieval strategy.
        
        Args:
            strategy: The strategy to register
        """
        self.strategies.append(strategy)
    
    def register_retrieval_handler(self, storage_type: StorageType, 
                                 handler: RagRetrieval) -> None:
        """
        Register a retrieval handler for a specific storage type.
        
        Args:
            storage_type: The storage type this handler is for
            handler: The retrieval handler
        """
        self.retrieval_handlers[storage_type] = handler
    
    def determine_sources(self, query: str, metadata: Dict[str, Any] = None) -> Set[StorageType]:
        """
        Determine which sources to retrieve from based on the query.
        
        Args:
            query: The query string
            metadata: Optional query metadata
            
        Returns:
            Set of storage types to query
        """
        metadata = metadata or {}
        
        # Apply strategies to determine sources
        for strategy in self.strategies:
            sources = strategy.determine_sources(query, metadata)
            if sources:
                return sources
        
        # Default to all sources if no strategy matched
        return {StorageType.ALL}
    
    async def retrieve(self, query: str, top_k: int = 5, 
                     metadata: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Orchestrate retrieval from multiple sources.
        
        Args:
            query: The query string
            top_k: Maximum total results to return
            metadata: Optional query metadata
            
        Returns:
            Combined and ranked retrieval results
        """
        metadata = metadata or {}
        
        # Determine which sources to query
        sources = self.determine_sources(query, metadata)
        
        # If ALL is specified, query all available handlers
        if StorageType.ALL in sources:
            sources = set(self.retrieval_handlers.keys())
        
        # Query each source in parallel
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_source = {}
            
            for source in sources:
                if source in self.retrieval_handlers:
                    handler = self.retrieval_handlers[source]
                    future = executor.submit(
                        asyncio.run, 
                        handler.retrieve(query, top_k, metadata)
                    )
                    future_to_source[future] = source
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    source_results = future.result()
                    # Add source information to each result
                    for result in source_results:
                        result.source = source.value
                    results.extend(source_results)
                except Exception as e:
                    print(f"Error retrieving from {source}: {e}")
        
        # Combine and rank results
        return self._rank_and_combine(results, top_k)
    
    def _rank_and_combine(self, results: List[RetrievalResult], 
                         top_k: int) -> List[RetrievalResult]:
        """
        Rank and combine results from multiple sources.
        
        Args:
            results: List of retrieval results from all sources
            top_k: Maximum number of results to return
            
        Returns:
            Ranked and combined results
        """
        # Sort by score in descending order
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        
        # Deduplicate results based on item ID
        unique_results = []
        seen_ids = set()
        
        for result in sorted_results:
            item_id = getattr(result.item, 'id', None)
            if item_id is not None and item_id in seen_ids:
                continue
            if item_id is not None:
                seen_ids.add(item_id)
            unique_results.append(result)
        
        # Return top K results
        return unique_results[:top_k] 