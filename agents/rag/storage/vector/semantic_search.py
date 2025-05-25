"""Semantic search utilities for vector storage.

This module provides enhanced semantic search capabilities for the vector storage
adaptors in the RAG system, including query expansion, reranking, and hybrid search.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio

from agents.rag.models.document import Document, DocumentChunk
from agents.rag.api.rag_retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class SemanticSearchLayer:
    """
    Enhanced semantic search layer that sits on top of vector storage.
    
    This layer provides advanced semantic search capabilities:
    1. Query expansion - enrich the original query 
    2. Hybrid search - combine vector search with keyword/filter search
    3. Reranking - improve relevance by reordering results
    """
    
    def __init__(self, 
                 query_expansion_fn: Optional[Callable[[str], List[str]]] = None,
                 reranker_fn: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
                 hybrid_search_fn: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None):
        """
        Initialize the semantic search layer.
        
        Args:
            query_expansion_fn: Optional function to expand queries
            reranker_fn: Optional function to rerank results
            hybrid_search_fn: Optional function for hybrid search
        """
        self.query_expansion_fn = query_expansion_fn
        self.reranker_fn = reranker_fn
        self.hybrid_search_fn = hybrid_search_fn
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple query variations.
        
        Args:
            query: The original query string
            
        Returns:
            List of expanded queries
        """
        if self.query_expansion_fn:
            try:
                return self.query_expansion_fn(query)
            except Exception as e:
                logger.error(f"Error in query expansion: {e}")
        
        # Default: return original query
        return [query]
    
    async def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search results based on semantic relevance.
        
        Args:
            query: The original query
            results: List of search results
            
        Returns:
            Reranked list of results
        """
        if self.reranker_fn and results:
            try:
                return self.reranker_fn(query, results)
            except Exception as e:
                logger.error(f"Error in reranking: {e}")
        
        # Default: return original results
        return results
    
    async def apply_hybrid_search(self, query: str, 
                                vector_results: List[Dict[str, Any]], 
                                filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Apply hybrid search combining vector search with keyword/filter matches.
        
        Args:
            query: The original query
            vector_results: Results from vector search
            filter_criteria: Optional filter criteria
            
        Returns:
            Enhanced search results
        """
        if self.hybrid_search_fn and vector_results:
            try:
                return self.hybrid_search_fn(query, vector_results, filter_criteria)
            except Exception as e:
                logger.error(f"Error in hybrid search: {e}")
        
        # Default: return vector results
        return vector_results
    
    async def process_query(self, 
                          base_search_fn: Callable, 
                          query: str, 
                          top_k: int = 5,
                          filter_criteria: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Process a query through the full semantic search pipeline.
        
        Args:
            base_search_fn: The base vector search function
            query: The query string
            top_k: Maximum number of results to return
            filter_criteria: Optional filters to apply
            
        Returns:
            List of retrieval results
        """
        # 1. Expand the query
        expanded_queries = await self.expand_query(query)
        
        # 2. Execute vector search for each expanded query
        all_results = []
        for expanded_query in expanded_queries:
            try:
                # Call the base search function with the expanded query
                results = await base_search_fn(expanded_query, top_k, filter_criteria)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error in base search with expanded query '{expanded_query}': {e}")
        
        # 3. Apply hybrid search if available
        if all_results and self.hybrid_search_fn:
            all_results = await self.apply_hybrid_search(query, all_results, filter_criteria)
        
        # 4. Rerank results if available
        if all_results and self.reranker_fn:
            all_results = await self.rerank_results(query, all_results)
        
        # 5. Convert to retrieval results and limit to top_k
        retrieval_results = []
        seen_ids = set()
        
        for result in all_results:
            item = result.get("item")
            if not item:
                continue
                
            item_id = getattr(item, "id", None)
            if item_id and item_id in seen_ids:
                continue
                
            if item_id:
                seen_ids.add(item_id)
                
            score = result.get("score", 1.0)
            retrieval_results.append(RetrievalResult(item=item, score=score))
        
        # Return top_k results
        return retrieval_results[:top_k]


# Common query expansion implementations
def basic_query_expansion(query: str) -> List[str]:
    """
    Basic query expansion that creates variations of the original query.
    
    Args:
        query: The original query
        
    Returns:
        List of expanded queries
    """
    expanded = [query]
    
    # Add a "what is" variation
    if not query.lower().startswith("what is"):
        expanded.append(f"what is {query}")
    
    # Add a "how to" variation
    if not query.lower().startswith("how to"):
        expanded.append(f"how to {query}")
    
    return expanded


# Common reranking implementations
def relevance_reranker(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple relevance reranker that boosts results containing query terms.
    
    Args:
        query: The original query
        results: List of search results
        
    Returns:
        Reranked list of results
    """
    query_terms = query.lower().split()
    
    # Function to boost score based on term presence
    def boost_score(result):
        item = result.get("item")
        content = ""
        
        if isinstance(item, Document):
            content = item.content.lower()
        elif isinstance(item, DocumentChunk):
            content = item.content.lower()
        elif isinstance(item, dict) and "content" in item:
            content = item["content"].lower()
        
        # Count query terms in content
        term_matches = sum(1 for term in query_terms if term in content)
        
        # Boost score based on matches
        base_score = result.get("score", 0.5)
        boost = 0.1 * term_matches if term_matches > 0 else 0
        
        # Update the score
        result["score"] = min(1.0, base_score + boost)
        return result
    
    # Apply boost to each result and sort
    boosted_results = [boost_score(result) for result in results]
    return sorted(boosted_results, key=lambda x: x.get("score", 0), reverse=True) 