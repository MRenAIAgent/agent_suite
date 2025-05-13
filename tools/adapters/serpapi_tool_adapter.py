"""
Adapter to convert original SerpAPI tools to use the enhanced tool system.

This module provides adapters for the original SerpAPI tools to make them
compatible with the enhanced tool registration system.
"""
import logging
from typing import Any, Optional

from tools.types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
)
from tools.base import EnhancedTool
from tools.registry import registry
from tools.serpapi_tool import SerpApiSearchTool

logger = logging.getLogger(__name__)


class EnhancedSerpApiSearchTool(EnhancedTool):
    """Enhanced tool for searching using SerpAPI."""
    
    query: str
    location: Optional[str] = None
    
    def __init__(self, **data):
        """Initialize with enhanced metadata."""
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.SEARCH, ToolCategory.WEB_BROWSING],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.SEARCH, ToolCapability.FETCH],
                applicable_roles=["researcher", "analyst", "assistant"],
                task_patterns=["search web", "find information", "google search", "web query"],
                examples=[
                    "Search for 'climate change impact' on Google",
                    "Find information about 'Python programming' in 'San Francisco'"
                ],
                is_cacheable=True,
                avg_execution_time_ms=2000,
                prerequisites=["SERPAPI_API_KEY environment variable must be set"],
                keywords=["search", "google", "web", "serpapi", "query"]
            )
        
        super().__init__(**data)
        
        # Initialize the original SerpAPI search tool
        self.serp_tool = SerpApiSearchTool()
    
    async def arun(self, query: str, location: Optional[str] = None) -> Any:
        """
        Search the web using SerpAPI.
        
        Args:
            query: Search query to send to SerpAPI
            location: Optional location for the search
            
        Returns:
            Search results from SerpAPI
        """
        return await self.serp_tool.arun(query, location)


def register_serpapi_tools():
    """Register all SerpAPI tools with the registry."""
    # Create and register the enhanced SerpAPI tool
    registry.register_tool(EnhancedSerpApiSearchTool())
    
    logger.info("Registered SerpAPI tools with the registry") 