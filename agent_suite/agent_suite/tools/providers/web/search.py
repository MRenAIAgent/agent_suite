import json
import os
import asyncio
import logging
from typing import Dict, List, Optional, Union

import aiohttp
from pydantic import BaseModel, Field

from log import log_manager
from tools.tool import Tool
from tools.tool_types import ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
from tools.base import EnhancedTool

# Configure logging
logger = logging.getLogger(__name__)


class SerperSearchTool(Tool):
    """
    Search Google using the Serper.dev API.
    
    This tool performs a Google search using the Serper.dev API and returns
    the search results in a structured format.
    """
    
    query: str = Field(
        ...,
        description="The search query to send to Google"
    )
    
    num_results: int = Field(
        5,
        description="The number of results to return"
    )
    
    include_snippets: bool = Field(
        True,
        description="Whether to include snippets in the results"
    )
    
    include_images: bool = Field(
        False,
        description="Whether to include image results"
    )
    
    async def arun(self, **kwargs) -> str:
        """
        Execute the Google search asynchronously.
        
        Args:
            **kwargs: Additional parameters for the search
            
        Returns:
            String with formatted search results
        """
        # Merge passed kwargs with current attributes
        query = kwargs.get("query", self.query)
        num_results = kwargs.get("num_results", self.num_results)
        include_snippets = kwargs.get("include_snippets", self.include_snippets)
        include_images = kwargs.get("include_images", self.include_images)
        
        try:
            # Get API key from environment variable
            api_key = os.environ.get("SERPER_API_KEY")
            if not api_key:
                return "Error: SERPER_API_KEY environment variable not set"
            
            # Call Google Search API
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }
            data = {
                "q": query,
                "num": num_results,
                "includeSnippets": include_snippets,
                "includeImages": include_images
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return f"Error {response.status}: {error_text}"
                    
                    # Parse response
                    result_json = await response.json()
                    
                    # Format results as simple text
                    return self._format_results(result_json, num_results)
                    
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return f"Error performing search: {e}"
    
    def run(self, **kwargs) -> str:
        """
        Execute the Google search synchronously.
        
        Args:
            **kwargs: Additional parameters for the search
            
        Returns:
            String with formatted search results
        """
        return asyncio.run(self.arun(**kwargs))
    
    def _format_results(self, result_json: Dict, limit: int) -> str:
        """Format search results as a readable string."""
        result = f"Search results for '{result_json.get('searchParameters', {}).get('q', 'unknown query')}':\n\n"
        
        # Add organic results
        organic = result_json.get("organic", [])
        for i, item in enumerate(organic[:limit]):
            title = item.get("title", "No title")
            link = item.get("link", "No link")
            snippet = item.get("snippet", "No description available")
            
            result += f"{i+1}. {title}\n   {link}\n   {snippet}\n\n"
        
        return result.strip()
    
    def to_enhanced_tool(self):
        """Convert to an enhanced tool with rich metadata."""
        # Create a basic wrapper using the parent method
        enhanced_tool = super().to_enhanced_tool()
        
        # Update with more specific metadata
        enhanced_tool.metadata = ToolMetadata(
            name="google_search",
            display_name="Google Search",
            description="Search Google and get the top results",
            examples=[
                "search for the latest news on climate change",
                "find information about the COVID-19 vaccine",
            ],
            categories=[ToolCategory.SEARCH],
            domains=[ToolDomain.GENERAL],
            capabilities=[
                ToolCapability.SEARCH,
                ToolCapability.READ
            ],
            api_dependencies=["SerpAPI"],
            source=ToolSource.CUSTOM,
            expected_response_time_ms=2500,
            is_cacheable=True
        )
        
        return enhanced_tool