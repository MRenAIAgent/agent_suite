import os
from pydantic import Field
from serpapi import SerpApiClient
from tools.tool import Tool
from typing import Any, Optional


class SerpApiSearchTool(Tool):
    """Search tool using SerpAPI.
    This tool allows sending search queries to the SerpAPI and returning search results.
    """
    query: str = Field(
        ..., description="Mandatory search query you want to use to Google search."
    )
    location: Optional[str] = Field(
        default=None,
        description="Location you want the search to be performed in."
    )

    async def arun(self, query: str, location: Optional[str] = None) -> Any:
        """Asynchronously execute search query using SerpAPI.
        
        Args:
            query: Search query to send to SerpAPI
            location: Optional location for the search
            
        Returns:
            dict: Search results from SerpAPI
            
        Raises:
            Exception: If API call fails or search errors occur
        """
        api_key = os.getenv("SERPAPI_API_KEY")
        search = SerpApiClient({
            "q": query, 
            "location": location, 
            "engine": "google",
            "api_key": api_key
            })
        result = search.get_json()
        return result
    
    def run(self, query: str, location: Optional[str] = None) -> Any:
        """Synchronously execute search query using SerpAPI.
        
        Args:
            query: Search query to send to SerpAPI
            location: Optional location for the search
            
        Returns:
            dict: Search results from SerpAPI
            
        Raises:
            Exception: If API call fails or search errors occur
        """
        api_key = os.getenv("SERPAPI_API_KEY")
        search = SerpApiClient({
            "q": query, 
            "location": location, 
            "engine": "google",
            "api_key": api_key
            })
        result = search.get_json()
        return result
