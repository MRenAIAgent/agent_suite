import json
import os
import requests

from log import log_manager
from pydantic import Field
from tools.tool import Tool
from typing import Any, Optional



class SerperSearchTool(Tool):
    """Search tool using SerpAPI.
    This tool allows sending search queries to the SerpAPI and returning search results.
    """
    query: str = Field(
        ..., description="Mandatory search query you want to use to Google search."
    )

    async def arun(self, query: str) -> Any:
        """Asynchronously execute search query using SerpAPI.
        
        Args:
            query: Search query to send to SerpAPI
            location: Optional location for the search
            
        Returns:
            dict: Search results from SerpAPI
            
        Raises:
            Exception: If API call fails or search errors occur
        """
        api_key = os.getenv("SERPER_API_KEY")
        result = self._get_serper_results(query, api_key)
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        
        log_manager.log_debug(f"{YELLOW}Query sent to SerpAPI: {query}. \nSerpAPI result: {result}{RESET}")
        return json.loads(result)["organic"]
    
    def run(self, query: str) -> Any:
        """Synchronously execute search query using SerpAPI.
        
        Args:
            query: Search query to send to SerpAPI
            location: Optional location for the search
            
        Returns:
            dict: Search results from SerpAPI
            
        Raises:
            Exception: If API call fails or search errors occur
        """
        api_key = os.getenv("SERPER_API_KEY")
        return self._get_serper_results(query, api_key)
    
    def _get_serper_results(self, query: str, api_key: str) -> Any:
        """Get search results from SerpAPI.
        
        Args:
            query: Search query to send to SerpAPI
            location: Optional location for the search

        Returns:
            dict: Search results from SerpAPI
            
        Raises:
            Exception: If API call fails or search errors occur
        """

        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query
        })
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.text