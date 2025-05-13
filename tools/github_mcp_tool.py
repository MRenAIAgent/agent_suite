"""
GitHub MCP tool for agents to interact with GitHub.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from tools.tool import Tool
from agents.mcp_client.github_client import GitHubMCPClient

logger = logging.getLogger(__name__)


class GitHubSearchRepos(Tool):
    """Tool for searching GitHub repositories."""
    
    query: str = Field(
        ...,
        description="Search query (e.g., 'language:python stars:>1000')"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of results to return"
    )
    
    def __init__(self):
        super().__init__()
        self.client = None
    
    async def arun(self, query: str, limit: int = 5) -> Any:
        """
        Search for GitHub repositories.
        
        Args:
            query: Search query (e.g., 'language:python stars:>1000')
            limit: Maximum number of results to return
            
        Returns:
            List of repository information
        """
        try:
            client = await self._get_client()
            repos = await client.search_repositories(query, limit)
            
            # Format the results to be more readable
            formatted_repos = []
            for repo in repos:
                formatted_repos.append({
                    "name": repo.get("name"),
                    "full_name": repo.get("full_name"),
                    "description": repo.get("description"),
                    "stars": repo.get("stargazers_count"),
                    "url": repo.get("html_url"),
                    "language": repo.get("language")
                })
                
            return json.dumps(formatted_repos, indent=2)
        except Exception as e:
            logger.error(f"Error searching GitHub repositories: {e}")
            return f"Error: {str(e)}"
    
    def run(self, query: str, limit: int = 5) -> Any:
        """
        Synchronously search for GitHub repositories.
        
        Args:
            query: Search query (e.g., 'language:python stars:>1000')
            limit: Maximum number of results to return
            
        Returns:
            List of repository information
        """
        return asyncio.run(self.arun(query, limit))
    
    async def _get_client(self) -> GitHubMCPClient:
        """Get or create a GitHub MCP client."""
        if self.client is None or not self.client._initialized:
            self.client = GitHubMCPClient()
            await self.client.connect()
        return self.client


class GitHubRepoInfo(Tool):
    """Tool for getting information about a GitHub repository."""
    
    owner_repo: str = Field(
        ...,
        description="Repository in the format 'owner/repo' (e.g., 'modelcontextprotocol/python-sdk')"
    )
    
    def __init__(self):
        super().__init__()
        self.client = None
    
    async def arun(self, owner_repo: str) -> Any:
        """
        Get information about a GitHub repository.
        
        Args:
            owner_repo: Repository in the format 'owner/repo' (e.g., 'modelcontextprotocol/python-sdk')
            
        Returns:
            Repository information
        """
        try:
            client = await self._get_client()
            repo_info = await client.get_repository_info(owner_repo)
            return json.dumps(repo_info, indent=2)
        except Exception as e:
            logger.error(f"Error getting GitHub repository info: {e}")
            return f"Error: {str(e)}"
    
    def run(self, owner_repo: str) -> Any:
        """
        Synchronously get information about a GitHub repository.
        
        Args:
            owner_repo: Repository in the format 'owner/repo' (e.g., 'modelcontextprotocol/python-sdk')
            
        Returns:
            Repository information
        """
        return asyncio.run(self.arun(owner_repo))
    
    async def _get_client(self) -> GitHubMCPClient:
        """Get or create a GitHub MCP client."""
        if self.client is None or not self.client._initialized:
            self.client = GitHubMCPClient()
            await self.client.connect()
        return self.client


class GitHubIssues(Tool):
    """Tool for getting issues from a GitHub repository."""
    
    owner_repo: str = Field(
        ...,
        description="Repository in the format 'owner/repo' (e.g., 'modelcontextprotocol/python-sdk')"
    )
    state: str = Field(
        default="OPEN",
        description="Issue state (OPEN or CLOSED)"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of issues to return"
    )
    
    def __init__(self):
        super().__init__()
        self.client = None
    
    async def arun(self, owner_repo: str, state: str = "OPEN", limit: int = 5) -> Any:
        """
        Get issues from a GitHub repository.
        
        Args:
            owner_repo: Repository in the format 'owner/repo' (e.g., 'modelcontextprotocol/python-sdk')
            state: Issue state (OPEN or CLOSED)
            limit: Maximum number of issues to return
            
        Returns:
            List of issues
        """
        try:
            client = await self._get_client()
            issues = await client.get_issues(owner_repo, limit=limit, state=state)
            return json.dumps(issues, indent=2)
        except Exception as e:
            logger.error(f"Error getting GitHub issues: {e}")
            return f"Error: {str(e)}"
    
    def run(self, owner_repo: str, state: str = "OPEN", limit: int = 5) -> Any:
        """
        Synchronously get issues from a GitHub repository.
        
        Args:
            owner_repo: Repository in the format 'owner/repo' (e.g., 'modelcontextprotocol/python-sdk')
            state: Issue state (OPEN or CLOSED)
            limit: Maximum number of issues to return
            
        Returns:
            List of issues
        """
        return asyncio.run(self.arun(owner_repo, state, limit))
    
    async def _get_client(self) -> GitHubMCPClient:
        """Get or create a GitHub MCP client."""
        if self.client is None or not self.client._initialized:
            self.client = GitHubMCPClient()
            await self.client.connect()
        return self.client 