"""
GitHub MCP client implementation.
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from agents.mcp_client.client import MCPServerConfig, MCPTransportType
from agents.mcp_client.mcp_sdk_client import MCPSDKClient

logger = logging.getLogger(__name__)


class GitHubMCPClient(MCPSDKClient):
    """
    Specialized MCP client for interacting with GitHub MCP servers.
    
    This client provides GitHub-specific functionality and helper methods
    for common GitHub operations.
    """
    
    def __init__(self, server_name: str = "github"):
        """
        Initialize the GitHub MCP client.
        
        Args:
            server_name: Name for the GitHub MCP server connection
        """
        # Create a GitHub server configuration
        server_config = MCPServerConfig(
            name=server_name,
            transport_type=MCPTransportType.STDIO,
            command="python",
            args=["-m", "mcp_github_cli"]
        )
        
        super().__init__(server_config)
        
    async def get_user_info(self) -> Dict[str, Any]:
        """
        Get information about the authenticated GitHub user.
        
        Returns:
            Dict containing user information
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to GitHub MCP server")
            
        result = await self.call_tool("gh_get_me", {})
        return self._parse_result(result)
    
    async def search_repositories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for GitHub repositories.
        
        Args:
            query: Search query (e.g., "language:python stars:>1000")
            limit: Maximum number of results to return
            
        Returns:
            List of repository information
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to GitHub MCP server")
            
        result = await self.call_tool("gh_rest_search_repos", {
            "query": query,
            "limit": limit
        })
        return self._parse_result(result)
    
    async def get_repository_info(self, owner_repo: str) -> Dict[str, Any]:
        """
        Get detailed information about a GitHub repository.
        
        Args:
            owner_repo: Repository in the format "owner/repo"
            
        Returns:
            Dict containing repository information
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to GitHub MCP server")
            
        result = await self.call_tool("gh_graphql_repo_info", {
            "owner_repo": owner_repo
        })
        return self._parse_result(result)
    
    async def get_repository_contents(self, owner_repo: str, path: str = "", ref: str = "") -> List[Dict[str, Any]]:
        """
        Get contents of a file or directory in a GitHub repository.
        
        Args:
            owner_repo: Repository in the format "owner/repo"
            path: Path to the file or directory (default: root)
            ref: Git reference (branch, tag, commit)
            
        Returns:
            List of file/directory information
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to GitHub MCP server")
            
        args = {"owner_repo": owner_repo}
        if path:
            args["path"] = path
        if ref:
            args["ref"] = ref
            
        result = await self.call_tool("gh_rest_repo_contents", args)
        return self._parse_result(result)
    
    async def list_repository_branches(self, owner_repo: str) -> List[Dict[str, Any]]:
        """
        List branches in a GitHub repository.
        
        Args:
            owner_repo: Repository in the format "owner/repo"
            
        Returns:
            List of branch information
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to GitHub MCP server")
            
        result = await self.call_tool("gh_rest_branches", {
            "owner_repo": owner_repo
        })
        return self._parse_result(result)
    
    async def create_issue(self, owner_repo: str, title: str, body: str, 
                         labels: List[str] = None) -> Dict[str, Any]:
        """
        Create an issue in a GitHub repository.
        
        Args:
            owner_repo: Repository in the format "owner/repo"
            title: Issue title
            body: Issue body
            labels: List of labels to apply
            
        Returns:
            Dict containing the created issue
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to GitHub MCP server")
            
        args = {
            "owner_repo": owner_repo,
            "title": title,
            "body": body
        }
        
        if labels:
            args["labels"] = labels
            
        result = await self.call_tool("gh_rest_create_issue", args)
        return self._parse_result(result)
    
    async def create_pull_request(self, owner_repo: str, title: str, body: str,
                                head: str, base: str, draft: bool = False) -> Dict[str, Any]:
        """
        Create a pull request in a GitHub repository.
        
        Args:
            owner_repo: Repository in the format "owner/repo"
            title: Pull request title
            body: Pull request body
            head: Branch containing changes
            base: Branch to merge into
            draft: Whether to create as a draft PR
            
        Returns:
            Dict containing the created pull request
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to GitHub MCP server")
            
        result = await self.call_tool("gh_rest_create_pr", {
            "owner_repo": owner_repo,
            "title": title,
            "body": body,
            "head": head,
            "base": base,
            "draft": draft
        })
        return self._parse_result(result)
    
    async def get_pull_requests(self, owner_repo: str, limit: int = 10, 
                              state: str = "OPEN") -> List[Dict[str, Any]]:
        """
        Get pull requests from a GitHub repository.
        
        Args:
            owner_repo: Repository in the format "owner/repo"
            limit: Maximum number of PRs to retrieve
            state: PR state (OPEN, CLOSED, MERGED)
            
        Returns:
            List of pull request information
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to GitHub MCP server")
            
        result = await self.call_tool("gh_graphql_pull_requests", {
            "owner_repo": owner_repo,
            "limit": limit,
            "state": state
        })
        return self._parse_result(result)
    
    async def get_issues(self, owner_repo: str, limit: int = 10, state: str = "OPEN",
                       labels: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get issues from a GitHub repository.
        
        Args:
            owner_repo: Repository in the format "owner/repo"
            limit: Maximum number of issues to retrieve
            state: Issue state (OPEN, CLOSED)
            labels: Filter by labels
            
        Returns:
            List of issue information
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to GitHub MCP server")
            
        args = {
            "owner_repo": owner_repo,
            "limit": limit,
            "state": state
        }
        
        if labels:
            args["labels"] = labels
            
        result = await self.call_tool("gh_graphql_issues", args)
        return self._parse_result(result)
    
    def _parse_result(self, result: Any) -> Any:
        """Parse the result from a tool call."""
        if hasattr(result, "content") and hasattr(result.content, "text"):
            # Try to parse JSON
            try:
                return json.loads(result.content.text)
            except json.JSONDecodeError:
                return result.content.text
        elif hasattr(result, "text"):
            # Try to parse JSON
            try:
                return json.loads(result.text)
            except json.JSONDecodeError:
                return result.text
        elif isinstance(result, str):
            # Try to parse JSON
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
        else:
            return result 