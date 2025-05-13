"""
Test script for GitHub MCP client functionality.
"""
import asyncio
import logging
import os
import sys
from typing import Dict, Any, List

from agents.mcp_client.github_client import GitHubMCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_github_client():
    """Test GitHub MCP client functionality."""
    logger.info("Testing GitHub MCP client functionality...")
    
    try:
        # Create and connect to the client
        client = GitHubMCPClient()
        connected = await client.connect()
        
        if connected:
            logger.info("Successfully connected to GitHub MCP server")
            
            # Get authenticated user info
            user_info = await client.get_user_info()
            logger.info(f"Authenticated user: {user_info}")
            
            # Search for repositories
            repos = await client.search_repositories("language:python stars:>1000", limit=3)
            logger.info(f"Top Python repositories: {repos}")
            
            # Get repository info
            repo_info = await client.get_repository_info("modelcontextprotocol/python-sdk")
            logger.info(f"MCP Python SDK repository info: {repo_info}")
            
            # List branches
            branches = await client.list_repository_branches("modelcontextprotocol/python-sdk")
            logger.info(f"Branches: {branches}")
            
            # Get issues
            issues = await client.get_issues("modelcontextprotocol/python-sdk", limit=3)
            logger.info(f"Recent issues: {issues}")
            
            # Clean up
            await client.disconnect()
            logger.info("Successfully disconnected from GitHub MCP server")
        else:
            logger.error("Failed to connect to GitHub MCP server")
    except Exception as e:
        logger.error(f"Error during GitHub client test: {e}")


async def main():
    """Run GitHub MCP client tests."""
    logger.info("Starting GitHub MCP client tests...")
    
    # Check if MCP SDK is installed
    try:
        import mcp
        logger.info("MCP SDK is installed")
    except ImportError:
        logger.warning("MCP SDK is not installed. Install with 'pip install mcp'")
        logger.warning("Tests will be skipped.")
        return
    
    # Check if GitHub CLI MCP is installed
    try:
        import importlib.util
        spec = importlib.util.find_spec("mcp_github_cli")
        if spec is None:
            logger.warning("GitHub MCP CLI is not installed. Install with 'pip install mcp-github-cli'")
            logger.warning("Tests will be skipped.")
            return
        logger.info("GitHub MCP CLI is installed")
    except ImportError:
        logger.warning("GitHub MCP CLI is not installed. Install with 'pip install mcp-github-cli'")
        logger.warning("Tests will be skipped.")
        return
    
    # Run tests
    await test_github_client()
    
    logger.info("GitHub MCP client tests completed")


if __name__ == "__main__":
    asyncio.run(main()) 