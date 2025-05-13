"""
MCP Tool Registration Example

This script demonstrates how to register various MCP tools with the enhanced tool system:
1. Individual MCP tool registration with detailed metadata
2. Bulk MCP tool registration from different providers
3. Using the registered MCP tools in a tool selection scenario
"""
import asyncio
import os
import sys
from typing import Dict, Any, List

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.registry import registry
from tools.tool_selector import ToolSelector, SelectionMethod
from tools.mcp_wrapper import (
    register_mcp_tool, 
    register_mcp_tools,
    MCPToolWrapper
)
from tools.tool_types import ToolCategory, ToolCapability, ToolDomain


# Example MCP tool data for demonstration
MOCK_GITHUB_TOOLS = {
    "github_search_repositories": {
        "description": "Search for GitHub repositories",
        "parameters": {
            "properties": {
                "query": {
                    "description": "Search query",
                    "type": "string"
                },
                "page": {
                    "description": "Page number (default 1)",
                    "type": "number",
                    "default": 1
                },
                "per_page": {
                    "description": "Results per page (default 30, max 100)",
                    "type": "number",
                    "default": 30
                }
            },
            "required": ["query"]
        },
        "categories": ["search", "developer-tools"],
        "capabilities": ["search", "fetch"],
        "domains": ["programming", "development"],
        "keywords": ["github", "repositories", "code", "search", "developer"]
    },
    "github_search_code": {
        "description": "Search for code across GitHub repositories",
        "parameters": {
            "properties": {
                "q": {
                    "description": "Search query using GitHub code search syntax",
                    "type": "string"
                },
                "sort": {
                    "description": "Sort field ('indexed' only)",
                    "type": "string",
                    "enum": ["indexed"]
                },
                "order": {
                    "description": "Sort order",
                    "type": "string",
                    "enum": ["asc", "desc"]
                },
                "page": {
                    "description": "Page number (default 1)",
                    "type": "number",
                    "default": 1
                },
                "per_page": {
                    "description": "Results per page (default 30, max 100)",
                    "type": "number",
                    "default": 30
                }
            },
            "required": ["q"]
        },
        "categories": ["search", "developer-tools"],
        "capabilities": ["search", "fetch"],
        "domains": ["programming"],
        "keywords": ["github", "code", "search", "developer", "source code"]
    }
}

MOCK_ZAPIER_TOOLS = {
    "zapier_create_gmail_draft": {
        "description": "Create a new Gmail draft message",
        "parameters": {
            "properties": {
                "to": {
                    "description": "Recipient email address",
                    "type": "string"
                },
                "subject": {
                    "description": "Email subject",
                    "type": "string"
                },
                "body": {
                    "description": "Email body content",
                    "type": "string"
                },
                "cc": {
                    "description": "CC recipients",
                    "type": "string"
                },
                "bcc": {
                    "description": "BCC recipients",
                    "type": "string"
                }
            },
            "required": ["to", "subject", "body"]
        },
        "categories": ["communication", "email"],
        "capabilities": ["write", "communicate"],
        "domains": ["business", "general"],
        "keywords": ["email", "gmail", "draft", "compose", "communication"]
    },
    "zapier_search_trello_cards": {
        "description": "Search for cards in Trello boards",
        "parameters": {
            "properties": {
                "query": {
                    "description": "Search query",
                    "type": "string"
                },
                "board_id": {
                    "description": "Board ID to search in (optional)",
                    "type": "string"
                }
            },
            "required": ["query"]
        },
        "categories": ["search", "productivity"],
        "capabilities": ["search", "fetch"],
        "domains": ["business", "project-management"],
        "keywords": ["trello", "cards", "task", "project", "management", "search"]
    }
}


class MCPRegistrationExample:
    """Example class demonstrating MCP tool registration."""
    
    def __init__(self):
        """Initialize the example."""
        # Clear the registry to start fresh
        registry.clear()
        self.tool_selector = ToolSelector(registry)
        
        # Track registered tools
        self.github_tools = {}
        self.zapier_tools = {}
    
    def register_individual_tool(self):
        """Register an individual MCP tool with detailed metadata."""
        print("Registering individual GitHub repository search tool...")
        
        # Get the tool data
        tool_data = MOCK_GITHUB_TOOLS["github_search_repositories"]
        
        # Create a mock tool instance (in real usage, this would be the actual tool implementation)
        mock_tool_instance = type("GitHubSearchRepositoriesTool", (), {
            "execute": lambda **kwargs: {"results": ["Repository 1", "Repository 2"]}
        })()
        
        # Register the tool
        wrapped_tool = register_mcp_tool(
            mcp_tool_name="github_search_repositories",
            mcp_namespace="github",
            mcp_tool_data=tool_data,
            mcp_tool_instance=mock_tool_instance,
            registry_namespace="github"
        )
        
        # Store the reference
        self.github_tools["search_repositories"] = wrapped_tool
        
        print(f"Successfully registered: {wrapped_tool.metadata.name}")
        print(f"Categories: {[cat.value for cat in wrapped_tool.metadata.categories]}")
        print(f"Capabilities: {[cap.value for cap in wrapped_tool.metadata.capabilities]}")
        print(f"Domains: {[dom.value for dom in wrapped_tool.metadata.domains]}")
        print()
    
    def register_tool_collection(self):
        """Register multiple MCP tools at once."""
        print("Registering GitHub code search and Zapier tools...")
        
        # Register GitHub code search
        github_tools = register_mcp_tools(
            mcp_tools={"github_search_code": MOCK_GITHUB_TOOLS["github_search_code"]},
            registry_namespace_mapping={"github_search_code": "github"}
        )
        self.github_tools.update(github_tools)
        
        # Register Zapier tools
        zapier_tools = register_mcp_tools(
            mcp_tools=MOCK_ZAPIER_TOOLS,
            registry_namespace_mapping={
                "zapier_create_gmail_draft": "zapier",
                "zapier_search_trello_cards": "zapier"
            }
        )
        self.zapier_tools = zapier_tools
        
        # Print all registered tools
        print(f"Successfully registered {len(github_tools) + len(zapier_tools)} additional tools:")
        for tool_name, tool in {**github_tools, **zapier_tools}.items():
            print(f" - {tool.metadata.name}: {tool.metadata.description}")
        print()
    
    def demonstrate_tool_selection(self):
        """Demonstrate tool selection with the registered MCP tools."""
        print("Demonstrating tool selection with registered MCP tools...")
        
        # Example user queries
        queries = [
            "I need to find some GitHub repositories about machine learning",
            "Can you search for code examples of binary search on GitHub?",
            "I want to create a draft email to my boss about the project status",
            "Help me find task cards in my Trello board about the website redesign"
        ]
        
        # Test each query
        for query in queries:
            print(f"\nQuery: \"{query}\"")
            selected_tools = self.tool_selector.select_tools(
                query=query,
                method=SelectionMethod.HYBRID,
                max_tools=2
            )
            
            print("Selected tools:")
            for i, tool in enumerate(selected_tools):
                print(f"  {i+1}. {tool.metadata.name} ({tool.metadata.display_name})")
                print(f"     Source: {tool.source}")
                print(f"     Categories: {[cat.value for cat in tool.metadata.categories]}")
                print(f"     Capabilities: {[cap.value for cap in tool.metadata.capabilities][:3]}...")
            print()


async def main():
    """Run the MCP registration example."""
    example = MCPRegistrationExample()
    
    # Register tools
    example.register_individual_tool()
    example.register_tool_collection()
    
    # Demonstrate tool selection
    example.demonstrate_tool_selection()


if __name__ == "__main__":
    asyncio.run(main()) 