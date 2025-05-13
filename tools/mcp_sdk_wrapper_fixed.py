"""
MCP SDK Wrapper (Fixed)

This module connects the MCP SDK client with the enhanced tool system,
allowing tools from an MCP server to be registered and used in agents.
This version includes a fixed implementation to address Pydantic compatibility issues.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set

from tools.mcp_sdk_client import MCPSDKClient
from tools.registry import registry
from tools.tool_types import (
    ToolCategory, 
    ToolCapability, 
    ToolDomain,
    ToolSource,
    ToolMetadata
)
from tools.base import EnhancedTool

# Setup logging
logger = logging.getLogger(__name__)


class MCPSDKToolInstance:
    """Wrapper around an MCP SDK tool for execution."""
    
    def __init__(self, client: MCPSDKClient, tool_name: str):
        """Initialize the MCP SDK tool instance.
        
        Args:
            client: The MCP SDK client
            tool_name: The name of the tool
        """
        self.client = client
        self.tool_name = tool_name
    
    async def arun(self, **kwargs) -> Any:
        """Execute the tool asynchronously.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            Results from the tool execution
        """
        try:
            result = await self.client.execute_tool(self.tool_name, kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.tool_name}: {e}")
            return {"error": str(e)}
    
    def run(self, **kwargs) -> Any:
        """Execute the tool synchronously.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            Results from the tool execution
        """
        return asyncio.run(self.arun(**kwargs))


class MCPSDKToolWrapper(EnhancedTool):
    """Wrapper for MCP SDK tools to be used with the enhanced tool system."""
    
    def __init__(
        self,
        mcp_tool_name: str,
        mcp_namespace: str,
        mcp_tool_data: Dict[str, Any],
        mcp_tool_instance: Any,
        metadata: Optional[ToolMetadata] = None
    ):
        """Initialize the MCP SDK tool wrapper.
        
        Args:
            mcp_tool_name: The name of the MCP tool
            mcp_namespace: The namespace of the MCP tool
            mcp_tool_data: The tool data from the MCP server
            mcp_tool_instance: The MCPSDKToolInstance
            metadata: Optional tool metadata (will be created if not provided)
        """
        # Store properties as instance variables, not using pydantic
        self.mcp_tool_name = mcp_tool_name
        self.mcp_namespace = mcp_namespace
        self.mcp_tool_data = mcp_tool_data
        self._mcp_tool_instance = mcp_tool_instance
        
        # Create metadata if not provided
        if metadata is None:
            metadata = self._create_metadata()
        
        # Initialize the parent class with the metadata
        super().__init__()
        self.metadata = metadata
    
    def _create_metadata(self) -> ToolMetadata:
        """Create tool metadata from MCP tool data.
        
        Returns:
            Tool metadata instance
        """
        # Extract data from the MCP tool
        description = self.mcp_tool_data.get("description", "")
        display_name = self.mcp_tool_data.get("display_name", self.mcp_tool_name)
        
        # Create the metadata
        return ToolMetadata(
            name=self.mcp_tool_name,
            display_name=display_name,
            description=description,
            categories=self._extract_categories(),
            capabilities=self._extract_capabilities(),
            domains=self._extract_domains(),
            source=ToolSource.MCP,
            keywords=self._extract_keywords(),
            examples=[],
        )
    
    def _extract_categories(self) -> List[ToolCategory]:
        """Extract categories from the MCP tool data."""
        name = self.mcp_tool_name.lower()
        desc = self.mcp_tool_data.get("description", "").lower()
        
        categories = []
        
        if any(x in name or x in desc for x in ["search", "find", "lookup"]):
            categories.append(ToolCategory.SEARCH)
        
        if any(x in name or x in desc for x in ["calculate", "compute", "math"]):
            categories.append(ToolCategory.UTILITY)
        
        if any(x in name or x in desc for x in ["weather", "forecast", "temperature"]):
            categories.append(ToolCategory.UTILITY)
        
        if any(x in name or x in desc for x in ["stock", "price", "market", "financial"]):
            categories.append(ToolCategory.FINANCE)
        
        # Default to utility if nothing else matches
        if not categories:
            categories.append(ToolCategory.UTILITY)
            
        return categories
    
    def _extract_capabilities(self) -> List[ToolCapability]:
        """Extract capabilities from the MCP tool data."""
        name = self.mcp_tool_name.lower()
        desc = self.mcp_tool_data.get("description", "").lower()
        
        capabilities = []
        
        if any(x in name or x in desc for x in ["search", "find", "lookup"]):
            capabilities.append(ToolCapability.SEARCH)
        
        if any(x in name or x in desc for x in ["calculate", "compute", "math"]):
            capabilities.append(ToolCapability.COMPUTE)
        
        if any(x in name or x in desc for x in ["get", "retrieve", "fetch"]):
            capabilities.append(ToolCapability.FETCH)
        
        # Default to read if nothing else matches
        if not capabilities:
            capabilities.append(ToolCapability.READ)
            
        return capabilities
    
    def _extract_domains(self) -> List[ToolDomain]:
        """Extract domains from the MCP tool data."""
        name = self.mcp_tool_name.lower()
        desc = self.mcp_tool_data.get("description", "").lower()
        
        domains = []
        
        if any(x in name or x in desc for x in ["weather", "forecast", "temperature"]):
            domains.append(ToolDomain.GENERAL)
        
        if any(x in name or x in desc for x in ["stock", "price", "market", "financial"]):
            domains.append(ToolDomain.FINANCE)
        
        if any(x in name or x in desc for x in ["code", "programming", "developer"]):
            domains.append(ToolDomain.PROGRAMMING)
        
        # Default to general if nothing else matches
        if not domains:
            domains.append(ToolDomain.GENERAL)
            
        return domains
    
    def _extract_keywords(self) -> List[str]:
        """Extract keywords from the MCP tool data."""
        name = self.mcp_tool_name.lower()
        desc = self.mcp_tool_data.get("description", "").lower()
        
        # Split name and description into words
        name_parts = name.replace("_", " ").split()
        desc_words = [word for word in desc.split() if len(word) > 3]
        
        # Combine all potential keywords
        all_keywords = name_parts + desc_words
        
        # Remove duplicates
        keywords = list(set(all_keywords))
        
        return keywords
    
    async def arun(self, **kwargs) -> Any:
        """Run the MCP tool asynchronously.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            Results from the tool execution
        """
        if self._mcp_tool_instance is None:
            return {"error": f"MCP tool instance not initialized for {self.mcp_tool_name}"}
        
        return await self._mcp_tool_instance.arun(**kwargs)


class MCPSDKToolProvider:
    """Provider for MCP SDK tools."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None, namespace: str = "mcp_sdk"):
        """Initialize the MCP SDK tool provider.
        
        Args:
            server_url: URL of the MCP server
            api_key: Optional API key for authentication
            namespace: Namespace for tool registration
        """
        self.server_url = server_url
        self.api_key = api_key
        self.namespace = namespace
        self.client = MCPSDKClient(base_url=server_url, api_key=api_key)
        self.registered_tools: Set[str] = set()
        self.registered_wrappers: List[MCPSDKToolWrapper] = []
    
    async def discover_and_register_tools(self) -> List[MCPSDKToolWrapper]:
        """Discover tools from the MCP server and register them.
        
        Returns:
            List of registered MCP tool wrappers
        """
        # Initialize the client
        await self.client.initialize()
        
        # Get available tools
        tools = self.client.get_available_tools()
        logger.info(f"Discovered {len(tools)} tools from MCP server at {self.server_url}")
        
        # Create wrappers for each tool
        wrappers = []
        for tool in tools:
            tool_name = tool.get("name", "")
            
            # Skip already registered tools
            if tool_name in self.registered_tools:
                continue
            
            # Create the tool instance
            tool_instance = MCPSDKToolInstance(self.client, tool_name)
            
            # Create metadata for the tool
            metadata = ToolMetadata(
                name=tool_name,
                display_name=tool_name,
                description=tool.get("description", ""),
                categories=self._extract_categories(tool),
                capabilities=self._extract_capabilities(tool),
                domains=self._extract_domains(tool),
                source=ToolSource.MCP,
                keywords=self._extract_keywords(tool),
                examples=[]
            )
            
            # Create the wrapper
            wrapper = MCPSDKToolWrapper(
                mcp_tool_name=tool_name,
                mcp_namespace=self.namespace,
                mcp_tool_data=tool,
                mcp_tool_instance=tool_instance,
                metadata=metadata
            )
            
            # Add to registry with namespace
            registry_name = f"{self.namespace}.{tool_name}"
            registry.register_tool(wrapper, namespace=self.namespace)
            
            # Track registered tools
            self.registered_tools.add(tool_name)
            wrappers.append(wrapper)
            self.registered_wrappers.append(wrapper)
        
        logger.info(f"Registered {len(wrappers)} tools with namespace '{self.namespace}'")
        
        return wrappers
    
    def _extract_categories(self, tool: Dict[str, Any]) -> List[ToolCategory]:
        """Extract categories from tool data."""
        name = tool.get("name", "").lower()
        desc = tool.get("description", "").lower()
        
        categories = []
        
        if any(x in name or x in desc for x in ["search", "find", "lookup"]):
            categories.append(ToolCategory.SEARCH)
        
        if any(x in name or x in desc for x in ["calculate", "compute", "math"]):
            categories.append(ToolCategory.UTILITY)
        
        if any(x in name or x in desc for x in ["weather", "forecast", "temperature"]):
            categories.append(ToolCategory.UTILITY)
        
        if any(x in name or x in desc for x in ["stock", "price", "market", "financial"]):
            categories.append(ToolCategory.FINANCE)
        
        # Default to utility if nothing else matches
        if not categories:
            categories.append(ToolCategory.UTILITY)
            
        return categories
    
    def _extract_capabilities(self, tool: Dict[str, Any]) -> List[ToolCapability]:
        """Extract capabilities from tool data."""
        name = tool.get("name", "").lower()
        desc = tool.get("description", "").lower()
        
        capabilities = []
        
        if any(x in name or x in desc for x in ["search", "find", "lookup"]):
            capabilities.append(ToolCapability.SEARCH)
        
        if any(x in name or x in desc for x in ["calculate", "compute", "math"]):
            capabilities.append(ToolCapability.COMPUTE)
        
        if any(x in name or x in desc for x in ["get", "retrieve", "fetch"]):
            capabilities.append(ToolCapability.FETCH)
        
        # Default to read if nothing else matches
        if not capabilities:
            capabilities.append(ToolCapability.READ)
            
        return capabilities
    
    def _extract_domains(self, tool: Dict[str, Any]) -> List[ToolDomain]:
        """Extract domains from tool data."""
        name = tool.get("name", "").lower()
        desc = tool.get("description", "").lower()
        
        domains = []
        
        if any(x in name or x in desc for x in ["weather", "forecast", "temperature"]):
            domains.append(ToolDomain.GENERAL)
        
        if any(x in name or x in desc for x in ["stock", "price", "market", "financial"]):
            domains.append(ToolDomain.FINANCE)
        
        if any(x in name or x in desc for x in ["code", "programming", "developer"]):
            domains.append(ToolDomain.PROGRAMMING)
        
        # Default to general if nothing else matches
        if not domains:
            domains.append(ToolDomain.GENERAL)
            
        return domains
    
    def _extract_keywords(self, tool: Dict[str, Any]) -> List[str]:
        """Extract keywords from tool data."""
        name = tool.get("name", "").lower()
        desc = tool.get("description", "").lower()
        
        # Split name and description into words
        name_parts = name.replace("_", " ").split()
        desc_words = [word for word in desc.split() if len(word) > 3]
        
        # Combine all potential keywords
        all_keywords = name_parts + desc_words
        
        # Remove duplicates
        keywords = list(set(all_keywords))
        
        return keywords
    
    async def close(self):
        """Close the MCP SDK client."""
        await self.client.close()


async def test_mcp_sdk_wrapper_fixed():
    """Test the fixed MCP SDK wrapper with a mock server."""
    # Import the mock server
    from tools.mock_mcp_server import MockMCPServer
    
    # Start a mock server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        # Clear the registry
        registry.clear()
        
        # Create the MCP SDK tool provider
        provider = MCPSDKToolProvider(
            server_url=server.base_url,
            namespace="mock_mcp"
        )
        
        # Discover and register tools
        registered_tools = await provider.discover_and_register_tools()
        
        print(f"Registered {len(registered_tools)} tools:")
        for i, tool in enumerate(registered_tools):
            print(f"  {i+1}. {tool.metadata.name}")
            print(f"     Description: {tool.metadata.description}")
            print(f"     Categories: {[cat.value for cat in tool.metadata.categories]}")
            print(f"     Capabilities: {[cap.value for cap in tool.metadata.capabilities]}")
            print(f"     Domains: {[dom.value for dom in tool.metadata.domains]}")
        
        # Test selecting tools
        selector = ToolSelector(registry)
        
        query = "Calculate 125 * 8 - 32"
        print(f"\nSelecting tools for query: '{query}'")
        selected_tools = selector.select_tools(
            query=query,
            method=SelectionMethod.HYBRID,
            max_tools=2
        )
        
        print(f"Selected {len(selected_tools)} tools:")
        for i, tool in enumerate(selected_tools):
            print(f"  {i+1}. {tool.metadata.name}")
        
        # Test executing a calculator tool
        calculator_tool = next(
            (tool for tool in registered_tools if "calculator" in tool.metadata.name.lower()),
            None
        )
        
        if calculator_tool:
            print("\nExecuting calculator tool...")
            result = await calculator_tool.arun(expression="5 * 10")
            print(f"Result: {result}")
        
        # Close the provider
        await provider.close()
    
    finally:
        # Stop the server
        server.stop()


if __name__ == "__main__":
    # Run the test
    from tools.tool_selector import ToolSelector, SelectionMethod
    asyncio.run(test_mcp_sdk_wrapper_fixed()) 