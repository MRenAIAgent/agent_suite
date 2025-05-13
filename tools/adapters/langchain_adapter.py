"""
LangChain Tool Adapter for integrating LangChain tools into the enhanced tool system.

This module provides classes for adapting LangChain tools to the EnhancedTool format
and registering them with the tool registry system.
"""
import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Union, Type, Callable

from tools.types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
)
from tools.base import EnhancedTool
from tools.registry import registry

# Configure logging
logger = logging.getLogger(__name__)


class LangChainToolAdapter:
    """
    Adapter for integrating LangChain tools into the enhanced tool system.
    
    This adapter converts LangChain tools to EnhancedTool instances
    and registers them with the tool registry.
    """
    
    def __init__(
        self,
        registry_instance=None,
        namespace_prefix: Optional[str] = None
    ):
        """
        Initialize the LangChain tool adapter.
        
        Args:
            registry_instance: Optional custom tool registry to use
            namespace_prefix: Optional custom namespace prefix for LangChain tools
        """
        self.registry = registry_instance or registry
        self.namespace_prefix = namespace_prefix or "langchain"
    
    def register_langchain_tool(self, lc_tool: Any, namespace: Optional[str] = None) -> EnhancedTool:
        """
        Register a LangChain tool with the tool registry.
        
        Args:
            lc_tool: LangChain tool instance
            namespace: Optional custom namespace
            
        Returns:
            EnhancedTool instance
        """
        try:
            # Import here to avoid import errors if LangChain is not installed
            from langchain.tools.base import BaseTool, Tool
            
            # Check if the tool is a LangChain tool
            if not isinstance(lc_tool, BaseTool) and not isinstance(lc_tool, Tool):
                raise ValueError(f"Object is not a LangChain tool: {type(lc_tool)}")
            
            # Convert the LangChain tool to an EnhancedTool
            enhanced_tool = self._convert_to_enhanced_tool(lc_tool)
            
            # Register the tool
            if namespace:
                self.registry.register_tools_with_namespace(namespace, [enhanced_tool])
            else:
                namespace = f"{self.namespace_prefix}"
                self.registry.register_tools_with_namespace(namespace, [enhanced_tool])
            
            return enhanced_tool
        
        except ImportError as e:
            logger.error(f"LangChain is not installed or there was an import error: {e}")
            raise
        
        except Exception as e:
            logger.error(f"Error registering LangChain tool: {e}")
            raise
    
    def register_langchain_tools(self, lc_tools: List[Any], namespace: Optional[str] = None) -> List[EnhancedTool]:
        """
        Register multiple LangChain tools with the tool registry.
        
        Args:
            lc_tools: List of LangChain tool instances
            namespace: Optional custom namespace
            
        Returns:
            List of EnhancedTool instances
        """
        enhanced_tools = []
        
        for lc_tool in lc_tools:
            try:
                enhanced_tool = self.register_langchain_tool(lc_tool, namespace)
                enhanced_tools.append(enhanced_tool)
            except Exception as e:
                logger.error(f"Error registering LangChain tool: {e}")
        
        return enhanced_tools
    
    def _convert_to_enhanced_tool(self, lc_tool: Any) -> EnhancedTool:
        """
        Convert a LangChain tool to an EnhancedTool.
        
        Args:
            lc_tool: LangChain tool instance
            
        Returns:
            EnhancedTool instance
        """
        # Extract basic information from the LangChain tool
        tool_name = lc_tool.name
        description = lc_tool.description
        
        # Create asyncio-friendly run function wrapper
        async def execute_langchain_tool(**kwargs):
            # Get the LangChain tool's run method
            lc_run = lc_tool.run
            
            # Check if the run method is async or sync
            if asyncio.iscoroutinefunction(lc_run):
                # If it's async, await it
                return await lc_run(**kwargs)
            else:
                # If it's sync, run it in an executor to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: lc_run(**kwargs)
                )
        
        # Extract schema and parameters
        # First check if the tool has an args_schema
        if hasattr(lc_tool, "args_schema"):
            schema_cls = lc_tool.args_schema
            if schema_cls:
                parameters = {}
                
                # Get the schema properties
                if hasattr(schema_cls, "schema") and callable(schema_cls.schema):
                    schema = schema_cls.schema()
                    parameters = schema.get("properties", {})
                
                # If we couldn't get the properties from schema(), try model_json_schema()
                if not parameters and hasattr(schema_cls, "model_json_schema") and callable(schema_cls.model_json_schema):
                    schema = schema_cls.model_json_schema()
                    parameters = schema.get("properties", {})
        else:
            # No schema available, use empty dict
            parameters = {}
        
        # Create enhanced metadata
        metadata = ToolMetadata(
            categories=[ToolCategory.CUSTOM],
            domains=[ToolDomain.GENERAL],
            capabilities=[ToolCapability.CUSTOM],
            semantic_description=description,
            source=ToolSource.LANGCHAIN
        )
        
        # Try to infer more specific metadata from the tool name and description
        if any(kw in tool_name.lower() for kw in ["search", "find", "lookup", "query"]):
            metadata.categories.append(ToolCategory.SEARCH)
            metadata.capabilities.append(ToolCapability.SEARCH)
        
        if any(kw in tool_name.lower() for kw in ["web", "http", "browser"]):
            metadata.categories.append(ToolCategory.WEB_BROWSING)
            metadata.capabilities.append(ToolCapability.FETCH)
        
        if any(kw in tool_name.lower() for kw in ["file", "read", "write", "save"]):
            metadata.categories.append(ToolCategory.FILE_SYSTEM)
            if "read" in tool_name.lower():
                metadata.capabilities.append(ToolCapability.READ)
            if "write" in tool_name.lower() or "save" in tool_name.lower():
                metadata.capabilities.append(ToolCapability.WRITE)
        
        # Create the enhanced tool
        tool_cls = EnhancedTool.create_tool(
            name=tool_name,
            description=description,
            parameters=parameters,
            execute_fn=execute_langchain_tool,
            metadata=metadata,
            source=ToolSource.LANGCHAIN
        )
        
        # Create an instance
        enhanced_tool = tool_cls()
        
        # Store a reference to the original LangChain tool
        enhanced_tool.original_tool = lc_tool
        
        return enhanced_tool


# Factory function to create a LangChain tool adapter
def create_langchain_adapter(registry_instance=None, namespace_prefix=None) -> LangChainToolAdapter:
    """
    Create a LangChain tool adapter.
    
    Args:
        registry_instance: Optional custom tool registry
        namespace_prefix: Optional custom namespace prefix
        
    Returns:
        LangChainToolAdapter instance
    """
    return LangChainToolAdapter(registry_instance, namespace_prefix) 