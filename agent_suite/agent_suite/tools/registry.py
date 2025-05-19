"""
Tool Registry for managing the registration and retrieval of tools.

This module provides a centralized registry for tools that allows
tools to be registered, retrieved, and organized by source, category,
and other metadata.
"""
import importlib
import inspect
import logging
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from tools.tool_types import ToolCategory, ToolDomain, ToolMetadata, ToolSource, Task, Role, ToolCapability
from tools.base import EnhancedTool

# Configure logging
logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Centralized registry for managing tools.
    
    The registry provides:
    1. Registration of tools from various sources
    2. Tool retrieval by name, category, or capability
    3. Namespace management to avoid name collisions
    4. Tool usage statistics tracking
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ToolRegistry':
        """Get or create the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        # Main tool storage - dictionary of tool name to tool instance
        self.tools = {}
        
        # Instance counter for generating unique names
        self.instance_counts = {}
        
        # Namespace tracking - dictionary of namespace to set of tool names
        self.namespaces = {}
        
        # Initialize with empty system namespace
        self.namespaces["system"] = set()
        
        # Connect to the EnhancedTool class
        from tools.base import EnhancedTool
        EnhancedTool._registry = self
    
    def register_tool(self, tool: Union[EnhancedTool, Type[EnhancedTool]], override: bool = False) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: Tool instance or class to register
            override: If True, overrides existing tool with the same name
        """
        # Convert class to instance if needed
        if inspect.isclass(tool) and issubclass(tool, EnhancedTool):
            # Use get_instance to ensure singleton behavior
            if hasattr(tool, 'get_instance'):
                tool = tool.get_instance()
            else:
                tool = tool()
        
        # Get the tool name
        tool_name = tool.__class__.__name__
        
        # Check for name collisions and handle accordingly
        if tool_name in self.tools:
            if override:
                # Override existing tool
                self.tools[tool_name] = tool
                logger.info(f"Overriding existing tool: {tool_name}")
            else:
                # Generate a unique name using instance counter
                if tool_name not in self.instance_counts:
                    self.instance_counts[tool_name] = 1
                else:
                    self.instance_counts[tool_name] += 1
                
                new_name = f"{tool_name}_{self.instance_counts[tool_name]}"
                logger.warning(f"Tool with name {tool_name} already registered, using unique name: {new_name}")
                self.tools[new_name] = tool
                tool_name = new_name
        else:
            # Register the tool with its original name
            self.tools[tool_name] = tool
        
        # Track in appropriate namespace
        namespace = self._get_namespace_for_source(tool.source)
        namespace_set = self.namespaces.get(namespace, set())
        namespace_set.add(tool_name)
        self.namespaces[namespace] = namespace_set
        
        logger.info(f"Registered tool {tool_name} from source {tool.source}")
    
    def register_tools(self, tools: List[Union[EnhancedTool, Type[EnhancedTool]]]) -> None:
        """
        Register multiple tools at once.
        
        Args:
            tools: List of tool instances or classes to register
        """
        for tool in tools:
            self.register_tool(tool)
    
    def register_tools_with_namespace(self, namespace: str, tools: List[Union[EnhancedTool, Type[EnhancedTool]]]) -> None:
        """
        Register tools with a custom namespace.
        
        Args:
            namespace: Namespace to register tools under
            tools: List of tool instances or classes to register
        """
        # Create namespace tracking if it doesn't exist
        if namespace not in self.namespaces:
            self.namespaces[namespace] = set()
        
        # Register each tool
        for tool in tools:
            # Convert class to instance if needed
            if inspect.isclass(tool) and issubclass(tool, EnhancedTool):
                if hasattr(tool, 'get_instance'):
                    tool = tool.get_instance()
                else:
                    tool = tool()
            
            # Create a namespaced name
            tool_name = f"{namespace}.{tool.__class__.__name__}"
            
            # Check for name collisions
            if tool_name in self.tools:
                # Generate a unique name
                if tool_name not in self.instance_counts:
                    self.instance_counts[tool_name] = 1
                else:
                    self.instance_counts[tool_name] += 1
                
                new_name = f"{tool_name}_{self.instance_counts[tool_name]}"
                logger.warning(f"Tool with name {tool_name} already registered, using unique name: {new_name}")
                tool_name = new_name
            
            # Register the tool
            self.tools[tool_name] = tool
            
            # Track in namespace
            self.namespaces[namespace].add(tool_name)
            
            logger.info(f"Registered tool {tool_name} in namespace {namespace}")
    
    def get_tool(self, name: str) -> Optional[EnhancedTool]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)
    
    def get_tools_by_source(self, source: ToolSource) -> List[EnhancedTool]:
        """
        Get all tools from a specific source.
        
        Args:
            source: Source to filter by
            
        Returns:
            List of tool instances from the specified source
        """
        return [tool for tool in self.tools.values() if tool.source == source]
    
    def get_tools_by_category(self, category: Union[ToolCategory, str]) -> List[EnhancedTool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of tool instances in the specified category
        """
        return [
            tool for tool in self.tools.values() 
            if any(cat == category for cat in tool.metadata.categories)
        ]
    
    def get_tools_by_domain(self, domain: Union[ToolDomain, str]) -> List[EnhancedTool]:
        """
        Get all tools for a specific domain.
        
        Args:
            domain: Domain to filter by
            
        Returns:
            List of tool instances for the specified domain
        """
        return [
            tool for tool in self.tools.values() 
            if any(dom == domain for dom in tool.metadata.domains)
        ]
    
    def get_tools_in_namespace(self, namespace: str) -> List[EnhancedTool]:
        """
        Get all tools in a specific namespace.
        
        Args:
            namespace: Namespace to filter by
            
        Returns:
            List of tool instances in the specified namespace
        """
        tool_names = self.namespaces.get(namespace, set())
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_all_tools(self) -> List[EnhancedTool]:
        """
        Get all registered tools.
        
        Returns:
            List of all registered tool instances
        """
        return list(self.tools.values())
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self.tools = {}
        self.namespaces = {"system": set()}
        self.instance_counts = {}
    
    def _get_namespace_for_source(self, source: ToolSource) -> str:
        """
        Get the default namespace for a tool source.
        
        Args:
            source: Tool source
            
        Returns:
            Default namespace for the source
        """
        # Use the source name as the namespace
        return source.value
    
    def scan_module_for_tools(self, module_name: str) -> List[EnhancedTool]:
        """
        Scan a module for tool classes and register them.
        
        Args:
            module_name: Name of the module to scan
            
        Returns:
            List of registered tool instances
        """
        registered_tools = []
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get all classes in the module
            for name, obj in inspect.getmembers(module):
                # Check if it's a class and a subclass of EnhancedTool
                if (inspect.isclass(obj) and 
                    issubclass(obj, EnhancedTool) and 
                    obj.__module__ == module_name and
                    obj != EnhancedTool):
                    
                    # Register the tool
                    self.register_tool(obj)
                    if hasattr(obj, 'get_instance'):
                        registered_tools.append(obj.get_instance())
                    else:  
                        registered_tools.append(obj())
        
        except Exception as e:
            logger.error(f"Error scanning module {module_name} for tools: {e}")
        
        return registered_tools
    
    def scan_package_for_tools(self, package_name: str) -> List[EnhancedTool]:
        """
        Recursively scan a package for tool classes and register them.
        
        Args:
            package_name: Name of the package to scan
            
        Returns:
            List of registered tool instances
        """
        registered_tools = []
        
        try:
            # Import the package
            package = importlib.import_module(package_name)
            
            # Get the package path
            if hasattr(package, '__path__'):
                # Scan modules in the package
                for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
                    if is_pkg:
                        # Recursively scan subpackages
                        registered_tools.extend(self.scan_package_for_tools(module_name))
                    else:
                        # Scan the module for tools
                        registered_tools.extend(self.scan_module_for_tools(module_name))
        
        except Exception as e:
            logger.error(f"Error scanning package {package_name} for tools: {e}")
        
        return registered_tools


# Create the global registry instance
registry = ToolRegistry.get_instance() 