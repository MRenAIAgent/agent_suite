"""
Enhanced Tool base class with metadata and registration support.

This module extends the original Tool class with enhanced metadata
and integration with the tool registry system.
"""
import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints, ClassVar

from pydantic import BaseModel, create_model, Field, validator

from tools.tool_types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata, 
    ToolSource, ToolUsageStats
)


class EnhancedTool(BaseModel, ABC):
    """
    Enhanced Tool base class with metadata and registration support.
    
    This class extends the original Tool with:
    1. Rich metadata for better tool selection
    2. Built-in registration with the tool registry
    3. Usage statistics tracking
    4. Parameter validation
    """
    
    # Class variables for registration - use ClassVar to avoid them being treated as fields
    _registry: ClassVar[Any] = None  # Will be set by the registry module to avoid circular imports
    _registered: ClassVar[bool] = False
    _instances: ClassVar[Dict[str, Any]] = {}  # Store instances for singleton behavior
    
    # Tool metadata
    metadata: ToolMetadata = Field(
        default_factory=ToolMetadata,
        description="Enhanced metadata for the tool"
    )
    
    # Source tracking
    source: ToolSource = Field(
        default=ToolSource.CUSTOM,
        description="Source of the tool"
    )
    
    # Usage statistics
    _usage_stats: Optional[ToolUsageStats] = None
    
    def __init__(self, **data):
        """Initialize the tool and register it if not already registered."""
        super().__init__(**data)
        
        # Initialize usage stats
        if self._usage_stats is None:
            self._usage_stats = ToolUsageStats()
            
        # Auto-register with registry if available and not already registered
        cls = self.__class__
        if not cls._registered and cls._registry is not None:
            cls._registry.register_tool(self)
            cls._registered = True
            
        # Store instance in class dict for singleton behavior
        cls._instances[cls.__name__] = self
    
    @abstractmethod
    async def arun(self, **kwargs) -> Any:
        """
        Asynchronously execute the tool's functionality.
        
        Args:
            **kwargs: Tool parameters passed from the LLM
            
        Returns:
            Any: Result of the tool execution
        """
        pass

    def run(self, **kwargs) -> Any:
        """
        Synchronously execute the tool's functionality.
        
        Args:
            **kwargs: Tool parameters passed from the LLM
            
        Returns:
            Any: Result of the tool execution
        """
        return asyncio.run(self.arun(**kwargs))
    
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool and track usage statistics.
        
        Args:
            **kwargs: Tool parameters passed from the LLM
            
        Returns:
            Any: Result of the tool execution
        """
        start_time = time.time()
        success = True
        result = None
        
        try:
            # Execute the tool
            result = await self.arun(**kwargs)
            return result
        except Exception as e:
            success = False
            raise e
        finally:
            # Track usage statistics
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            if self._usage_stats:
                self._usage_stats.update(success, execution_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary representation."""
        schema = self.model_json_schema()
        
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__.strip() if self.__doc__ else "",
            "parameters": schema.get("properties", {}),
            "required": schema.get("required", []),
            "metadata": self.metadata.model_dump() if self.metadata else {},
            "source": self.source
        }
    
    def convert_to_function_call(self) -> Dict:
        """Convert the tool to OpenAI function call format."""
        # Get model fields from Pydantic
        schema = self.model_json_schema()
        
        return {
            "type": "function",
            "function": {
                "name": self.__class__.__name__.lower(),
                "description": self.__doc__.strip() if self.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", [])
                }
            }
        }
        
    def get_usage_stats(self) -> ToolUsageStats:
        """Get usage statistics for this tool."""
        return self._usage_stats if self._usage_stats else ToolUsageStats()
    
    @classmethod
    def create_tool(
        cls,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        execute_fn: callable,
        metadata: Optional[ToolMetadata] = None,
        source: ToolSource = ToolSource.CUSTOM
    ) -> Type['EnhancedTool']:
        """
        Create a new tool class dynamically.
        
        Args:
            name: Name of the tool class
            description: Description for the tool
            parameters: Dictionary of parameter names to type annotations
            execute_fn: Function that implements the tool's functionality
            metadata: Tool metadata
            source: Source of the tool
            
        Returns:
            A new tool class
        """
        # Create a pydantic model for the parameters
        param_fields = {}
        for param_name, param_type in parameters.items():
            param_fields[param_name] = (param_type, ...)
            
        # Create the new tool class
        tool_cls = type(
            name,
            (cls,),
            {
                "__doc__": description,
                "arun": execute_fn,
                "metadata": metadata or ToolMetadata(),
                "source": source
            }
        )
        
        return tool_cls
        
    @classmethod
    def get_instance(cls) -> 'EnhancedTool':
        """Get or create a singleton instance of this tool class."""
        if cls.__name__ not in cls._instances:
            cls._instances[cls.__name__] = cls()
        return cls._instances[cls.__name__] 