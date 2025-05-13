from abc import ABC, abstractmethod
from typing import Any, Dict, ForwardRef
from pydantic import BaseModel, Field

# Import enhanced tool system types - but not EnhancedTool itself
from tools.tool_types import ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource

# Create a forward reference to avoid circular imports
EnhancedTool = ForwardRef("EnhancedTool")


class Tool(BaseModel):
    """Base class for tools that can be used by LLMs.
    
    Tools should define their parameters as class attributes which will be 
    converted to function call parameters when used with LLMs.
    """

    @abstractmethod
    async def arun(self, **kwargs) -> Any:
        """Asynchronously execute the tool's functionality.
        
        Args:
            **kwargs: Tool parameters passed from the LLM
            
        Returns:
            Any: Result of the tool execution
        """
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Synchronously execute the tool's functionality.
        
        Args:
            **kwargs: Tool parameters passed from the LLM
            
        Returns:
            Any: Result of the tool execution
        """
        pass

    def convert_to_function_call(self) -> Dict:
        """Convert the tool to OpenAI function call format.
        
        Returns:
            Dict: Function definition in OpenAI format
        """
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
    
    def to_enhanced_tool(self) -> EnhancedTool:
        """Convert this tool to an enhanced tool with default metadata.
        
        This provides backward compatibility for existing tools by generating
        a generic wrapper that delegates to the original tool implementation.
        
        Returns:
            An EnhancedTool version of this tool
        """
        # To avoid circular imports, import EnhancedTool inside the method
        from tools.base import EnhancedTool
        
        # Create a dynamic enhanced tool class for this tool
        tool_class = self.__class__
        tool_name = tool_class.__name__
        
        # Create a generic enhanced tool wrapper
        class EnhancedToolWrapper(EnhancedTool):
            """Enhanced tool wrapper for legacy tool."""
            
            # Define original_tool as a field
            original_tool: Tool = Field(..., description="Original tool being wrapped")
            
            def __init__(self, original_tool, **data):
                """Initialize with enhanced metadata and original tool reference."""
                # Generate default metadata if not provided
                if "metadata" not in data:
                    # Extract field info from schema for parameters
                    schema = original_tool.model_json_schema()
                    
                    data["metadata"] = ToolMetadata(
                        name=tool_name.lower(),
                        display_name=tool_name,
                        description=tool_class.__doc__.strip() if tool_class.__doc__ else f"Legacy tool: {tool_name}",
                        categories=[ToolCategory.UTILITY],  # Use UTILITY instead of GENERAL
                        domains=[ToolDomain.GENERAL],
                        capabilities=[ToolCapability.CUSTOM],  # Use CUSTOM instead of GENERAL
                        source=ToolSource.CUSTOM,  # Use CUSTOM instead of LEGACY
                        parameters=schema.get("properties", {}),
                        required_parameters=schema.get("required", [])
                    )
                
                # Copy parameters from original tool
                for field_name, field_value in original_tool.model_dump().items():
                    if field_name not in data:
                        data[field_name] = field_value
                
                # Add original_tool to data
                data["original_tool"] = original_tool
                
                super().__init__(**data)
            
            async def arun(self, **kwargs) -> Any:
                """Run the tool asynchronously by delegating to the original tool."""
                # Create a new instance with any updated parameters
                if kwargs:
                    # Get current parameters
                    params = self.original_tool.model_dump()
                    
                    # Update with any provided parameters
                    params.update(kwargs)
                    
                    # Create a new instance with the updated parameters
                    tool_instance = self.original_tool.__class__(**params)
                    return await tool_instance.arun(**kwargs)
                else:
                    # No changes, use the original tool
                    return await self.original_tool.arun(**kwargs)
            
            def run(self, **kwargs) -> Any:
                """Run the tool synchronously by delegating to the original tool."""
                # Create a new instance with any updated parameters
                if kwargs:
                    # Get current parameters
                    params = self.original_tool.model_dump()
                    
                    # Update with any provided parameters
                    params.update(kwargs)
                    
                    # Create a new instance with the updated parameters
                    tool_instance = self.original_tool.__class__(**params)
                    return tool_instance.run(**kwargs)
                else:
                    # No changes, use the original tool
                    return self.original_tool.run(**kwargs)
        
        # Return an instance of the enhanced wrapper
        return EnhancedToolWrapper(original_tool=self)
