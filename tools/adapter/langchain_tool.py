from typing import Any, Dict, Optional, Type, Union, List, ClassVar
from pydantic import Field, create_model, BaseModel
import inspect
import asyncio
import sys
import logging

from tools.tool import Tool

# Try to import LangChain BaseTool
try:
    from langchain.tools import BaseTool as LangChainBaseTool
except ImportError:
    # For newer versions of LangChain
    try:
        from langchain.tools.base import BaseTool as LangChainBaseTool
    except ImportError:
        raise ImportError("Could not import BaseTool from langchain. Please install langchain.")

# Set up logging
logger = logging.getLogger(__name__)

# Check if we're using Pydantic v2
try:
    import pydantic
    PYDANTIC_V2 = int(pydantic.__version__.split('.')[0]) >= 2
except (AttributeError, ImportError, IndexError):
    PYDANTIC_V2 = False

logger.debug(f"Using Pydantic v2: {PYDANTIC_V2}")

class LangChainToolAdapter(Tool):
    """Adapter for LangChain tools to work with our agent framework."""
    
    # Use different field definition based on Pydantic version
    if PYDANTIC_V2:
        model_config = {
            "arbitrary_types_allowed": True,
            "extra": "allow",  # Allow extra fields
            "populate_by_name": True  # Allow population by field name
        }
        langchain_tool: Any = Field(default=None)  # Make it optional with a default
    else:
        langchain_tool: Any = Field(exclude=True)
        
        class Config:
            arbitrary_types_allowed = True
            extra = "allow"  # Allow extra fields
    
    def __init__(self, langchain_tool=None, **kwargs):
        """Initialize the adapter with a LangChain tool.
        
        Args:
            langchain_tool: The LangChain tool to adapt
            **kwargs: Additional arguments to pass to the Tool constructor
        """
        try:
            # Set default values for required fields to avoid validation errors
            if "query" not in kwargs and hasattr(langchain_tool, "args_schema"):
                kwargs["query"] = ""  # Default empty string for query
                
            # Set name and description from the LangChain tool
            kwargs["name"] = kwargs.get("name", getattr(langchain_tool, "name", "unknown_tool"))
            kwargs["description"] = kwargs.get("description", getattr(langchain_tool, "description", "No description available"))
            
            # Extract parameters from the LangChain tool's args_schema if available
            if hasattr(langchain_tool, "args_schema") and langchain_tool.args_schema:
                try:
                    # Try to get fields from args_schema
                    schema_fields = {}
                    if hasattr(langchain_tool.args_schema, "__fields__"):
                        # Pydantic v1
                        schema_fields = langchain_tool.args_schema.__fields__
                    elif hasattr(langchain_tool.args_schema, "model_fields"):
                        # Pydantic v2
                        schema_fields = langchain_tool.args_schema.model_fields
                    
                    for field_name, field in schema_fields.items():
                        if field_name not in kwargs:
                            # Get default value safely
                            default_value = getattr(field, "default", None)
                            # If no default and field is required, provide an empty default
                            if default_value is None or default_value == ...:
                                # Determine field type and provide appropriate default
                                field_type = getattr(field, "type_", None) or getattr(field, "annotation", str)
                                if field_type == str:
                                    default_value = ""
                                elif field_type in (int, float):
                                    default_value = 0
                                elif field_type == bool:
                                    default_value = False
                                elif field_type == list or field_type == List:
                                    default_value = []
                                elif field_type == dict or field_type == Dict:
                                    default_value = {}
                            kwargs[field_name] = default_value
                except Exception as e:
                    logger.warning(f"Error extracting schema fields: {str(e)}")
            
            # Initialize the parent class first
            super().__init__(**kwargs)
            
            # Set the langchain_tool attribute directly to avoid Pydantic validation
            object.__setattr__(self, "langchain_tool", langchain_tool)
        except Exception as e:
            logger.error(f"Error initializing LangChainToolAdapter: {str(e)}")
            raise
    
    def run(self, **kwargs) -> str:
        """Run the LangChain tool synchronously."""
        try:
            # Get the actual tool instance, not the field definition
            tool = object.__getattribute__(self, "langchain_tool")
            
            # Check if we have a query parameter and the tool expects a single input
            if "query" in kwargs and len(kwargs) == 1:
                # Use the query as the tool_input
                result = tool.run(kwargs["query"])
            else:
                # Try to use the first parameter as tool_input
                try:
                    result = tool.run(next(iter(kwargs.values())))
                except (StopIteration, TypeError):
                    # Fall back to passing all kwargs
                    try:
                        result = tool.run(**kwargs)
                    except TypeError:
                        # Last resort: convert all kwargs to a string
                        result = tool.run(str(kwargs))
            
            # Ensure result is a string
            return str(result) if result is not None else ""
        except Exception as e:
            logger.error(f"Error running LangChain tool: {str(e)}")
            return f"Error: {str(e)}"
    
    async def arun(self, **kwargs) -> str:
        """Run the LangChain tool asynchronously."""
        try:
            # Get the actual tool instance, not the field definition
            tool = object.__getattribute__(self, "langchain_tool")
            
            # Check if we have a query parameter and the tool expects a single input
            if "query" in kwargs and len(kwargs) == 1:
                # Use the query as the tool_input
                result = await tool.arun(kwargs["query"])
            else:
                # Try to use the first parameter as tool_input
                try:
                    result = await tool.arun(next(iter(kwargs.values())))
                except (StopIteration, TypeError):
                    # Fall back to passing all kwargs
                    try:
                        result = await tool.arun(**kwargs)
                    except TypeError:
                        # Last resort: convert all kwargs to a string
                        result = await tool.arun(str(kwargs))
            
            # Ensure result is a string
            return str(result) if result is not None else ""
        except Exception as e:
            logger.error(f"Error running LangChain tool asynchronously: {str(e)}")
            return f"Error: {str(e)}"
    
    @classmethod
    def from_langchain_tool(cls, langchain_tool: Any) -> 'LangChainToolAdapter':
        """Create a LangChainToolAdapter from a LangChain tool."""
        try:
            # Create a new class that inherits from LangChainToolAdapter
            tool_params = {}
            
            # Extract parameters from the LangChain tool's args_schema if available
            if hasattr(langchain_tool, "args_schema") and langchain_tool.args_schema:
                try:
                    # Try to get fields from args_schema
                    schema_fields = {}
                    if hasattr(langchain_tool.args_schema, "__fields__"):
                        # Pydantic v1
                        schema_fields = langchain_tool.args_schema.__fields__
                    elif hasattr(langchain_tool.args_schema, "model_fields"):
                        # Pydantic v2
                        schema_fields = langchain_tool.args_schema.model_fields
                    
                    for field_name, field in schema_fields.items():
                        # Handle both Pydantic v1 and v2 field structures
                        field_type = str  # Default to string
                        
                        if hasattr(field, 'type_'):
                            # Pydantic v1
                            field_type = field.type_
                        elif hasattr(field, 'annotation'):
                            # Pydantic v2
                            field_type = field.annotation
                        
                        # Get description from field info
                        description = "No description"
                        if hasattr(field, 'field_info') and hasattr(field.field_info, 'description'):
                            description = field.field_info.description or "No description"
                        elif hasattr(field, 'description'):
                            description = field.description or "No description"
                        
                        # Get default value
                        default_value = getattr(field, "default", None)
                        
                        # If no default and field is required, provide an empty default
                        if default_value is None or default_value == ...:
                            # Determine field type and provide appropriate default
                            if field_type == str:
                                default_value = ""
                            elif field_type in (int, float):
                                default_value = 0
                            elif field_type == bool:
                                default_value = False
                            elif field_type == list or field_type == List:
                                default_value = []
                            elif field_type == dict or field_type == Dict:
                                default_value = {}
                        
                        tool_params[field_name] = (field_type, Field(
                            default=default_value,
                            description=description
                        ))
                except Exception as e:
                    logger.warning(f"Error extracting schema fields: {str(e)}")
            
            # Create a new class with the parameters
            if PYDANTIC_V2:
                # For Pydantic v2, we need to use a different approach
                class_name = f"{langchain_tool.__class__.__name__}Adapter"
                
                # Create a new class with the parameters
                adapted_tool_cls = type(class_name, (cls,), {
                    "__annotations__": {k: v[0] for k, v in tool_params.items()},
                    **{k: v[1] for k, v in tool_params.items()},
                    "model_config": {
                        "arbitrary_types_allowed": True,
                        "extra": "allow",
                        "populate_by_name": True
                    }
                })
            else:
                # For Pydantic v1, use create_model
                adapted_tool_cls = create_model(
                    f"{langchain_tool.__class__.__name__}Adapter",
                    __base__=cls,
                    **tool_params
                )
            
            # Create an instance of the new class with langchain_tool explicitly set
            return adapted_tool_cls(langchain_tool=langchain_tool)
        except Exception as e:
            logger.error(f"Error creating LangChainToolAdapter: {str(e)}")
            # Fallback to a simple adapter without dynamic fields
            return cls(langchain_tool=langchain_tool)


# Helper function to convert multiple LangChain tools
def convert_langchain_tools(langchain_tools: List[Any]) -> List[Tool]:
    """Convert a list of LangChain tools to our Tool format."""
    result = []
    for tool in langchain_tools:
        try:
            adapted_tool = LangChainToolAdapter.from_langchain_tool(tool)
            result.append(adapted_tool)
        except Exception as e:
            logger.error(f"Error converting LangChain tool {tool.__class__.__name__}: {str(e)}")
    return result
