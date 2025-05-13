"""
Integration module for converting and registering legacy tools to the enhanced tool system.
"""
import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Type

from tools.base import EnhancedTool
from tools.tool import Tool
from tools.registry import registry

logger = logging.getLogger(__name__)


def register_all_adapted_tools() -> List[EnhancedTool]:
    """
    Register all adapter implementations of legacy tools.
    
    This function registers all enhanced tools that have been explicitly 
    implemented as adapters for legacy tools.
    
    Returns:
        List of registered enhanced tools
    """
    registered_tools = []
    
    # Try to register SerperSearchTool adapter
    try:
        from tools.serper_api import SerperSearchTool
        serper_tool = SerperSearchTool(query="")
        enhanced_tool = serper_tool.to_enhanced_tool()
        registry.register_tool(enhanced_tool)
        registered_tools.append(enhanced_tool)
        logger.info(f"Registered SerperSearchTool adapter: {enhanced_tool.__class__.__name__}")
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to register SerperSearchTool adapter: {e}")
    
    # Try to register AnimalType adapter
    try:
        from tools.animal_type import AnimalType
        animal_tool = AnimalType(animal_name="")
        enhanced_tool = animal_tool.to_enhanced_tool()
        registry.register_tool(enhanced_tool)
        registered_tools.append(enhanced_tool)
        logger.info(f"Registered AnimalType adapter: {enhanced_tool.__class__.__name__}")
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to register AnimalType adapter: {e}")
    
    # Add more explicit registrations here
    
    return registered_tools


def auto_convert_legacy_tools() -> List[EnhancedTool]:
    """
    Auto-convert any legacy tools found in the tools module.
    
    This function scans the tools module for any Tool subclasses that 
    haven't been registered yet and auto-converts them to enhanced tools.
    
    Returns:
        List of auto-converted and registered enhanced tools
    """
    registered_tools = []
    
    # Get all registered tool names for checking
    registered_names = [tool.__class__.__name__.lower() for tool in registry.get_all_tools()]
    
    # Try to import common tool modules
    tool_modules = [
        "tools.tool",
        "tools.serpapi_tool",
        "tools.mcp_tool",
        "tools.animal_type",
        "tools.serper_api",
        # Add more modules to scan here
    ]
    
    # Scan each module for Tool subclasses
    for module_name in tool_modules:
        try:
            module = importlib.import_module(module_name)
            
            # Find all Tool subclasses in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Tool) and 
                    obj != Tool and 
                    name.lower() not in registered_names):
                    
                    try:
                        # Create an instance and convert to enhanced tool
                        # Create default params for required fields
                        params = {}
                        
                        # Get model schema to find required fields
                        schema = obj.model_json_schema()
                        required_fields = schema.get("required", [])
                        properties = schema.get("properties", {})
                        
                        # Set empty strings for required string fields
                        for field_name in required_fields:
                            field_type = properties.get(field_name, {}).get("type")
                            if field_type == "string":
                                params[field_name] = ""
                            elif field_type == "integer" or field_type == "number":
                                params[field_name] = 0
                            elif field_type == "boolean":
                                params[field_name] = False
                            # Other types (arrays, objects) are more complex to handle
                        
                        # Add any defaults for optional fields  
                        for field_name, field in obj.model_fields.items():
                            if hasattr(field, 'default') and field.default is not None and field.default != ...:
                                params[field_name] = field.default
                        
                        try:
                            # Create an instance with the collected parameters
                            instance = obj(**params)
                            enhanced_tool = instance.to_enhanced_tool()
                            
                            # Register the enhanced tool
                            registry.register_tool(enhanced_tool)
                            registered_tools.append(enhanced_tool)
                            logger.info(f"Auto-converted and registered legacy tool: {name}")
                        except Exception as e:
                            logger.warning(f"Failed to create instance of {name}: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to auto-convert legacy tool {name}: {e}")
        
        except ImportError:
            # Skip modules that can't be imported
            logger.debug(f"Could not import module: {module_name}")
        except Exception as e:
            logger.warning(f"Error scanning module {module_name}: {e}")
    
    return registered_tools 