from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel


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
