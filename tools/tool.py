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
        # Get all class attributes that don't start with _ 
        params = {
            name: getattr(self, name) 
            for name in dir(self) 
            if not name.startswith('_') and not callable(getattr(self, name))
        }
        
        return {
            "name": self.__class__.__name__.lower(),
            "description": self.__doc__,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(params.keys())
            }
        }
