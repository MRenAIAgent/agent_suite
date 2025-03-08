from abc import ABC, abstractmethod


class LLMBase(ABC):
    """Base class for LLM implementations."""
    
    @abstractmethod
    def create_llm(self):
        """Create a chat completion using the LLM.
        
        Args:
            messages (list): List of message dictionaries
            stream (bool): Whether to stream the response
            
        Returns:
            str: The model's response
        """
        pass

    @abstractmethod
    def chat_completion(self, model: str, messages: list, tools: list = None, stream: bool = False, **kwargs):
        """Create a chat completion using the LLM.
        
        Args:
            model (str): The model to use for completion
            messages (list): List of message dictionaries
            tools (list, optional): List of tools available to the model
            stream (bool, optional): Whether to stream the response
            tool_choice (str, optional): Specify which tool to use
            **kwargs: Additional parameters to pass to the LLM
        """
        pass
