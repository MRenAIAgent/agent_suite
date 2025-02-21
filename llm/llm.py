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
    def chat_completion(self, model: str, messages: list, tools: list = None, stream: bool = False):
        """Create a chat completion using the LLM.
        
        Args:
            messages (list): List of message dictionaries
            stream (bool): Whether to stream the response
        """
        pass

