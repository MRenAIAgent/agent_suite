from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from agents.base_classes.base_think_pattern import AgentThinkPattern
from agents.memory.memory_manager import MemoryManager
from agents.prompt import PromptManager
from log.logging import LogManager
from llm.llm import LLMBase
from tools.tool import Tool


class BaseAgent(ABC):
    """
    Abstract base class that defines the interface for all agents.
    
    This class provides the basic structure that all agent implementations
    must follow, ensuring consistency across different agent types.
    """
    
    @abstractmethod
    def __init__(
            self,
            llm: LLMBase,
            prompt_manager: PromptManager,
            tools: Optional[List[Tool]] = None,
            thinking_pattern: AgentThinkPattern = None,
            memory_manager: MemoryManager = None,
            log_manager: LogManager = None):
        """
        Initialize a new agent instance.
        
        Args:
            llm: Language model for generating responses
            prompt_manager: Manager for handling prompts
            tools: List of tools available to the agent
            thinking_pattern: Pattern for structuring agent's thinking
            persona: Dictionary defining the agent's persona
            memory_capacity: Number of past interactions to remember
            context_window: Maximum context length for the agent
        """
        pass
    
    @abstractmethod
    async def arun(self, user_input: str, model: str, **kwargs) -> str:
        """
        Process user input and generate a response asynchronously.
        
        Args:
            user_input: The user's question or request
            model: The LLM model to use
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
        """
        pass
    
    @abstractmethod
    def run(self, user_input: str, model: str, **kwargs) -> str:
        """
        Process user input and generate a response synchronously.
        
        Args:
            user_input: The user's question or request
            model: The LLM model to use
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
        """
        pass
    
    @abstractmethod
    async def think(self, user_input: str, model: str) -> Dict[str, Any]:
        """
        Process user input using the agent's thinking process.
        
        Args:
            user_input: The user's input
            model: The LLM model to use
            
        Returns:
            The agent's thoughts
        """
        pass
    
    
    @abstractmethod
    async def handle_single_tool_call(self, tool_name: str, tool_input: Any) -> Any:
        """
        Handle a single tool call.
        
        Args:
            tool_name: Name of the tool to call
            tool_input: Input for the tool
            
        Returns:
            Result of the tool call
        """
        pass
    
    @abstractmethod
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent's toolset.
        
        Args:
            tool: The tool to add
        """
        pass
    
    @abstractmethod
    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent's toolset.
        
        Args:
            tool_name: Name of the tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        pass
    
    @abstractmethod
    async def save_memory(self, context: str) -> None:
        """
        Save the agent's current state to a file.
        
        Args:
            context: Context identifier for saving the state
        """
        pass
    
    @abstractmethod
    async def load_memory(self) -> None:
        """
        Load the agent's state from a file.
        """
        pass

