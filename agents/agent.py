from abc import ABC
from datetime import datetime
from typing import List, Any, Optional
import json

from agents.base_classes.base_agent import BaseAgent
from agents.base_classes.base_think_pattern import AgentThinkPattern
from agents.cache import CacheManager
from agents.memory_manager import MemoryManager
from agents.prompt import PromptManager
from log.logging import LogManager
from llm.llm import LLMBase
from tools.tool import Tool

class Agent(BaseAgent):
    """This is Agent with tools and memory."""

    def __init__(
            self,
            llm: LLMBase,
            prompt_manager: PromptManager,
            tools: Optional[List[Tool]] = None,
            thinking_pattern: AgentThinkPattern = None,
            memory_manager: MemoryManager = None,
            log_manager: LogManager = None):

        self.llm = llm
        self.prompt_manager = prompt_manager
        self.memory_manager = memory_manager if memory_manager else MemoryManager()
        self.cache_manager = CacheManager()
        self.tools = tools or []
        self.log_manager = log_manager
        self.max_iterations = 5
        self.thinking_pattern = thinking_pattern
        self.metadata = {}

    async def think(self, user_input: str, model: str) -> dict:
        """
        Process user input using the agent's thinking process.
        
        Args:
            user_input: The user's input
            model: The LLM model to use
            
        Returns:
            The agent's thoughts
        """
        # Implement basic thinking process
        return {
            "input": user_input,
            "thoughts": f"Processing input: {user_input}"
        }

    async def arun(self, user_input: str, model: str, **kwargs) -> str:
        """Process user input asynchronously and return response."""
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        
        while True:
            response = self.llm.chat_completion(
                model=model,
                messages=messages,
                tools=[tool.convert_to_function_call() for tool in self.tools]
            )
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                break
                
            tool_results = await self.handle_tool_calls(response.tool_calls)

            # Add tool results to messages for next iteration
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls
            })
            messages.append({
                "role": "tool",
                "content": str(tool_results)
            })
        # Update history
        self.memory_manager.add({"role": "user", "content": user_input})
        self.memory_manager.add({"role": "assistant", "content": response})
        
        # Log interaction
        self.log_manager.log_interaction(
            user_input=user_input,
            agent_response=response,
            model=model,
            timestamp=datetime.now().isoformat()
        )
        return response
    
    def run(self, user_input: str, model: str, **kwargs) -> str:
        """Process user input synchronously and return response."""
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        
        response = self.llm.chat_completion(
            model=model,
            messages=messages
        )
        
        # Update history
        self.memory_manager.add({"role": "user", "content": user_input})
        self.memory_manager.add({"role": "assistant", "content": response})
        
        # Log interaction
        self.log_interaction(user_input, response, model)
        
        return response
    
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent's toolset.
        
        Args:
            tool: The tool to add
        """
        self.tools.append(tool)
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent's toolset.
        
        Args:
            tool_name: Name of the tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        initial_count = len(self.tools)
        self.tools = [t for t in self.tools if t.__class__.__name__ != tool_name]
        return len(self.tools) < initial_count

    async def handle_tool_calls(self, tool_calls):
        """Handle tool calls from LLM response.
        
        Args:
            tool_calls: List of tool calls from LLM response
            
        Returns:
            Results from executing the tool calls
        """
        results = []
        for tool_call in tool_calls:
            # Find matching tool
            tool_name = tool_call.function.name
            tool = next((t for t in self.tools if t.__class__.__name__.lower() == tool_name), None)
            
            if tool:
                arguments = json.loads(tool_call.function.arguments)
                # Execute tool with provided arguments
                result = await tool.arun(**arguments)
                results.append(result)
        
        return results
        
    async def handle_single_tool_call(self, tool_name: str, tool_input: Any) -> Any:
        """Handle a single tool call."""
        tool = None
        for t in self.tools:
            if t.__class__.__name__.lower() == tool_name.replace('functions.', ''):
                tool = t
        if tool:
            result = await tool.arun(tool_input)
            return result
        return 'Error to get result from tool'
    
    async def save_memory(self, context: str) -> None:
        """
        Save the agent's current state to a file.
        
        Args:
            context: Context identifier for saving the state
        """
        return await self.memory_manager.save_memory(context)
    
    async def load_memory(self) -> None:
        """
        Load the agent's state from a file.
        """
        return await self.memory_manager.load_memory()

    
    def log_interaction(self, user_input: str, response: str, model: str, **kwargs) -> None:
        """
        Log an interaction with the agent.
        
        Args:
            user_input: The user's input
            response: The agent's response
            model: The model used for the interaction
            **kwargs: Additional metadata to log
        """
        self.log_manager.log_interaction(
            user_input=user_input,
            agent_response=response,
            model=model,
            timestamp=datetime.now().isoformat(),
            metadata=kwargs
        )