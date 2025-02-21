from abc import ABC
from datetime import datetime
from typing import List

from llm.llm import LLMBase
from agents.prompt import PromptManager
from agents.memory import MemoryManager
from agents.cache import CacheManager
from agents.logging import LogManager
from tools.tool import Tool

class Agent(ABC):
    """Base class for AI agents."""
    
    def __init__(self, llm: LLMBase, system_prompt: str, tools: List[Tool] = None):
        self.llm = llm
        self.prompt_manager = PromptManager(system_prompt)
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager()
        self.log_manager = LogManager()
        self.tools = tools
        
    async def aprocess(self, user_input: str, model: str) -> str:
        """Process user input asynchronously and return response."""
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        
        while True:
            response = await self.llm.chat_completion(
                model=model,
                messages=messages,
                tools=self.tools
            )
            
            if not response.tool_calls:
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
    
    def process(self, user_input: str, model: str) -> str:
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
        self.log_manager.log_interaction(
            user_input=user_input,
            agent_response=response,
            model=model,
            timestamp=datetime.now().isoformat()
        )
        
        return response
    

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
                # Execute tool with provided arguments
                result = await tool.arun(**tool_call.function.arguments)
                results.append(result)
        
        return results
        
