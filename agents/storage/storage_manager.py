from typing import Dict, List, Any, Optional, Type
import time
import json
from datetime import datetime

from agents.storage.intermediate_storage import (
    IntermediateStorageBase, 
    InMemoryIntermediateStorage,
    TokenEfficientIntermediateStorage
)


class IntermediateStepManager:
    """
    Manager class for intermediate execution steps and tool results.
    
    This class provides a centralized way to track, store, and retrieve
    intermediate reasoning steps and tool results during agent execution.
    """
    
    def __init__(
        self, 
        storage_implementation: Optional[IntermediateStorageBase] = None,
        token_efficient: bool = False,
        max_detailed_steps: int = 5
    ):
        """
        Initialize the intermediate step manager.
        
        Args:
            storage_implementation: Optional custom storage implementation
            token_efficient: Whether to use token-efficient storage
            max_detailed_steps: Number of detailed steps to preserve when using token-efficient storage
        """
        if storage_implementation:
            self.storage = storage_implementation
        elif token_efficient:
            self.storage = TokenEfficientIntermediateStorage(max_detailed_steps=max_detailed_steps)
        else:
            self.storage = InMemoryIntermediateStorage()
            
        self.current_iteration = 0
        self.start_time = None
    
    def start_execution(self):
        """Start an execution run, resetting storage."""
        self.storage.clear()
        self.current_iteration = 0
        self.start_time = time.time()
    
    def start_iteration(self):
        """Start a new iteration."""
        self.current_iteration += 1
        
        # Mark start of a new iteration
        self.storage.add_step(
            "iteration_start", 
            f"Iteration {self.current_iteration}", 
            {"timestamp": time.time()}
        )
    
    def add_thought(self, thought: str):
        """
        Add a thought step.
        
        Args:
            thought: The agent's reasoning step
        """
        self.storage.add_step(
            "thought", 
            thought, 
            {
                "iteration": self.current_iteration,
                "timestamp": time.time()
            }
        )
    
    def add_action(self, action: str, action_input: Any = None):
        """
        Add an action step.
        
        Args:
            action: The action to take
            action_input: Input parameters for the action
        """
        self.storage.add_step(
            "action", 
            action, 
            {
                "iteration": self.current_iteration,
                "timestamp": time.time()
            }
        )
        
        if action_input is not None:
            self.storage.add_step(
                "action_input", 
                action_input, 
                {
                    "iteration": self.current_iteration,
                    "timestamp": time.time()
                }
            )
    
    def add_observation(self, observation: Any):
        """
        Add an observation result.
        
        Args:
            observation: The result of an action
        """
        self.storage.add_step(
            "observation", 
            observation, 
            {
                "iteration": self.current_iteration,
                "timestamp": time.time()
            }
        )
    
    def add_tool_result(self, tool_name: str, tool_input: Any, tool_output: Any):
        """
        Add a tool execution result.
        
        Args:
            tool_name: Name of the tool called
            tool_input: Input provided to the tool
            tool_output: Output returned by the tool
        """
        self.storage.add_tool_call(
            tool_name,
            tool_input,
            tool_output,
            {
                "iteration": self.current_iteration,
                "timestamp": time.time()
            }
        )
    
    def add_final_answer(self, final_answer: str):
        """
        Add the final answer.
        
        Args:
            final_answer: The agent's final response
        """
        self.storage.add_step(
            "final_answer", 
            final_answer,
            {
                "iteration": self.current_iteration,
                "timestamp": time.time(),
                "execution_time": time.time() - (self.start_time or time.time())
            }
        )
    
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get intermediate steps as messages for LLM context.
        
        Args:
            max_tokens: Optional token limit
            
        Returns:
            List of message objects for LLM context
        """
        return self.storage.get_as_messages(max_tokens=max_tokens)
    
    def get_formatted_context(self, format_type: str = "default") -> str:
        """
        Get formatted context for prompt inclusion.
        
        Args:
            format_type: Formatting style
            
        Returns:
            Formatted context string
        """
        return self.storage.get_formatted_context(format_type=format_type)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the execution.
        
        Returns:
            Dictionary with execution summary information
        """
        stats = self.storage.get_stats()
        
        # Enrich with additional information
        duration = None
        if self.start_time:
            duration = time.time() - self.start_time
            
        return {
            "total_iterations": self.current_iteration,
            "execution_duration": duration,
            "timestamp": datetime.now().isoformat(),
            **stats
        }
    
    def clear(self):
        """Clear all stored steps."""
        self.storage.clear()
        self.current_iteration = 0
        self.start_time = None 