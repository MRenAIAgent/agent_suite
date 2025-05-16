"""
Intermediate storage implementations for agent execution.

This module provides various implementations for storing and managing
intermediate reasoning steps and tool results during agent execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import json


class IntermediateStorageBase(ABC):
    """
    Abstract base class for storing intermediate reasoning steps and tool results.
    
    This class defines the interface for storing, retrieving, and managing
    intermediate steps generated during agent execution. Different implementations
    can optimize for different characteristics (e.g., memory usage, retrieval speed,
    token efficiency).
    """
    
    @abstractmethod
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an intermediate step to storage.
        
        Args:
            step_type: Type of step (e.g., 'thought', 'action', 'tool_result')
            content: The content of the step
            metadata: Optional metadata about the step (e.g., timestamps)
        """
        pass
    
    @abstractmethod
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a tool call result to storage.
        
        Args:
            tool_name: Name of the tool called
            tool_input: Input provided to the tool
            tool_output: Output returned by the tool
            metadata: Optional metadata about the tool call (e.g., latency)
        """
        pass
    
    @abstractmethod
    def get_steps(self, step_types: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve stored intermediate steps.
        
        Args:
            step_types: Optional filter for specific step types
            limit: Optional limit on number of steps to return
            
        Returns:
            List of stored steps matching criteria
        """
        pass
    
    @abstractmethod
    def get_formatted_context(self, format_type: str = "default") -> str:
        """
        Get formatted context string for inclusion in LLM prompts.
        
        Args:
            format_type: Type of formatting to use (e.g., 'default', 'compact', 'detailed')
            
        Returns:
            Formatted string representation of relevant context
        """
        pass
    
    @abstractmethod
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get stored steps as message objects for LLM context.
        
        Args:
            max_tokens: Optional maximum token limit to respect
            
        Returns:
            List of message objects for LLM context
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored intermediate steps."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored steps.
        
        Returns:
            Dictionary with statistics (e.g., count by type, token estimates)
        """
        pass


class InMemoryIntermediateStorage(IntermediateStorageBase):
    """
    Simple in-memory implementation of intermediate step storage.
    
    Stores all steps in memory as a list of dictionaries.
    """
    
    def __init__(self):
        self.steps = []
        self.tool_calls = []
    
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an intermediate step to storage."""
        step = {
            "type": step_type,
            "content": content,
            "metadata": metadata or {}
        }
        self.steps.append(step)
    
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a tool call result to storage."""
        tool_call = {
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output,
            "metadata": metadata or {}
        }
        self.tool_calls.append(tool_call)
        
        # Also add as a step for unified access
        self.add_step(
            "tool_call", 
            {"tool_name": tool_name, "input": tool_input, "output": tool_output},
            metadata
        )
    
    def get_steps(self, step_types: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve stored intermediate steps."""
        filtered_steps = self.steps
        
        if step_types:
            filtered_steps = [step for step in filtered_steps if step["type"] in step_types]
        
        if limit is not None:
            filtered_steps = filtered_steps[-limit:]
            
        return filtered_steps
    
    def get_formatted_context(self, format_type: str = "default") -> str:
        """Get formatted context string for inclusion in LLM prompts."""
        if format_type == "compact":
            # Compact format focuses on key information with minimal verbosity
            formatted_steps = []
            for step in self.steps:
                if step["type"] == "thought":
                    formatted_steps.append(f"Thought: {step['content']}")
                elif step["type"] == "action":
                    formatted_steps.append(f"Action: {step['content']}")
                elif step["type"] == "tool_call":
                    tool_info = step["content"]
                    formatted_steps.append(
                        f"Tool: {tool_info['tool_name']}\n"
                        f"Input: {json.dumps(tool_info['input'], ensure_ascii=False)}\n"
                        f"Result: {json.dumps(tool_info['output'], ensure_ascii=False)}"
                    )
            return "\n".join(formatted_steps)
            
        elif format_type == "detailed":
            # Detailed format includes all available information
            formatted_steps = []
            for i, step in enumerate(self.steps):
                step_content = json.dumps(step["content"], ensure_ascii=False) if isinstance(step["content"], (dict, list)) else step["content"]
                formatted_steps.append(f"Step {i+1} [{step['type']}]: {step_content}")
            return "\n\n".join(formatted_steps)
            
        else:  # default format - ReAct style
            formatted_steps = []
            for step in self.steps:
                if step["type"] == "thought":
                    formatted_steps.append(f"Thought: {step['content']}")
                elif step["type"] == "action":
                    formatted_steps.append(f"Action: {step['content']}")
                elif step["type"] == "action_input":
                    formatted_steps.append(f"Action Input: {step['content']}")
                elif step["type"] == "observation" or step["type"] == "tool_call":
                    if step["type"] == "tool_call":
                        tool_info = step["content"]
                        formatted_steps.append(f"Observation: {json.dumps(tool_info['output'], ensure_ascii=False)}")
                    else:
                        formatted_steps.append(f"Observation: {step['content']}")
            return "\n".join(formatted_steps)
    
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get stored steps as message objects for LLM context."""
        messages = []
        
        # Convert tool calls to message format
        for tool_call in self.tool_calls:
            # Create unique ID for the tool call
            call_id = f"call_{len(messages)}"
            
            # Add assistant message with tool call
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call["tool_name"],
                        "arguments": json.dumps(tool_call["input"])
                    }
                }]
            })
            
            # Add tool response
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(tool_call["output"])
            })
        
        # Add other relevant steps as system messages
        # Group consecutive thoughts/actions/observations
        current_block = []
        for step in self.steps:
            if step["type"] in ["thought", "action", "action_input", "observation"] and step["type"] != "tool_call":
                if step["type"] == "thought":
                    current_block.append(f"Thought: {step['content']}")
                elif step["type"] == "action":
                    current_block.append(f"Action: {step['content']}")
                elif step["type"] == "action_input":
                    current_block.append(f"Action Input: {step['content']}")
                elif step["type"] == "observation":
                    current_block.append(f"Observation: {step['content']}")
        
        # Add the accumulated block as a system message
        if current_block:
            messages.append({
                "role": "system",
                "content": "\n".join(current_block)
            })
        
        # TODO: Implement token limiting if max_tokens is specified
        
        return messages
    
    def clear(self) -> None:
        """Clear all stored intermediate steps."""
        self.steps = []
        self.tool_calls = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored steps."""
        # Count steps by type
        step_counts = {}
        for step in self.steps:
            step_type = step["type"]
            step_counts[step_type] = step_counts.get(step_type, 0) + 1
        
        # Estimate total tokens
        total_token_estimate = 0
        for step in self.steps:
            # Rough estimate - in a real implementation you would use a tokenizer
            content = step["content"]
            if isinstance(content, str):
                total_token_estimate += len(content.split()) * 1.3  # Rough approximation
            elif isinstance(content, (dict, list)):
                json_str = json.dumps(content)
                total_token_estimate += len(json_str.split()) * 1.3
        
        return {
            "total_steps": len(self.steps),
            "step_counts": step_counts,
            "total_tool_calls": len(self.tool_calls),
            "token_estimate": int(total_token_estimate)
        }


class TokenEfficientIntermediateStorage(IntermediateStorageBase):
    """
    Token-optimized implementation of intermediate step storage.
    
    This implementation optimizes for token efficiency by:
    1. Keeping full details for the most recent steps
    2. Summarizing older steps to reduce token usage
    3. Dropping less important metadata
    """
    
    def __init__(self, max_detailed_steps: int = 5):
        """
        Initialize token-efficient storage.
        
        Args:
            max_detailed_steps: Number of most recent steps to keep in full detail
        """
        self.detailed_steps = []  # Most recent steps in full detail
        self.summarized_steps = []  # Older steps in summarized form
        self.tool_calls = []  # Tool calls
        self.max_detailed_steps = max_detailed_steps
    
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an intermediate step to storage with token optimization."""
        step = {
            "type": step_type,
            "content": content,
            "metadata": metadata or {}
        }
        
        # Add to detailed steps
        self.detailed_steps.append(step)
        
        # If we exceed the max detailed steps, move oldest to summarized
        if len(self.detailed_steps) > self.max_detailed_steps:
            oldest_step = self.detailed_steps.pop(0)
            self._summarize_step(oldest_step)
    
    def _summarize_step(self, step: Dict[str, Any]) -> None:
        """
        Summarize a step to reduce token usage.
        
        Args:
            step: The step to summarize
        """
        # Create a summarized version with reduced content
        summarized_content = self._compress_content(step["content"])
        
        # Keep only essential metadata
        essential_metadata = {}
        if "iteration" in step["metadata"]:
            essential_metadata["iteration"] = step["metadata"]["iteration"]
            
        # Add the summarized step
        self.summarized_steps.append({
            "type": step["type"],
            "content": summarized_content,
            "metadata": essential_metadata
        })
    
    def _compress_content(self, content: Any) -> str:
        """
        Compress content to reduce token usage.
        
        Args:
            content: Content to compress
            
        Returns:
            Compressed content
        """
        if isinstance(content, str):
            # For strings, truncate if too long
            if len(content) > 100:
                return content[:97] + "..."
            return content
            
        elif isinstance(content, (dict, list)):
            # For structured data, convert to a compact string
            try:
                json_str = json.dumps(content, separators=(',', ':'))
                if len(json_str) > 100:
                    return json_str[:97] + "..."
                return json_str
            except:
                return str(content)
                
        return str(content)
    
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a tool call result to storage with token optimization."""
        # Store only essential information about the tool call
        tool_call = {
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output
        }
        
        # Only keep essential metadata
        essential_metadata = {}
        if metadata:
            if "iteration" in metadata:
                essential_metadata["iteration"] = metadata["iteration"]
        
        self.tool_calls.append({
            "data": tool_call,
            "metadata": essential_metadata
        })
        
        # Also add as a step
        self.add_step(
            "tool_call", 
            {"tool_name": tool_name, "input": tool_input, "output": tool_output},
            metadata
        )
    
    def get_steps(self, step_types: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve stored intermediate steps with token-efficient optimization."""
        # Combine summarized and detailed steps
        all_steps = self.summarized_steps + self.detailed_steps
        
        # Filter by step type if specified
        if step_types:
            all_steps = [step for step in all_steps if step["type"] in step_types]
        
        # Apply limit if specified
        if limit is not None:
            all_steps = all_steps[-limit:]
            
        return all_steps
    
    def get_formatted_context(self, format_type: str = "default") -> str:
        """Get formatted context string optimized for token efficiency."""
        # For token efficiency, we'll use a compact format by default
        if format_type == "detailed":
            # Use compact format even for detailed view, but include more information
            all_steps = self.summarized_steps + self.detailed_steps
            formatted_steps = []
            
            for i, step in enumerate(all_steps):
                if step["type"] in ["thought", "action", "action_input", "observation", "tool_call"]:
                    # Add step information in a compact format
                    prefix = f"Step {i+1}"
                    if "iteration" in step["metadata"]:
                        prefix += f" (Iter {step['metadata']['iteration']})"
                    
                    if step["type"] == "thought":
                        formatted_steps.append(f"{prefix} Thought: {step['content']}")
                    elif step["type"] == "action":
                        formatted_steps.append(f"{prefix} Action: {step['content']}")
                    elif step["type"] == "action_input":
                        formatted_steps.append(f"{prefix} Input: {step['content']}")
                    elif step["type"] == "observation":
                        formatted_steps.append(f"{prefix} Observed: {step['content']}")
                    elif step["type"] == "tool_call":
                        tool_info = step["content"]
                        formatted_steps.append(
                            f"{prefix} Tool: {tool_info['tool_name']} → {tool_info['output']}"
                        )
            
            return "\n".join(formatted_steps)
        
        else:  # default or compact format - use the most token-efficient approach
            # Use ReAct style but focus only on the most recent detailed steps
            formatted_steps = []
            
            # Add brief summary of earlier steps if any
            if self.summarized_steps:
                summarized_count = len(self.summarized_steps)
                step_types = {}
                for step in self.summarized_steps:
                    step_type = step["type"]
                    step_types[step_type] = step_types.get(step_type, 0) + 1
                
                summary = f"Previous {summarized_count} steps: "
                summary += ", ".join([f"{count} {step_type}s" for step_type, count in step_types.items()])
                formatted_steps.append(summary)
            
            # Add detailed steps in ReAct format
            for step in self.detailed_steps:
                if step["type"] == "thought":
                    formatted_steps.append(f"Thought: {step['content']}")
                elif step["type"] == "action":
                    formatted_steps.append(f"Action: {step['content']}")
                elif step["type"] == "action_input":
                    formatted_steps.append(f"Action Input: {step['content']}")
                elif step["type"] == "observation" or step["type"] == "tool_call":
                    if step["type"] == "tool_call":
                        tool_info = step["content"]
                        formatted_steps.append(f"Observation: {tool_info['output']}")
                    else:
                        formatted_steps.append(f"Observation: {step['content']}")
            
            return "\n".join(formatted_steps)
    
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get stored steps as message objects, optimized for token efficiency."""
        messages = []
        
        # First, add a brief summary of earlier steps
        if self.summarized_steps:
            summary_message = "Earlier steps summary:\n"
            step_types = {}
            
            for step in self.summarized_steps:
                step_type = step["type"]
                step_types[step_type] = step_types.get(step_type, 0) + 1
            
            summary_message += ", ".join([f"{count} {step_type}s" for step_type, count in step_types.items()])
            
            # Add a few example summarized steps for context
            sample_size = min(3, len(self.summarized_steps))
            if sample_size > 0:
                summary_message += "\n\nSample of earlier steps:\n"
                
                for step in self.summarized_steps[-sample_size:]:
                    if step["type"] == "thought":
                        summary_message += f"Thought: {step['content']}\n"
                    elif step["type"] == "action":
                        summary_message += f"Action: {step['content']}\n"
                    elif step["type"] == "tool_call":
                        tool_info = step["content"]
                        summary_message += f"Tool {tool_info['tool_name']}: {...}\n"
            
            messages.append({
                "role": "system",
                "content": summary_message
            })
        
        # Add tool calls from the detailed steps
        for idx, step in enumerate(self.detailed_steps):
            if step["type"] == "tool_call":
                tool_info = step["content"]
                
                # Create unique ID
                call_id = f"call_{idx}"
                
                # Add assistant message with tool call
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_info["tool_name"],
                            "arguments": json.dumps(tool_info["input"])
                        }
                    }]
                })
                
                # Add tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(tool_info["output"])
                })
        
        # Add current context as system message
        current_context = []
        for step in self.detailed_steps:
            if step["type"] in ["thought", "action", "action_input", "observation"] and step["type"] != "tool_call":
                if step["type"] == "thought":
                    current_context.append(f"Thought: {step['content']}")
                elif step["type"] == "action":
                    current_context.append(f"Action: {step['content']}")
                elif step["type"] == "action_input":
                    current_context.append(f"Action Input: {step['content']}")
                elif step["type"] == "observation":
                    current_context.append(f"Observation: {step['content']}")
        
        if current_context:
            messages.append({
                "role": "system",
                "content": "\n".join(current_context)
            })
        
        return messages
    
    def clear(self) -> None:
        """Clear all stored intermediate steps."""
        self.detailed_steps = []
        self.summarized_steps = []
        self.tool_calls = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored steps with token optimization."""
        # Count steps by type and location (detailed vs summarized)
        detailed_counts = {}
        for step in self.detailed_steps:
            step_type = step["type"]
            detailed_counts[step_type] = detailed_counts.get(step_type, 0) + 1
            
        summarized_counts = {}
        for step in self.summarized_steps:
            step_type = step["type"]
            summarized_counts[step_type] = summarized_counts.get(step_type, 0) + 1
        
        # Estimate tokens (rough approximation)
        detailed_tokens = 0
        for step in self.detailed_steps:
            content = step["content"]
            if isinstance(content, str):
                detailed_tokens += len(content.split()) * 1.3
            elif isinstance(content, (dict, list)):
                json_str = json.dumps(content)
                detailed_tokens += len(json_str.split()) * 1.3
                
        summarized_tokens = 0
        for step in self.summarized_steps:
            content = step["content"]
            if isinstance(content, str):
                summarized_tokens += len(content.split()) * 1.3
            elif isinstance(content, (dict, list)):
                json_str = json.dumps(content)
                summarized_tokens += len(json_str.split()) * 1.3
        
        return {
            "total_steps": len(self.detailed_steps) + len(self.summarized_steps),
            "detailed_steps": len(self.detailed_steps),
            "summarized_steps": len(self.summarized_steps),
            "detailed_counts": detailed_counts,
            "summarized_counts": summarized_counts,
            "total_tool_calls": len(self.tool_calls),
            "token_estimate": {
                "detailed": int(detailed_tokens),
                "summarized": int(summarized_tokens),
                "total": int(detailed_tokens + summarized_tokens)
            },
            "token_savings_estimate": int((detailed_tokens + summarized_tokens) * 0.5)  # Rough estimate of tokens saved
        } 