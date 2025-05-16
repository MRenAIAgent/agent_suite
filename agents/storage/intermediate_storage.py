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
        stats = {
            "total_steps": len(self.steps),
            "total_tool_calls": len(self.tool_calls),
            "steps_by_type": {}
        }
        
        # Count steps by type
        for step in self.steps:
            step_type = step["type"]
            if step_type not in stats["steps_by_type"]:
                stats["steps_by_type"][step_type] = 0
            stats["steps_by_type"][step_type] += 1
        
        return stats


class TokenEfficientIntermediateStorage(IntermediateStorageBase):
    """
    Token-optimized implementation of intermediate step storage.
    
    This implementation focuses on reducing token usage by:
    1. Summarizing/compressing older steps
    2. Only keeping the most relevant information
    3. Using compact representations
    """
    
    def __init__(self, max_detailed_steps: int = 5):
        self.steps = []  # Full detailed steps (recent)
        self.tool_calls = []  # All tool calls
        self.summarized_steps = []  # Summarized older steps
        self.max_detailed_steps = max_detailed_steps
    
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an intermediate step to storage."""
        step = {
            "type": step_type,
            "content": content,
            "metadata": metadata or {}
        }
        
        # Add the new step
        self.steps.append(step)
        
        # If we exceed our detailed steps limit, summarize the oldest one
        if len(self.steps) > self.max_detailed_steps:
            oldest_step = self.steps.pop(0)
            self._summarize_step(oldest_step)
    
    def _summarize_step(self, step: Dict[str, Any]) -> None:
        """Convert a detailed step into a summarized form to save tokens."""
        # Skip summarization for certain critical step types
        if step["type"] in ["final_answer"]:
            return
            
        # Create a summarized version (less detailed)
        summary = {
            "type": step["type"],
            # Just keep the step type and a compressed indicator of content
            "summary": f"{step['type']}: {self._compress_content(step['content'])}"
        }
        
        self.summarized_steps.append(summary)
    
    def _compress_content(self, content: Any) -> str:
        """Compress content to reduce token usage."""
        if isinstance(content, str):
            # For strings, truncate if too long
            if len(content) > 100:
                return content[:97] + "..."
            return content
        elif isinstance(content, (dict, list)):
            # For structured data, just indicate type and size
            if isinstance(content, dict):
                return f"<dict with {len(content)} keys>"
            else:
                return f"<list with {len(content)} items>"
        else:
            return str(content)
    
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a tool call result to storage."""
        # Store minimal representation of tool calls to save tokens
        tool_call = {
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output,
            # Only keep essential metadata
            "metadata": {k: v for k, v in (metadata or {}).items() if k in ["timestamp", "latency"]}
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
        # Combine summarized and detailed steps
        all_steps = self.summarized_steps + self.steps
        
        if step_types:
            all_steps = [step for step in all_steps if step["type"] in step_types]
        
        if limit is not None:
            all_steps = all_steps[-limit:]
            
        return all_steps
    
    def get_formatted_context(self, format_type: str = "default") -> str:
        """Get formatted context string for inclusion in LLM prompts."""
        # Start with the summarized steps
        formatted_steps = []
        
        if self.summarized_steps:
            formatted_steps.append("Previous steps (summarized):")
            for summary in self.summarized_steps:
                formatted_steps.append(f"- {summary['summary']}")
        
        # Then add the detailed recent steps
        formatted_steps.append("\nRecent steps:")
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
                    # Use a compact representation of tool output
                    output_str = self._compress_content(tool_info['output'])
                    formatted_steps.append(f"Tool '{tool_info['tool_name']}' returned: {output_str}")
                else:
                    formatted_steps.append(f"Observation: {step['content']}")
        
        return "\n".join(formatted_steps)
    
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get stored steps as message objects for LLM context."""
        messages = []
        
        # Include a summary of earlier steps as a system message
        if self.summarized_steps:
            summary_content = "Previous steps (summarized):\n"
            summary_content += "\n".join([f"- {summary['summary']}" for summary in self.summarized_steps])
            messages.append({
                "role": "system",
                "content": summary_content
            })
        
        # Add recent tool calls - these are more important to keep complete
        recent_tool_calls = self.tool_calls[-self.max_detailed_steps:] if self.tool_calls else []
        
        for i, tool_call in enumerate(recent_tool_calls):
            call_id = f"call_{i}"
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
            
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(tool_call["output"])
            })
        
        # Add detailed non-tool steps as a single system message
        detailed_steps = []
        for step in self.steps:
            if step["type"] in ["thought", "action", "action_input", "observation"] and step["type"] != "tool_call":
                if step["type"] == "thought":
                    detailed_steps.append(f"Thought: {step['content']}")
                elif step["type"] == "action":
                    detailed_steps.append(f"Action: {step['content']}")
                elif step["type"] == "action_input":
                    detailed_steps.append(f"Action Input: {step['content']}")
                elif step["type"] == "observation":
                    detailed_steps.append(f"Observation: {step['content']}")
        
        if detailed_steps:
            messages.append({
                "role": "system",
                "content": "\n".join(detailed_steps)
            })
        
        return messages
    
    def clear(self) -> None:
        """Clear all stored intermediate steps."""
        self.steps = []
        self.summarized_steps = []
        self.tool_calls = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored steps."""
        return {
            "detailed_steps": len(self.steps),
            "summarized_steps": len(self.summarized_steps),
            "total_tool_calls": len(self.tool_calls),
            "compression_ratio": len(self.summarized_steps) / (len(self.steps) + len(self.summarized_steps)) if (len(self.steps) + len(self.summarized_steps)) > 0 else 0
        } 