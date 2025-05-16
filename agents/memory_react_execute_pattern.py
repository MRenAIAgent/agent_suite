from typing import Dict, Any, List, Optional
import json

from agents.llm_execute_pattern import LLMExecutionPattern
from agents.storage.intermediate_storage import InMemoryIntermediateStorage


class MemoryReActLLMExecutionPattern(LLMExecutionPattern):
    """
    ReAct execution pattern that uses InMemoryIntermediateStorage for storing steps.
    
    This pattern extends the base pattern by using the InMemoryIntermediateStorage
    to track and manage execution steps.
    """
    
    def __init__(self):
        """Initialize with InMemoryIntermediateStorage."""
        self.storage = InMemoryIntermediateStorage()
        self.current_iteration = 0
    
    def parse_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw LLM response into structured format and store it."""
        # Initialize parsed response
        parsed_response = {}
        
        # Get the message content
        message = response.choices[0].message
        
        # Handle tool calls if present in the response
        if hasattr(message, 'tool_calls') and message.tool_calls:
            parsed_response["tool_calls"] = message.tool_calls
            return parsed_response
        
        # If response is an object with content attribute, get the content
        if hasattr(message, 'content'):
            content = message.content
        else:
            content = message
        
        # Split the response into lines for parsing
        lines = content.strip().split('\n')
        
        current_key = None
        current_value = []
        unstructured_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for thought pattern
            if line.startswith("Thought:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step
                    self._store_step(current_key, parsed_response[current_key])
                
                current_key = "thought"
                current_value = [line[len("Thought:"):].strip()]
            
            # Check for action pattern
            elif line.startswith("Action:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step
                    self._store_step(current_key, parsed_response[current_key])
                
                current_key = "action"
                current_value = [line[len("Action:"):].strip()]
            
            # Check for action input pattern
            elif line.startswith("Action Input:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step
                    self._store_step(current_key, parsed_response[current_key])
                
                current_key = "action_input"
                current_value = [line[len("Action Input:"):].strip()]
            
            # Check for observation pattern
            elif line.startswith("Observation:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step
                    self._store_step(current_key, parsed_response[current_key])
                
                current_key = "observation"
                current_value = [line[len("Observation:"):].strip()]
            
            # Check for final answer pattern
            elif line.startswith("[FINAL ANSWER]") or line.startswith("Final Answer:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step
                    self._store_step(current_key, parsed_response[current_key])
                
                current_key = "final_answer"
                if line.startswith("[FINAL ANSWER]"):
                    current_value = [line[len("[FINAL ANSWER]"):].strip()]
                else:
                    current_value = [line[len("Final Answer:"):].strip()]
            
            # Continue building the current value
            else:
                if current_key:
                    current_value.append(line)
                else:
                    unstructured_lines.append(line)
        
        # Add the last key-value pair
        if current_key and current_value:
            parsed_response[current_key] = '\n'.join(current_value).strip()
            # Store the final step
            self._store_step(current_key, parsed_response[current_key])
        
        # If no structured format detected, treat as final answer
        if not parsed_response:
            final_answer = '\n'.join(unstructured_lines).strip()
            parsed_response["final_answer"] = final_answer
            self.storage.add_step("final_answer", final_answer)
        
        return parsed_response
    
    def _store_step(self, step_type: str, content: Any) -> None:
        """Store a step in the InMemoryIntermediateStorage."""
        metadata = {"iteration": self.current_iteration}
        self.storage.add_step(step_type, content, metadata)
    
    def should_continue(self, parsed_response: Dict[str, Any]) -> bool:
        """Determine if agent should continue the invocation loop."""
        # Continue if there's an action but no final answer
        return "action" in parsed_response and "final_answer" not in parsed_response
    
    def get_final_answer(self, parsed_response: Dict[str, Any]) -> str:
        """Extract final answer when loop is complete."""
        if "final_answer" in parsed_response:
            return parsed_response["final_answer"]
        return "I couldn't determine a final answer."
    
    def format_intermediate_steps(self, parsed_response: Dict[str, Any]) -> str:
        """Format intermediate reasoning/action steps using storage."""
        return self.storage.get_formatted_context(format_type="default")
    
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get intermediate steps formatted as messages for LLM context."""
        return self.storage.get_as_messages(max_tokens=max_tokens)
    
    def handle_tool_results(self, tool_calls: List[Any], tool_results: List[Any]) -> None:
        """Store tool call results in storage."""
        for i, tool_call in enumerate(tool_calls):
            if i < len(tool_results):
                tool_name = tool_call.function.name if hasattr(tool_call.function, 'name') else "unknown_tool"
                tool_input = tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else "{}"
                
                # Try to parse the arguments as JSON
                try:
                    if isinstance(tool_input, str):
                        tool_input = json.loads(tool_input)
                except (json.JSONDecodeError, TypeError):
                    # Keep as string if parsing fails
                    pass
                
                self.storage.add_tool_call(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=tool_results[i],
                    metadata={"iteration": self.current_iteration}
                )
    
    def start_new_iteration(self) -> None:
        """Start a new iteration."""
        self.current_iteration += 1
        self.storage.add_step(
            "iteration_start",
            f"Iteration {self.current_iteration}",
            {"iteration": self.current_iteration}
        )
    
    def start_execution(self) -> None:
        """Start a new execution, resetting storage."""
        self.storage.clear()
        self.current_iteration = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the execution."""
        return self.storage.get_stats() 