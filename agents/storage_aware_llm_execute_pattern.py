from typing import Dict, Any, Optional, List
import json

from agents.llm_execute_pattern import LLMExecutionPattern
from agents.storage.storage_manager import IntermediateStepManager


class StorageAwareLLMExecutionPattern(LLMExecutionPattern):
    """
    LLM Execution Pattern that uses the IntermediateStepManager for memory.
    
    This pattern extends the base execution pattern by integrating with
    the IntermediateStepManager for tracking and managing execution steps.
    """
    
    def __init__(
        self, 
        step_manager: Optional[IntermediateStepManager] = None,
        token_efficient: bool = False
    ):
        """
        Initialize the execution pattern with storage management.
        
        Args:
            step_manager: Optional existing step manager to use
            token_efficient: Whether to use token-efficient storage (if creating a new manager)
        """
        self.step_manager = step_manager or IntermediateStepManager(token_efficient=token_efficient)
        
    def parse_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw LLM response into structured format and store it.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Dict containing parsed components
        """
        # Implement basic parsing logic that works for most agent patterns
        parsed_response = {}
        message = response.choices[0].message
        
        # Handle tool calls if present in the response
        if hasattr(message, 'tool_calls') and message.tool_calls:
            parsed_response["tool_calls"] = message.tool_calls
            return parsed_response
            
        # Get content from the message
        if hasattr(message, 'content'):
            content = message.content
        else:
            content = message
            
        # Try to parse the response as JSON first (for structured agents)
        try:
            json_content = json.loads(content)
            # If successful, check for common keys
            for key in ["thought", "action", "action_input", "observation", "final_answer"]:
                if key in json_content:
                    parsed_response[key] = json_content[key]
                    # Store in step manager if present
                    if key == "thought" and "thought" in json_content:
                        self.step_manager.add_thought(json_content["thought"])
                    elif key == "action" and "action" in json_content:
                        action_input = json_content.get("action_input", None)
                        self.step_manager.add_action(json_content["action"], action_input)
                    elif key == "observation" and "observation" in json_content:
                        self.step_manager.add_observation(json_content["observation"])
                    elif key == "final_answer" and "final_answer" in json_content:
                        self.step_manager.add_final_answer(json_content["final_answer"])
            
            # If we successfully parsed JSON but found no expected keys, store as-is
            if not parsed_response:
                parsed_response = json_content
                
            return parsed_response
        except (json.JSONDecodeError, TypeError):
            # Not JSON, continue with text parsing
            pass
        
        # Text-based parsing for ReAct format
        lines = content.strip().split('\n')
        
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for thought pattern
            if line.startswith("Thought:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step if needed
                    self._store_current_step(current_key, parsed_response[current_key])
                
                current_key = "thought"
                current_value = [line[len("Thought:"):].strip()]
                
            # Check for action pattern
            elif line.startswith("Action:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step if needed
                    self._store_current_step(current_key, parsed_response[current_key])
                
                current_key = "action"
                current_value = [line[len("Action:"):].strip()]
                
            # Check for action input pattern
            elif line.startswith("Action Input:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step if needed
                    self._store_current_step(current_key, parsed_response[current_key])
                
                current_key = "action_input"
                current_value = [line[len("Action Input:"):].strip()]
                
            # Check for observation pattern
            elif line.startswith("Observation:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step if needed
                    self._store_current_step(current_key, parsed_response[current_key])
                
                current_key = "observation"
                current_value = [line[len("Observation:"):].strip()]
                
            # Check for final answer pattern
            elif line.startswith("[FINAL ANSWER]") or line.startswith("Final Answer:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                    # Store the previous step if needed
                    self._store_current_step(current_key, parsed_response[current_key])
                
                current_key = "final_answer"
                if line.startswith("[FINAL ANSWER]"):
                    current_value = [line[len("[FINAL ANSWER]"):].strip()]
                else:
                    current_value = [line[len("Final Answer:"):].strip()]
                
            # Continue building the current value
            else:
                if current_key:
                    current_value.append(line)
        
        # Add the last key-value pair
        if current_key and current_value:
            parsed_response[current_key] = '\n'.join(current_value).strip()
            # Store the final step
            self._store_current_step(current_key, parsed_response[current_key])
        
        # If no structured format was detected, store as final answer
        if not parsed_response:
            parsed_response["final_answer"] = content.strip()
            self.step_manager.add_final_answer(content.strip())
            
        return parsed_response
    
    def _store_current_step(self, step_type: str, content: Any):
        """Store a parsed step in the step manager."""
        if step_type == "thought":
            self.step_manager.add_thought(content)
        elif step_type == "action":
            # Action input will be added separately later if present
            self.step_manager.add_action(content)
        elif step_type == "action_input":
            # This would be connected to the most recent action
            pass  # Handled together with action
        elif step_type == "observation":
            self.step_manager.add_observation(content)
        elif step_type == "final_answer":
            self.step_manager.add_final_answer(content)
    
    def should_continue(self, parsed_response: Dict[str, Any]) -> bool:
        """
        Determine if agent should continue the invocation loop.
        
        Args:
            parsed_response: Parsed response from LLM
            
        Returns:
            bool: True if should continue, False if should break loop
        """
        # Continue if there are tool calls or an action but no final answer
        if "tool_calls" in parsed_response:
            return True
        
        if "action" in parsed_response and "final_answer" not in parsed_response:
            return True
            
        return False
    
    def get_final_answer(self, parsed_response: Dict[str, Any]) -> str:
        """
        Extract final answer when loop is complete.
        
        Args:
            parsed_response: Parsed response containing final answer
            
        Returns:
            str: The final answer to return to user
        """
        if "final_answer" in parsed_response:
            return parsed_response["final_answer"]
        
        # If no final answer but we have an observation, that might be the answer
        if "observation" in parsed_response:
            return parsed_response["observation"]
            
        return "I couldn't determine a final answer."
    
    def format_intermediate_steps(self, parsed_response: Dict[str, Any]) -> str:
        """
        Format intermediate reasoning/action steps.
        
        Args:
            parsed_response: Parsed response with intermediate steps
            
        Returns:
            str: Formatted intermediate steps to add to conversation
        """
        # Use the step manager to generate formatted context
        return self.step_manager.get_formatted_context(format_type="default")
    
    def get_intermediate_steps_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get intermediate steps formatted as messages for LLM context.
        
        Args:
            max_tokens: Optional token limit
            
        Returns:
            List of messages to include in LLM context
        """
        return self.step_manager.get_as_messages(max_tokens=max_tokens)
    
    def handle_tool_results(self, tool_calls: List[Any], tool_results: List[Any]) -> None:
        """
        Store tool call results in the step manager.
        
        Args:
            tool_calls: List of tool calls
            tool_results: List of corresponding results
        """
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
                
                self.step_manager.add_tool_result(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=tool_results[i]
                )
    
    def start_new_iteration(self):
        """Start a new iteration in the step manager."""
        self.step_manager.start_iteration()
    
    def start_execution(self):
        """Start a new execution run, resetting the step manager."""
        self.step_manager.start_execution()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the current execution.
        
        Returns:
            Dictionary with execution summary
        """
        return self.step_manager.get_execution_summary()


class StorageAwareReActExecutionPattern(StorageAwareLLMExecutionPattern):
    """
    ReAct-specific implementation of the storage-aware execution pattern.
    
    Extends the base storage-aware pattern with ReAct-specific behaviors.
    """
    
    def should_continue(self, parsed_response: Dict[str, Any]) -> bool:
        """ReAct-specific continuation logic."""
        # Continue if there's an action but no final answer
        return "action" in parsed_response and "final_answer" not in parsed_response
    
    def format_intermediate_steps(self, parsed_response: Dict[str, Any]) -> str:
        """ReAct-specific formatting of intermediate steps."""
        steps = []
        
        if "thought" in parsed_response:
            steps.append(f"Thought: {parsed_response['thought']}")
            
        if "action" in parsed_response:
            steps.append(f"Action: {parsed_response['action']}")
            
        if "action_input" in parsed_response:
            steps.append(f"Action Input: {parsed_response['action_input']}")
            
        if "observation" in parsed_response:
            steps.append(f"Observation: {parsed_response['observation']}")
            
        return "\n".join(steps) 