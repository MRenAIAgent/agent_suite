from typing import Dict, Any
from agents.llm_execute_pattern import LLMExecutionPattern


class ReActLLMExecutionPattern(LLMExecutionPattern):
    """ReAct agent pattern that implements thought-action-observation cycle."""

    def parse_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw LLM response into structured format."""
        # Implement ReAct parsing logic here
        parsed_response = {}
        message = response.choices[0].message
        # Handle tool_calls if present in the response object
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
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for thought pattern
            if line.startswith("Thought:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                current_key = "thought"
                current_value = [line[len("Thought:"):].strip()]
                
            # Check for action pattern
            elif line.startswith("Action:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                current_key = "action"
                current_value = [line[len("Action:"):].strip()]
                
            # Check for action input pattern
            elif line.startswith("Action Input:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                current_key = "action_input"
                current_value = [line[len("Action Input:"):].strip()]
                
            # Check for observation pattern
            elif line.startswith("Observation:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
                current_key = "observation"
                current_value = [line[len("Observation:"):].strip()]
                
            # Check for final answer pattern
            elif line.startswith("[FINAL ANSWER]") or line.startswith("Final Answer:"):
                if current_key and current_value:
                    parsed_response[current_key] = '\n'.join(current_value).strip()
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
        return parsed_response

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
        """Format intermediate reasoning/action steps."""
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

