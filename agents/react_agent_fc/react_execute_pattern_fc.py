from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

import json
import uuid


class ReActFunction(BaseModel):
    """The function define llm response format, how it will be executed from the llm response
    React agent ask llm response with thought, action, action_input, observation, final_answer
    ReAct agent pattern
    """

    thought: str = Field(description="The thought of the agent")
    # action: Optional[str] = Field(description="The action of the agent")
    # action_input: Optional[str] = Field(description="The input of the action")
    observation: Optional[str] = Field(description="The observation of the action")
    final_answer: Optional[str] = Field(description="The final answer of the agent")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(description="The tool calls of the agent")
    action: Optional[str] = Field(description="The tool of the agent")
    action_input: Optional[Dict[str, Any]] = Field(description="The input of the tool")

    def parse_response(self, message):
        """Parse the LLM response message and update this ReActFunction instance.
        
        Args:
            message: The message object from the LLM response
        """
        # Check if the message has tool_calls5
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "reactfunction":
                    # Parse the function arguments
                    try:
                        args_dict = json.loads(tool_call.function.arguments)
                        # Update this instance with the values from the arguments
                        for key, value in args_dict.items():
                            if hasattr(self, key):
                                if value is not None:
                                    setattr(self, key, value)
                        return
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse function arguments: {str(e)}")
        
        # If we get here, check for function_call (older API format)
        if hasattr(message, 'function_call') and message.function_call:
            if message.function_call.name == "reactfunction":
                try:
                    args_dict = json.loads(message.function_call.arguments)
                    # Update this instance with the values from the arguments
                    for key, value in args_dict.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                    return
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse function arguments: {str(e)}")
        
        # If no function call was found, try to get data from the content
        if hasattr(message, 'content') and message.content:
            # This is a fallback for when the model doesn't use function calling
            # You might want to implement text parsing here if needed
            pass

    def should_continue(self) -> bool:
        """Determine if agent should continue the invocation loop."""
        # Continue if there's an action but no final answer
        return self.action and self.final_answer is not None

    def get_final_answer(self) -> str:
        """Extract final answer when loop is complete."""
        default_answer = "I couldn't determine a final answer."
        return self.final_answer or default_answer
    
    def get_observation(self) -> str:
        """Extract observation when loop is complete."""
        default_observation = "I couldn't determine an observation."
        return self.observation or default_observation
    
    def get_tool_call(self) -> str:
        """Get the tool call."""
        # if isinstance(self.tool_calls, list):
        #     self.action = self.tool_calls[0]["recipient_name"].replace('functions.', '')
        #     self.action_input = self.tool_calls[0]["parameters"]
        #     return self.action, self.action_input
        # elif isinstance(self.tool_calls, dict):
        #     self.action = self.tool_calls["recipient_name"].replace('functions.', '')
        #     self.action_input = self.tool_calls["parameters"]
        #     return self.action, self.action_input
        # else:
        #     return None, {}
        return self.action, self.action_input

    def format_intermediate_steps(self) -> str:
        """Format intermediate reasoning/action steps."""
        steps = []
        if self.thought:
            steps.append(f"Thought: {self.thought}")
            
        if self.action:
            action = str(self.action).replace('functions.', '')
            steps.append(f"Action: {action}")
            
        if self.action_input:
            steps.append(f"Action Input: {self.action_input}")
            
        if self.observation:
            steps.append(f"Observation: {self.observation}")
        return "\n".join(steps)

    def format_tool_call(self):
        """Format the tool call for the messages.
        
        Returns:
            list: Formatted tool calls
        """
        # Generate a shorter ID that stays within OpenAI's 40 character limit
        # Use a shorter prefix and only part of the UUID
        short_id = f"call_{uuid.uuid4().hex[:32]}"[:40]
        if isinstance(self.action, str):
            action_name = str(self.action).replace('functions.', '')
        else:
            action_name = self.action  # or handle the case where action is not a string
        if self.action:
            return [{
                "id": short_id,
                "type": "function",
            "function": {
                "name": action_name,
                "arguments": json.dumps(self.action_input) if isinstance(self.action_input, dict) else self.action_input
            }
        }]

    def format_observation(self) -> str:
        """Format the observation."""
        return self.observation
