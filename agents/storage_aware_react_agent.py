import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

from agents.base_classes.base_agent import BaseAgent
from agents.memory_manager import MemoryManager
from agents.prompt import PromptManager
from agents.react_prompt_template import ReActPromptTemplate
from agents.storage_aware_agent_pattern import (
    StorageAwareReActAgentPattern,
    create_token_efficient_react_pattern,
    create_in_memory_react_pattern
)
from llm.llm import LLMBase
from log.logging import LogManager
from tools.tool import Tool


class StorageAwareReActAgent(BaseAgent):
    """
    ReAct agent with optimized intermediate step storage.
    
    This agent uses the storage-aware patterns to improve performance
    and reduce token usage by intelligently managing intermediate reasoning steps.
    """

    def __init__(
            self,
            llm: LLMBase,
            role: str,
            task: str,
            guide: str,
            examples: list[str] = None,
            tools: List[Tool] = None,
            log_manager: LogManager = None,
            memory_manager: MemoryManager = None,
            max_iterations: int = 5,
            token_efficient: bool = False,
            max_detailed_steps: int = 5):
        """
        Initialize the StorageAwareReActAgent.
        
        Args:
            llm: LLM instance for generating responses
            role: Agent role description
            task: Task description
            guide: Guide for the agent
            examples: Optional examples to include in the prompt
            tools: List of available tools
            log_manager: Optional log manager
            memory_manager: Optional memory manager
            max_iterations: Maximum number of iterations
            token_efficient: Whether to use token-efficient storage
            max_detailed_steps: Maximum number of detailed steps to preserve (when token_efficient=True)
        """
        # Create the prompt template
        react_prompt_template = ReActPromptTemplate(
            role=role,
            task=task,
            guide=guide,
            examples=examples
        )
        system_prompt = react_prompt_template.format_prompt()
        prompt_manager = PromptManager(system_prompt)
        
        # Choose the appropriate agent pattern based on configuration
        if token_efficient:
            self.agent_pattern = create_token_efficient_react_pattern(
                max_detailed_steps=max_detailed_steps,
                prompt_template=react_prompt_template
            )
        else:
            self.agent_pattern = create_in_memory_react_pattern(
                prompt_template=react_prompt_template
            )
        
        # Get the execution pattern from the agent pattern
        self.execution_pattern = self.agent_pattern.llm_execution_pattern
        
        # Initialize other components
        self.log_manager = log_manager
        self.max_iterations = max_iterations
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.memory_manager = memory_manager if memory_manager else MemoryManager()
        self.tools = tools or []
        self.metadata = {}
        self.iterations = 0
        
        # Start the execution tracking
        self.execution_pattern.start_execution()

    async def arun(self, user_input: str, model: str, **kwargs) -> str:
        """
        Process user input using ReAct approach with optimized storage.
        
        Args:
            user_input: The user's question or request
            model: The LLM model to use
            **kwargs: Additional parameters
            
        Returns:
            str: The agent's final response
        """
        GREEN = "\033[92m"
        RESET = "\033[0m"
        
        def pretty_json(obj) -> str:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        
        # Reset execution tracking for a new run
        self.execution_pattern.start_execution()
        
        # Log initial setup
        if self.log_manager:
            self.log_manager.start_execution()
            
        # Get initial messages
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        
        if self.log_manager:
            self.log_manager.log_debug("# Initial Messages\n```json\n" + pretty_json(messages) + "\n```")
        
        iterations = 0
        self.iterations = 0  # Store for external access
        
        try:
            while iterations < self.max_iterations:
                # Start a new iteration
                self.execution_pattern.start_new_iteration()
                iterations += 1
                self.iterations = iterations
                
                if self.log_manager:
                    self.log_manager.log_debug(f"\n## ReAct Iteration {iterations}/{self.max_iterations}")
                
                # Measure LLM latency
                llm_start_time = time.time()
                response = self.llm.chat_completion(
                    model=model,
                    messages=messages,
                    tools=[tool.convert_to_function_call() for tool in self.tools],
                    tool_choice="auto" if self.tools else None
                )
                llm_latency = time.time() - llm_start_time
                
                # Log LLM response and latency
                if self.log_manager:
                    self.log_manager.log_llm_latency(llm_latency)
                    self.log_manager.log_debug(f"\n### LLM Response {GREEN}\n```json\n{pretty_json(response)}\n```{RESET}")
                
                # Parse the response using our execution pattern
                parsed_response = self.execution_pattern.parse_llm_response(response)
                
                if self.log_manager:
                    self.log_manager.log_debug(f"\n### Parsed Response {GREEN}\n```json\n{pretty_json(parsed_response)}\n```{RESET}")
                
                # Check if we reached the final iteration
                if iterations >= self.max_iterations:
                    final_answer = self.execution_pattern.get_final_answer(parsed_response)
                    if self.log_manager:
                        self.log_manager.log_debug(f"\n### Final Answer (max iterations) {GREEN}\n```\n{final_answer}\n```{RESET}")
                        self.log_manager.log_final_answer(final_answer)
                    return final_answer
                
                # Handle tool calls if present
                if "tool_calls" in parsed_response:
                    if self.log_manager:
                        self.log_manager.log_debug("\n### Executing Tool Calls")
                    
                    # For each tool call
                    for i, tool_call in enumerate(parsed_response["tool_calls"]):
                        if hasattr(tool_call.function, 'name'):
                            tool_name = tool_call.function.name
                            tool_args = tool_call.function.arguments
                            
                            # Log the action
                            if self.log_manager:
                                self.log_manager.log_action(tool_name, tool_args)
                    
                    # Execute tools and get results
                    tool_results = await self.handle_tool_calls(parsed_response["tool_calls"])
                    
                    # Log each result
                    if self.log_manager:
                        for i, result in enumerate(tool_results):
                            self.log_manager.log_result(i + 1, result)
                        self.log_manager.log_debug("\n### Tool Results\n```json\n" + pretty_json(tool_results) + "\n```")
                    
                    # Store tool results in our step manager
                    self.execution_pattern.handle_tool_results(
                        parsed_response["tool_calls"],
                        tool_results
                    )
                    
                    # Format tool calls for message context
                    formatted_tool_calls = []
                    for i, tool_call in enumerate(parsed_response["tool_calls"]):
                        formatted_tool_calls.append({
                            "id": tool_call.id if hasattr(tool_call, 'id') else f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                    
                    # Add assistant's tool calls to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": formatted_tool_calls
                    })
                    
                    # Add tool results to messages
                    for i, result in enumerate(tool_results):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": formatted_tool_calls[i]["id"],
                            "content": json.dumps(result)
                        })
                    
                    # Continue with the next iteration after tool calls
                    continue
                
                # Check if we should continue or return a final answer
                if self.execution_pattern.should_continue(parsed_response):
                    # Format intermediate steps for inclusion in the next prompt
                    formatted_steps = self.execution_pattern.format_intermediate_steps(parsed_response)
                    
                    if formatted_steps:
                        messages.append({
                            "role": "system", 
                            "content": formatted_steps
                        })
                        
                        if self.log_manager:
                            self.log_manager.log_debug(f"\n### Intermediate Steps {GREEN}\n```\n{formatted_steps}\n```{RESET}")
                else:
                    # Get the final answer and end execution
                    final_answer = self.execution_pattern.get_final_answer(parsed_response)
                    
                    if self.log_manager:
                        self.log_manager.log_debug(f"\n### Final Answer {GREEN}\n```\n{final_answer}\n```{RESET}")
                        self.log_manager.log_final_answer(final_answer)
                    
                    # Update conversation history for future interactions
                    self.memory_manager.add({"role": "user", "content": user_input})
                    self.memory_manager.add({"role": "assistant", "content": final_answer})
                    
                    # Log interaction summary
                    if self.log_manager:
                        self.log_manager.log_interaction(
                            user_input=user_input,
                            agent_response=final_answer,
                            model=model,
                            timestamp=datetime.now().isoformat()
                        )
                    
                    return final_answer
            
            # If we reach here, we've hit max iterations without a final answer
            final_answer = "I couldn't determine a final answer within the allowed iterations."
            
            if self.log_manager:
                self.log_manager.log_debug(f"\n### Max Iterations Reached {GREEN}\n```\n{final_answer}\n```{RESET}")
                self.log_manager.log_final_answer(final_answer)
            
            return final_answer
            
        except Exception as e:
            if self.log_manager:
                self.log_manager.log_debug("\n### Error\n```\n" + str(e) + "\n```")
            raise e
        finally:
            # End execution tracking and logging
            if self.log_manager:
                self.log_manager.end_execution()
                self.log_manager.log_debug("\n## Execution Complete")
                
                # Get and log execution summary
                summary = self.execution_pattern.get_execution_summary()
                self.log_manager.log_debug("\n### Execution Summary\n```json\n" + pretty_json(summary) + "\n```")

    def run(self, user_input: str, model: str, **kwargs) -> str:
        """Run the agent synchronously."""
        return asyncio.run(self.arun(user_input, model, **kwargs))
    
    async def think(self, user_input: str, model: str) -> Dict[str, Any]:
        """Generate agent thoughts based on user input."""
        # Get messages for the prompt
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        
        # Get response from LLM
        response = self.llm.chat_completion(
            model=model,
            messages=messages,
            tools=[tool.convert_to_function_call() for tool in self.tools],
            tool_choice="auto" if self.tools else None
        )
        
        # Parse the response
        parsed_response = self.execution_pattern.parse_llm_response(response)
        
        # Format thoughts
        thoughts = {
            "input": user_input,
            "reasoning": parsed_response.get("thought", ""),
            "observation": parsed_response.get("observation", ""),
            "plan": parsed_response.get("action", ""),
            "final_answer": parsed_response.get("final_answer", "")
        }
        
        return thoughts
    
    async def handle_tool_calls(self, tool_calls):
        """Execute multiple tool calls and return results."""
        results = []
        
        for tool_call in tool_calls:
            if hasattr(tool_call.function, 'name') and hasattr(tool_call.function, 'arguments'):
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                
                # Parse arguments
                try:
                    if isinstance(tool_args_str, str):
                        tool_args = json.loads(tool_args_str)
                    else:
                        tool_args = tool_args_str
                except json.JSONDecodeError:
                    tool_args = {"error": "Failed to parse arguments"}
                
                # Execute the tool
                result = await self.handle_single_tool_call(tool_name, tool_args)
                results.append(result)
            else:
                results.append({"error": "Invalid tool call format"})
        
        return results
    
    async def handle_single_tool_call(self, tool_name: str, tool_input: Any) -> Any:
        """Execute a single tool call."""
        # Find the tool by name
        tool = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            if self.log_manager:
                self.log_manager.log_debug(f"Tool '{tool_name}' not found")
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            # Execute the tool
            result = await tool.arun(**tool_input)
            return result
        except Exception as e:
            if self.log_manager:
                self.log_manager.log_debug(f"Error executing tool '{tool_name}': {str(e)}")
            return {"error": f"Error executing tool: {str(e)}"}
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the agent."""
        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                self.tools.pop(i)
                return True
        return False
    
    async def save_memory(self, context: str) -> None:
        """Save the agent's memory."""
        await self.memory_manager.save_memory(context)
    
    async def load_memory(self) -> None:
        """Load the agent's memory."""
        await self.memory_manager.load_memory()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current execution."""
        return self.execution_pattern.get_execution_summary() 