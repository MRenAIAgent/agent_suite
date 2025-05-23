from abc import ABC
from datetime import datetime
import asyncio
from typing import List, Any, Optional, Dict, Tuple, Union, Callable
import json
import uuid

from agents.base_classes.base_agent import BaseAgent
from agents.base_classes.base_think_pattern import AgentThinkPattern
from agents.cache import CacheManager
from agents.memory.memory_manager import MemoryManager
from agents.prompt import PromptManager
from agents.agent_pattern import AgentPattern
from agents.llm_execute_pattern import LLMExecutionPattern
from log.logging import LogManager
from llm.llm import LLMBase
from tools.tool import Tool
from openai.types.chat import ChatCompletionMessageParam
from llm.function_call_handler import OutputParser

class BasicExecutionPattern(LLMExecutionPattern):
    """Basic execution pattern for the standard Agent class.
    
    Implements basic functionality for parsing LLM responses,
    determining when to continue/stop execution loop, and formatting steps.
    """
    
    def parse_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw LLM response into structured format."""
        parsed_response = {}
        message = response.choices[0].message
        
        # Handle tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            parsed_response["tool_calls"] = message.tool_calls
            return parsed_response
            
        # Handle regular content
        if hasattr(message, 'content'):
            content = message.content
            parsed_response["content"] = content
            parsed_response["final_answer"] = content
            
        return parsed_response
        
    def should_continue(self, parsed_response: Dict[str, Any]) -> bool:
        """Determine if agent should continue the invocation loop."""
        # Continue if there are tool calls
        return "tool_calls" in parsed_response
        
    def get_final_answer(self, parsed_response: Dict[str, Any]) -> str:
        """Extract final answer when loop is complete."""
        if "final_answer" in parsed_response:
            return parsed_response["final_answer"]
        if "content" in parsed_response:
            return parsed_response["content"]
        return "I couldn't determine a final answer."
        
    def format_intermediate_steps(self, parsed_response: Dict[str, Any]) -> str:
        """Format intermediate reasoning/action steps."""
        if "tool_results" in parsed_response:
            return f"Tool results: {parsed_response['tool_results']}"
        return ""

class BasicAgentPattern(AgentPattern):
    """Basic agent pattern for the standard Agent class."""
    
    llm_execution_pattern = BasicExecutionPattern()

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
        self.agent_pattern = BasicAgentPattern()
        self.execution_pattern = self.agent_pattern.llm_execution_pattern

    async def think(self, user_input: str, model: str) -> Dict[str, Any]:
        """
        Process user input using the agent's thinking process.
        
        Args:
            user_input: The user's input
            model: The LLM model to use
            
        Returns:
            The agent's thoughts
        """
        # Get messages for the prompt
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        
        # Get response from LLM
        response = self.llm.chat_completion(
            model=model,
            messages=messages,
            tools=[tool.convert_to_function_call() for tool in self.tools]
        )
        
        # Parse the response
        parsed_response = self.execution_pattern.parse_llm_response(response)
        
        # Format thoughts based on the thinking pattern if available
        if self.thinking_pattern:
            thoughts = self.thinking_pattern.process_input({
                "input": user_input,
                "response": parsed_response
            })
        else:
            thoughts = {
                "input": user_input,
                "reasoning": "Processing input",
                "plan": "Generate a response",
                "final_answer": parsed_response.get("final_answer", "")
            }
        
        return thoughts

    async def arun(self, user_input: str, model: str, **kwargs) -> str:
        """Process user input asynchronously and return response.
        
        Args:
            user_input: The user's question or request
            model: The LLM model to use
            **kwargs: Additional parameters
            
        Returns:
            str: The agent's final response
        """
        # Define color codes for better log readability
        BLUE = "\033[94m"      # For iterations
        GREEN = "\033[92m"     # For successful operations
        YELLOW = "\033[93m"    # For tools and actions
        RESET = "\033[0m"      # Reset color
        
        def pretty_json(obj) -> str:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        
        # Initialize variables
        iterations = 0
        should_continue = True
        final_answer = "No final answer generated"
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        
        # Precompute tool function calls for efficiency
        tool_functions = [tool.convert_to_function_call() for tool in self.tools]
        
        # Log initial setup
        if self.log_manager:
            self.log_manager.log_debug(f"\n{BLUE}┌──────────────────────────────────────────┐")
            self.log_manager.log_debug(f"│             INITIAL PROMPT                 │")
            self.log_manager.log_debug(f"└──────────────────────────────────────────┘{RESET}")
            self.log_manager.log_debug(f"{BLUE}User Query: {user_input}{RESET}")
            self.log_manager.log_debug(f"{BLUE}System Context: {pretty_json(messages)}{RESET}")
        
        try:
            # Main agent loop with explicit continuation check
            while should_continue and iterations < self.max_iterations:
                # Log iteration start
                iterations += 1
                if self.log_manager:
                    self.log_manager.log_debug(f"\n{BLUE}┌──────────────────────────────────────────┐")
                    self.log_manager.log_debug(f"│          ITERATION {iterations}/{self.max_iterations}                 │")
                    self.log_manager.log_debug(f"└──────────────────────────────────────────┘{RESET}")
                
                # Get response from LLM
                response = self.llm.chat_completion(
                    model=model,
                    messages=messages,
                    tools=tool_functions,
                    tool_choice="auto" if tool_functions else None
                )
                
                # Parse the LLM response
                parsed_response = self.execution_pattern.parse_llm_response(response)
                
                if self.log_manager:
                    self.log_manager.log_debug(f"{GREEN}Response: {pretty_json(parsed_response)}{RESET}")
                
                # Check if we've reached the final iteration
                if iterations >= self.max_iterations:
                    final_answer = self.execution_pattern.get_final_answer(parsed_response)
                    if self.log_manager:
                        self.log_manager.log_debug(f"{GREEN}Final Answer (max iterations): {final_answer}{RESET}")
                    break
                
                # Handle tool calls if present
                if "tool_calls" in parsed_response:
                    if self.log_manager:
                        self.log_manager.log_debug(f"\n{YELLOW}┌──────────────────────────────────────────┐")
                        self.log_manager.log_debug(f"│              TOOL EXECUTION              │")
                        self.log_manager.log_debug(f"└──────────────────────────────────────────┘{RESET}")
                    
                    # Execute tool calls and get results
                    tool_results = await self.handle_tool_calls(parsed_response["tool_calls"])
                    parsed_response["tool_results"] = tool_results
                    
                    if self.log_manager:
                        self.log_manager.log_debug(f"{YELLOW}Tool Results: {pretty_json(tool_results)}{RESET}")
                    
                    # Format the tool calls properly
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
                
                # Determine if we should continue or return a final answer
                should_continue = self.execution_pattern.should_continue(parsed_response)
                
                if should_continue:
                    # Add intermediate steps if needed
                    formatted_steps = self.execution_pattern.format_intermediate_steps(parsed_response)
                    if formatted_steps:
                        if self.log_manager:
                            self.log_manager.log_debug(f"{BLUE}Intermediate Steps: {formatted_steps}{RESET}")
                else:
                    # Get the final answer and end the loop
                    final_answer = self.execution_pattern.get_final_answer(parsed_response)
                    if self.log_manager:
                        self.log_manager.log_debug(f"\n{GREEN}┌──────────────────────────────────────────┐")
                        self.log_manager.log_debug(f"│               FINAL ANSWER               │")
                        self.log_manager.log_debug(f"└──────────────────────────────────────────┘{RESET}")
                        self.log_manager.log_debug(f"{GREEN}{final_answer}{RESET}")
            
            # Update history and log the interaction
            self._update_history_and_log(user_input, final_answer, model)
            return final_answer
            
        except Exception as e:
            if self.log_manager:
                self.log_manager.log_debug(f"\n\033[91m┌──────────────────────────────────────────┐")
                self.log_manager.log_debug(f"│                  ERROR                   │")
                self.log_manager.log_debug(f"└──────────────────────────────────────────┘\033[0m")
                self.log_manager.log_debug(f"\033[91m{str(e)}\033[0m")
            raise e
        finally:
            if self.log_manager:
                self.log_manager.log_debug(f"\n{GREEN}┌──────────────────────────────────────────┐")
                self.log_manager.log_debug(f"│           INTERACTION COMPLETE           │")
                self.log_manager.log_debug(f"└──────────────────────────────────────────┘{RESET}")
                self.log_manager.print_logs_in_pretty_format()
    
    def run(self, user_input: str, model: str, **kwargs) -> str:
        """Process user input synchronously and return response.
        
        Args:
            user_input: The user's question or request
            model: The LLM model to use
            **kwargs: Additional parameters
            
        Returns:
            str: The agent's final response
        """
        return asyncio.run(self.arun(user_input, model, **kwargs))
    
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
            tool = next((t for t in self.tools if t.__class__.__name__.lower() == tool_name.lower()), None)
            
            if tool:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    # Execute tool with provided arguments
                    result = await tool.arun(**arguments)
                    results.append(result)
                except json.JSONDecodeError:
                    results.append(f"Error: Could not parse arguments for {tool_name}")
                except Exception as e:
                    results.append(f"Error executing {tool_name}: {str(e)}")
            else:
                results.append(f"Error: Tool '{tool_name}' not found")
        
        return results
        
    async def handle_single_tool_call(self, tool_name: str, tool_input: Any) -> Any:
        """Handle a single tool call.
        
        Args:
            tool_name: Name of the tool to call
            tool_input: Input for the tool
            
        Returns:
            Result of the tool call
        """
        tool = None
        for t in self.tools:
            if t.__class__.__name__.lower() == tool_name.replace('functions.', '').lower():
                tool = t
                break
                
        if tool:
            try:
                if isinstance(tool_input, dict):
                    result = await tool.arun(**tool_input)
                else:
                    result = await tool.arun(tool_input)
                return result
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
                
        return f"Error: Tool '{tool_name}' not found"
    
    async def save_memory(self, context: str) -> None:
        """
        Save the agent's current state to a file.
        
        Args:
            context: Context identifier for saving the state
        """
        if self.memory_manager:
            return await self.memory_manager.save_memory(context)
        return None
    
    async def load_memory(self) -> None:
        """
        Load the agent's state from a file.
        """
        if self.memory_manager:
            return await self.memory_manager.load_memory()
        return None
    
    def _update_history_and_log(self, user_input, final_answer, model):
        """Update history and log the interaction.
        
        Args:
            user_input: The user's question or request
            final_answer: The agent's final answer
            model: The LLM model used
        """
        # Update history
        if self.memory_manager:
            self.memory_manager.add({"role": "user", "content": user_input})
            self.memory_manager.add({"role": "assistant", "content": final_answer})
        
        # Log interaction summary
        if self.log_manager:
            self.log_manager.log_interaction(
                user_input=user_input,
                agent_response=final_answer,
                model=model,
                timestamp=datetime.now().isoformat(),
                metadata=self.metadata
            )