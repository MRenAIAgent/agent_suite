#!/usr/bin/env python3
"""
React Agent

This module implements the ReAct (Reasoning-Acting) agent pattern,
which alternates between reasoning and taking action using tools.
"""
import asyncio
import logging
import re
import time
import json
from typing import List, Dict, Any, Optional, Union, Callable

from ...llm.base import LLM
from ..storage.intermediate_storage import (
    IntermediateStorageBase,
    InMemoryIntermediateStorage
)

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    A ReAct (Reasoning-Acting) agent that alternates between 
    reasoning about a task and taking actions using tools.
    """
    
    def __init__(
        self,
        llm: LLM,
        role: str = "You are a helpful AI assistant.",
        task: str = "Help the user with their request.",
        guide: str = "Think step by step and use the available tools when necessary.",
        tools: Optional[List[Any]] = None,
        max_iterations: int = 10,
        intermediate_storage: Optional[IntermediateStorageBase] = None
    ):
        """Initialize the ReAct agent.
        
        Args:
            llm: LLM instance for generating agent responses
            role: Agent role description
            task: Task description
            guide: Guide for the agent's approach
            tools: List of tools available to the agent
            max_iterations: Maximum number of iterations to run
            intermediate_storage: Optional storage for intermediate results
        """
        self.llm = llm
        self.role = role
        self.task = task
        self.guide = guide
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.intermediate_storage = intermediate_storage or InMemoryIntermediateStorage()
        
        # Tool name to tool map
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Regular expressions for parsing actions
        self.thought_pattern = r"(?:Thought|THOUGHT):[\s]*(.*?)(?:$|(?=\nAction:))"
        self.action_pattern = r"(?:Action|ACTION):[\s]*(.*?)(?:$|(?=\nAction\s?Input:))"
        self.action_input_pattern = r"(?:Action\s?Input|ACTION\s?INPUT):[\s]*(.*?)(?:$|(?=\n(?:Observation|OBSERVATION):))"
        self.observation_pattern = r"(?:Observation|OBSERVATION):[\s]*(.*?)(?:$|\n\n)"
        self.answer_pattern = r"(?:Answer|ANSWER):[\s]*(.*?)(?:$|\n\n)"
        
    def _format_tools_description(self) -> str:
        """Format the tools description for inclusion in the prompt.
        
        Returns:
            Formatted string describing available tools
        """
        if not self.tools:
            return "No tools are available."
            
        description = "You have access to the following tools:\n\n"
        
        for i, tool in enumerate(self.tools, 1):
            name = getattr(tool, 'name', f"Tool {i}")
            desc = getattr(tool, 'description', "No description available.")
            description += f"{i}. {name}: {desc}\n"
            
        description += "\n"
        return description
        
    def _format_interaction_history(self, history: List[Dict[str, str]]) -> str:
        """Format the interaction history for inclusion in the prompt.
        
        Args:
            history: List of interaction steps
            
        Returns:
            Formatted string of interaction history
        """
        formatted_history = ""
        
        for step in history:
            if 'thought' in step:
                formatted_history += f"Thought: {step['thought']}\n"
                
            if 'action' in step and 'action_input' in step:
                formatted_history += f"Action: {step['action']}\n"
                formatted_history += f"Action Input: {step['action_input']}\n"
                
            if 'observation' in step:
                formatted_history += f"Observation: {step['observation']}\n\n"
                
        return formatted_history
        
    def _format_prompt(self, query: str, history: List[Dict[str, str]]) -> str:
        """Format the full prompt for the LLM.
        
        Args:
            query: User's query
            history: List of interaction steps so far
            
        Returns:
            Formatted prompt string
        """
        prompt = f"{self.role}\n\n"
        prompt += f"TASK: {self.task}\n\n"
        prompt += f"GUIDE: {self.guide}\n\n"
        prompt += f"USER QUERY: {query}\n\n"
        prompt += f"{self._format_tools_description()}\n"
        
        if history:
            prompt += "EXECUTION TRACE:\n\n"
            prompt += self._format_interaction_history(history)
            
        prompt += "Thought: "
        
        return prompt
        
    def _parse_llm_response(self, text: str) -> Dict[str, str]:
        """Parse the LLM response into structured components.
        
        Args:
            text: LLM response text
            
        Returns:
            Dictionary with parsed components
        """
        result = {}
        
        # Extract the thought
        thought_match = re.search(self.thought_pattern, text, re.DOTALL)
        if thought_match:
            result['thought'] = thought_match.group(1).strip()
            
        # Extract the action
        action_match = re.search(self.action_pattern, text, re.DOTALL)
        if action_match:
            result['action'] = action_match.group(1).strip()
            
        # Extract the action input
        action_input_match = re.search(self.action_input_pattern, text, re.DOTALL)
        if action_input_match:
            result['action_input'] = action_input_match.group(1).strip()
            
        # Check for an answer
        answer_match = re.search(self.answer_pattern, text, re.DOTALL)
        if answer_match:
            result['answer'] = answer_match.group(1).strip()
            
        return result
        
    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool with the given input.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input for the tool
            
        Returns:
            Result of the tool execution as a string
        """
        # Find the tool by name
        tool = self.tool_map.get(tool_name)
        
        if not tool:
            return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self.tool_map.keys())}"
            
        try:
            # Try to parse the input as JSON
            try:
                input_json = json.loads(tool_input)
                result = await tool.arun(**input_json)
            except json.JSONDecodeError:
                # If not JSON, pass the input as is
                result = await tool.arun(tool_input)
                
            # Convert the result to a string
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result, indent=2)
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool {tool_name}: {str(e)}"
            
    async def arun(self, query: str, model: Optional[str] = None) -> str:
        """Run the agent asynchronously.
        
        Args:
            query: User's query
            model: Optional model to use for generation
            
        Returns:
            Agent's response
        """
        history = []
        iterations = 0
        
        logger.info(f"Starting ReAct agent for query: {query}")
        
        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"Iteration {iterations}/{self.max_iterations}")
            
            # Format the current prompt
            prompt = self._format_prompt(query, history)
            
            # Save to intermediate storage
            iteration_key = f"iteration_{iterations}_prompt"
            self.intermediate_storage.store(iteration_key, prompt)
            
            # Get the LLM's response
            llm_response = await self.llm.generate(prompt, model=model)
            
            # Save LLM response to intermediate storage
            response_key = f"iteration_{iterations}_response"
            self.intermediate_storage.store(response_key, llm_response)
            
            # Parse the response
            parsed = self._parse_llm_response(llm_response)
            
            # Add thought to history
            if 'thought' in parsed:
                history.append({'thought': parsed['thought']})
                
            # Check if the agent wants to perform an action
            if 'action' in parsed and 'action_input' in parsed:
                action = parsed['action']
                action_input = parsed['action_input']
                
                # Add action to history
                current_step = history[-1] if history else {}
                current_step['action'] = action
                current_step['action_input'] = action_input
                
                if history and history[-1] == current_step:
                    pass  # Already added to history
                else:
                    history.append(current_step)
                
                # Execute the tool
                observation = await self._execute_tool(action, action_input)
                
                # Add observation to history
                history.append({'observation': observation})
                
            # Check if the agent has an answer
            if 'answer' in parsed:
                return parsed['answer']
                
            # Check if the agent is done
            if not any(k in parsed for k in ['action', 'action_input']):
                # If no action but has thought, return the thought
                if 'thought' in parsed:
                    return parsed['thought']
                    
                # Otherwise, return what we have
                return llm_response
                
        # If we've reached the maximum iterations without an answer,
        # make a final call to the LLM to summarize findings
        logger.warning(f"Reached maximum iterations ({self.max_iterations}), generating final answer")
        
        final_prompt = self._format_prompt(query, history)
        final_prompt += "\nYou have used all available iterations. Based on the information gathered, " \
                       "please provide your final answer:\n\nAnswer: "
        
        self.intermediate_storage.store("final_prompt", final_prompt)
        
        final_response = await self.llm.generate(final_prompt, model=model)
        self.intermediate_storage.store("final_response", final_response)
        
        # Try to extract just the answer
        answer_match = re.search(self.answer_pattern, final_response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
            
        return final_response
        
    def run(self, query: str, model: Optional[str] = None) -> str:
        """Run the agent synchronously.
        
        Args:
            query: User's query
            model: Optional model to use for generation
            
        Returns:
            Agent's response
        """
        return asyncio.run(self.arun(query, model))