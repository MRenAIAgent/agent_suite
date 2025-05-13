"""
Enhanced Tool System Example with React Agent.

This script demonstrates how to use the enhanced tool system with a React agent:
1. Registering legacy tools with the enhanced tool system
2. Using tool selection based on tasks and roles
3. Running a React agent with the selected tools
"""
import asyncio
import os
import sys
from typing import List, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.react_agent import ReActAgent
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager

# Import enhanced tool system
from tools.types import Task, Role, ToolDomain, ToolCapability
from tools.registry import registry
from tools.selection import tool_selector
from tools.base import EnhancedTool

# Import tool adapters
from tools.adapters.integration import register_all_adapted_tools, auto_convert_legacy_tools

# Import original tools for testing
from tools.serpapi_tool import SerpApiSearchTool
from tools.animal_type import AnimalType
from tools.mcp_tool import MCPListTools


class ReactEnhancedExample:
    """Example implementing ReactAgent with the enhanced tool system."""
    
    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-20250219", 
        role_name: str = "researcher",
        task_description: str = "Research and analyze information from various sources",
        max_iterations: int = 20
    ):
        """Initialize the ReactEnhancedExample.
        
        Args:
            model: The LLM model to use
            role_name: The role of the agent (researcher, programmer, etc.)
            task_description: Description of the task to perform
            max_iterations: Maximum number of iterations for the agent
        """
        self.llm = LiteLLM.create_llm()
        self.model = model
        self.role_name = role_name
        self.task_description = task_description
        
        # Initialize the enhanced tool system
        self._initialize_tools()
        
        # Create role and task for tool selection
        self.role = self._create_role(role_name)
        self.task = self._create_task(task_description)
        
        # Select tools for the role and task
        self.tools = self._select_tools()
        
        # Show selected tools
        print(f"\nSelected tools for role '{role_name}' and task '{task_description}':")
        for tool in self.tools:
            print(f"- {tool.__class__.__name__}")
        
        # Convert EnhancedTools back to original Tool format for React agent
        self.agent_tools = self._convert_to_original_tools()
        
        # Initialize the ReactAgent
        self.agent = ReActAgent(
            llm=self.llm,
            role=f"You are a {role_name} with expertise in finding and analyzing information.",
            task=f"Your task is to: {task_description}",
            guide="Use the available tools to complete the task. Think step by step and explain your reasoning.",
            tools=self.agent_tools,
            log_manager=LogManager(),
            max_iterations=max_iterations
        )
    
    def _initialize_tools(self):
        """Initialize the enhanced tool system and register tools."""
        # Clear registry to start fresh
        registry.clear()
        
        # Option 1: Register all adapted legacy tools (MCP, SerpAPI, etc.)
        register_all_adapted_tools()
        
        # Option 2 (alternative): Auto-convert legacy tools
        # Uncommenting the next line would automatically convert any original
        # Tool classes found in the system to EnhancedTool classes
        # auto_convert_legacy_tools()
        
        # Get registered tools count
        print(f"Registered {len(registry.get_all_tools())} tools with the enhanced tool system")
    
    def _create_role(self, role_name: str) -> Role:
        """Create a role for tool selection.
        
        Args:
            role_name: Name of the role
            
        Returns:
            Role object with capabilities and domains
        """
        if role_name == "researcher":
            return Role(
                name="researcher",
                description="Researcher who needs to find and analyze information",
                responsibilities=["Find information", "Analyze data", "Report findings"],
                required_capabilities=[ToolCapability.SEARCH, ToolCapability.FETCH, ToolCapability.ANALYZE],
                domains=[ToolDomain.GENERAL, ToolDomain.SCIENCE]
            )
        elif role_name == "programmer":
            return Role(
                name="programmer",
                description="Programmer who develops and analyzes code",
                responsibilities=["Write code", "Debug issues", "Review code"],
                required_capabilities=[ToolCapability.ANALYZE, ToolCapability.READ, ToolCapability.WRITE],
                domains=[ToolDomain.PROGRAMMING]
            )
        else:
            # Default role
            return Role(
                name=role_name,
                description=f"General {role_name} role",
                responsibilities=[],
                required_capabilities=[ToolCapability.FETCH, ToolCapability.SEARCH],
                domains=[ToolDomain.GENERAL]
            )
    
    def _create_task(self, task_description: str) -> Task:
        """Create a task for tool selection.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Task object with capabilities and domain
        """
        # Extract keywords from the task description to determine capabilities and domain
        task_lower = task_description.lower()
        
        # Default capabilities and domain
        capabilities = [ToolCapability.SEARCH, ToolCapability.FETCH]
        domain = ToolDomain.GENERAL
        
        # Add more capabilities based on task keywords
        if any(kw in task_lower for kw in ["analyze", "analysis", "investigate"]):
            capabilities.append(ToolCapability.ANALYZE)
        
        if any(kw in task_lower for kw in ["generate", "create", "write"]):
            capabilities.append(ToolCapability.GENERATE)
        
        if any(kw in task_lower for kw in ["code", "programming", "develop", "software"]):
            domain = ToolDomain.PROGRAMMING
        
        if any(kw in task_lower for kw in ["science", "scientific", "experiment"]):
            domain = ToolDomain.SCIENCE
        
        return Task(
            description=task_description,
            goal="Complete the task successfully",
            type="research" if "research" in task_lower else "general",
            domain=domain,
            required_capabilities=capabilities,
            expected_outputs=["Comprehensive report", "Analysis"]
        )
    
    def _select_tools(self) -> List[EnhancedTool]:
        """Select tools based on the role and task.
        
        Returns:
            List of EnhancedTool instances
        """
        # Get tools for both role and task
        role_tools = tool_selector.select_tools_for_role(self.role)
        task_tools = tool_selector.select_tools_for_task(self.task)
        
        # Combine unique tools
        all_tools = list(set(role_tools + task_tools))
        
        # If no tools found, use all available
        if not all_tools:
            print("No tools matched the role/task requirements, using all available tools")
            all_tools = registry.get_all_tools()
        
        return all_tools
    
    def _convert_to_original_tools(self):
        """Convert EnhancedTools to the original Tool format for the React agent.
        
        Returns:
            List of original Tool instances
        """
        # Create instances of original Tool classes for the ReactAgent
        # The ReactAgent expects instances of the original Tool class
        
        original_tools = []
        
        # For each enhanced tool, create the corresponding original tool
        for enhanced_tool in self.tools:
            # Get the original tool class based on enhanced tool name
            class_name = enhanced_tool.__class__.__name__
            
            # Check for specific tool types and create the appropriate instance
            if "SerpApiSearchTool" in class_name:
                original_tools.append(SerpApiSearchTool())
            elif "AnimalTypeTool" in class_name:
                original_tools.append(AnimalType())
            elif "MCPListTools" in class_name:
                original_tools.append(MCPListTools())
            # Add more mappings as needed
        
        # If no original tools were created, add a default tool
        if not original_tools:
            # Add a default tool 
            original_tools.append(SerpApiSearchTool())
            print("Warning: No original tools were mapped. Using default SerpApiSearchTool.")
        
        return original_tools
    
    async def arun(self, user_input: str) -> str:
        """Run the agent asynchronously.
        
        Args:
            user_input: The user's input
            
        Returns:
            The agent's response
        """
        print(f"Running agent with {len(self.agent_tools)} tools")
        return await self.agent.arun(user_input, self.model)
    
    def run(self, user_input: str) -> str:
        """Run the agent synchronously.
        
        Args:
            user_input: The user's input
            
        Returns:
            The agent's response
        """
        return asyncio.run(self.arun(user_input))


# Example usage
if __name__ == "__main__":
    # Create the enhanced example with researcher role
    agent = ReactEnhancedExample(
        role_name="researcher", 
        task_description="Research and analyze the impact of climate change on coastal cities",
        max_iterations=10
    )
    
    # Example queries to test the agent
    test_queries = [
        "Analyze the impact of climate change on coastal cities like Miami and New York"
    ]
    
    # Run the agent on each query
    for query in test_queries:
        print(f"\n\n=== Query: {query} ===")
        
        try:
            response = agent.run(query)
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error running agent: {e}") 