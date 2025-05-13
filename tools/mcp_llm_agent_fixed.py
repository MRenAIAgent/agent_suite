#!/usr/bin/env python3
"""
MCP LLM Agent (Fixed)

This script demonstrates an end-to-end LLM agent that uses tools from an MCP server
to answer user queries. The agent uses a direct approach without relying on the
enhanced tool system to avoid Pydantic compatibility issues.

This version uses the correct LiteLLM interface.
"""
import asyncio
import os
import sys
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.litellm.litellm import LiteLLM
from log.logging import LogManager
from tools.mcp_direct import MCPToolProvider, MCPTool
from tools.mock_mcp_server import MockMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCPLLMAgent:
    """An LLM agent that can use tools from an MCP server."""
    
    def __init__(
        self,
        mcp_server_url: str,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        max_iterations: int = 5,
        api_key: Optional[str] = None
    ):
        """Initialize the MCP LLM agent.
        
        Args:
            mcp_server_url: URL of the MCP server
            model: LLM model to use
            max_iterations: Maximum number of tool use iterations
            api_key: Optional API key for MCP server
        """
        self.mcp_server_url = mcp_server_url
        self.model = model
        self.max_iterations = max_iterations
        self.api_key = api_key
        
        # Initialize components
        self.llm = LiteLLM.create_llm()
        self.log_manager = LogManager()
        self.provider = MCPToolProvider(server_url=mcp_server_url, api_key=api_key)
        
        # Will be set after initialization
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent by discovering MCP tools."""
        if self.initialized:
            return
        
        logger.info(f"Initializing MCP LLM agent with server: {self.mcp_server_url}")
        
        # Initialize the provider to discover tools
        await self.provider.initialize()
        
        self.initialized = True
    
    def _format_tool_descriptions(self, tools: List[MCPTool]) -> str:
        """Format tool descriptions for the LLM prompt.
        
        Args:
            tools: List of MCP tools
            
        Returns:
            Formatted tool descriptions
        """
        descriptions = []
        
        for i, tool in enumerate(tools):
            params = tool.get_parameters()
            required_params = params.get("required", [])
            properties = params.get("properties", {})
            
            # Format parameters
            param_desc = []
            for param_name, param_info in properties.items():
                is_required = param_name in required_params
                param_type = param_info.get("type", "any")
                param_description = param_info.get("description", "")
                
                # Add required flag
                if is_required:
                    param_desc.append(f"- {param_name} ({param_type}, required): {param_description}")
                else:
                    param_desc.append(f"- {param_name} ({param_type}, optional): {param_description}")
            
            # Build the tool description
            tool_desc = [
                f"Tool {i+1}: {tool.name}",
                f"Description: {tool.description}",
                "Parameters:"
            ]
            
            if param_desc:
                tool_desc.extend(param_desc)
            else:
                tool_desc.append("- No parameters required")
            
            descriptions.append("\n".join(tool_desc))
        
        return "\n\n".join(descriptions)
    
    def _parse_tool_call(self, text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse a tool call from the LLM response.
        
        Args:
            text: The LLM response text
            
        Returns:
            Tuple of (tool_name, parameters) or (None, None) if not found
        """
        # Pattern to match a tool call
        pattern = r"(?:I will use|Using|Let me use|I'll use).*?(?:the\s+)?`?([a-zA-Z0-9_-]+)`?.*?(?:tool|function)"
        tool_match = re.search(pattern, text, re.IGNORECASE)
        
        if not tool_match:
            return None, None
        
        tool_name = tool_match.group(1).strip()
        
        # Parse parameters - look for JSON-like format or key-value pairs
        params = {}
        
        # Try to find JSON
        json_pattern = r'\{[^{}]*\}'
        json_match = re.search(json_pattern, text)
        
        if json_match:
            try:
                params = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                # Not valid JSON, fallback to key-value extraction
                pass
        
        if not params:
            # Extract key-value pairs
            kv_pattern = r'(?:Parameter|Param|Input|Argument)s?:?\s*((?:[a-zA-Z0-9_]+\s*=\s*[^,\n]+,?\s*)+)'
            kv_match = re.search(kv_pattern, text, re.IGNORECASE)
            
            if kv_match:
                kv_text = kv_match.group(1)
                pairs = re.findall(r'([a-zA-Z0-9_]+)\s*=\s*([^,\n]+),?', kv_text)
                
                for key, value in pairs:
                    # Clean up the value
                    value = value.strip().strip('"\'')
                    
                    # Try to convert to appropriate type
                    if value.isdigit():
                        params[key] = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                        params[key] = float(value)
                    elif value.lower() in ('true', 'false'):
                        params[key] = value.lower() == 'true'
                    else:
                        params[key] = value
            else:
                # Look for parameters in the text using common patterns
                for param_pattern in [
                    r'([a-zA-Z0-9_]+):\s*"([^"]+)"',
                    r'([a-zA-Z0-9_]+):\s*\'([^\']+)\'',
                    r'([a-zA-Z0-9_]+):\s*([^,\n\s]+)',
                    r'([a-zA-Z0-9_]+)\s*=\s*"([^"]+)"',
                    r'([a-zA-Z0-9_]+)\s*=\s*\'([^\']+)\'',
                    r'([a-zA-Z0-9_]+)\s*=\s*([^,\n\s]+)',
                ]:
                    for key, value in re.findall(param_pattern, text):
                        if key not in params:
                            params[key] = value
        
        return tool_name, params
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response text
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can use tools to answer questions."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_completion(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=False
        )
        
        # Extract the response content
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        
        # Fallback for other response formats
        try:
            if isinstance(response, dict):
                return response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Error extracting response content: {e}")
            return "Error: Unable to get a response from the LLM"
    
    async def answer_query(self, query: str) -> str:
        """Answer a user query using LLM and MCP tools.
        
        Args:
            query: The user's query
            
        Returns:
            Agent's response
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
        
        # Select tools for the query
        tools = self.provider.select_tools_for_query(query)
        
        if not tools:
            logger.warning("No tools selected for query")
            return "I don't have any suitable tools available to answer your question."
        
        # Format tool descriptions
        tool_descriptions = self._format_tool_descriptions(tools)
        
        # Initial conversation context
        conversation = [
            "You are an AI assistant that can use tools to answer user questions.",
            f"User Query: {query}",
            "",
            "Available Tools:",
            tool_descriptions,
            "",
            "Instructions:",
            "1. Analyze the user query",
            "2. Determine which tool would be best to answer the query",
            "3. Explain which tool you'll use and why",
            "4. Specify the tool name and parameters needed"
        ]
        
        # Collect agent's thoughts and results
        agent_thoughts = []
        
        # Tool execution loop
        for iteration in range(self.max_iterations):
            # Get an LLM response
            prompt = "\n".join(conversation)
            llm_response = await self._get_llm_response(prompt)
            
            # Store the thought
            agent_thoughts.append(f"Thought {iteration+1}:\n{llm_response}")
            
            # Parse tool call
            tool_name, parameters = self._parse_tool_call(llm_response)
            
            if not tool_name:
                # No tool call detected, probably a final answer
                break
            
            # Find the tool
            tool = self.provider.get_tool(tool_name)
            if not tool:
                logger.warning(f"Tool '{tool_name}' not found")
                conversation.append(f"Error: Tool '{tool_name}' not found. Please use one of the available tools.")
                continue
            
            # Execute the tool
            try:
                logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
                result = await tool.arun(**parameters)
                
                # Add the result to the conversation
                conversation.append(f"Tool Execution: {tool_name}")
                conversation.append(f"Parameters: {json.dumps(parameters)}")
                conversation.append(f"Result: {json.dumps(result, indent=2)}")
                conversation.append("")
                conversation.append("Based on the tool result, provide your final answer to the user's query.")
            
            except Exception as e:
                logger.error(f"Error executing tool '{tool_name}': {e}")
                conversation.append(f"Error executing tool '{tool_name}': {str(e)}")
                conversation.append("Please try a different approach or different parameters.")
        
        # Get the final answer from the LLM
        prompt = "\n".join(conversation)
        final_response = await self._get_llm_response(prompt)
        
        # Format the final output
        formatted_output = final_response
        
        # Optionally for debugging, show the thought process:
        # formatted_output = "\n\n".join(agent_thoughts + ["Final Answer:\n" + final_response])
        
        return formatted_output
    
    async def close(self):
        """Close the agent and release resources."""
        await self.provider.close()


async def main():
    """Run the MCP LLM agent with a mock server."""
    # Start a mock MCP server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        print("\n=== MCP LLM Agent Demo ===")
        
        # Create the agent
        agent = MCPLLMAgent(mcp_server_url=server.base_url)
        
        # Initialize
        await agent.initialize()
        
        # Test queries
        test_queries = [
            "Calculate 125 * 8 - 32",
            "What's the weather like in Seattle?",
            "What's the current stock price of AAPL?",
            "Search for information about artificial intelligence"
        ]
        
        for query in test_queries:
            print(f"\n\n{'='*50}")
            print(f"QUERY: {query}")
            print(f"{'='*50}")
            
            # Get answer
            answer = await agent.answer_query(query)
            
            print(f"\nANSWER: {answer}")
        
        # Close agent
        await agent.close()
    
    finally:
        server.stop()


if __name__ == "__main__":
    asyncio.run(main()) 