"""
Sequential thinking MCP client implementation.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from agents.mcp_client.client import MCPServerConfig
from agents.mcp_client.mcp_sdk_client import MCPSDKClient
from agents.mcp_client.sequential_thinking import SequentialThinkPattern

logger = logging.getLogger(__name__)


class SequentialThinkingMCPClient(MCPSDKClient):
    """
    MCP client implementation that uses sequential thinking for complex tasks.
    
    This client extends the standard MCP SDK client with sequential thinking
    capabilities, allowing for step-by-step reasoning to solve complex problems.
    """
    
    def __init__(self, server_config: MCPServerConfig):
        """
        Initialize the sequential thinking MCP client.
        
        Args:
            server_config: Configuration for connecting to an MCP server
        """
        super().__init__(server_config)
        self.thinking_pattern = SequentialThinkPattern()
        
    async def solve_problem_sequentially(self, problem: str, max_thoughts: int = 10, 
                                        model: str = None) -> Dict[str, Any]:
        """
        Solve a problem using sequential thinking.
        
        This method breaks down a complex problem into a sequence of steps,
        using the MCP server's tools and resources as needed during each step.
        
        Args:
            problem: The problem statement to solve
            max_thoughts: Maximum number of thoughts to generate
            model: Optional LLM model to use for sampling
            
        Returns:
            Result of the sequential thinking process
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to MCP server")
            
        # Initialize the thinking process
        thoughts = []
        current_thought = 1
        need_more_thoughts = True
        
        # Start the sequential thinking process
        logger.info(f"Starting sequential thinking process for problem: {problem}")
        
        while need_more_thoughts and current_thought <= max_thoughts:
            logger.info(f"Generating thought {current_thought}/{max_thoughts}")
            
            # Create the prompt for the current thought
            prompt = f"""
            # Sequential Thinking - Step {current_thought}

            ## Problem Statement:
            {problem}

            ## Previous Thoughts:
            {self._format_previous_thoughts(thoughts)}

            ## Current Thought (Step {current_thought}):
            """
            
            # Send the prompt to the MCP server for sampling
            try:
                # In real implementation, this would use a proper sampling call
                # For now, we'll use our dummy sampling handler
                sampling_result = await self._handle_sampling({
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": prompt
                    }
                })
                
                # Extract the thought from the sampling result
                if isinstance(sampling_result, dict) and "content" in sampling_result:
                    if isinstance(sampling_result["content"], dict) and "text" in sampling_result["content"]:
                        thought_content = sampling_result["content"]["text"]
                    else:
                        thought_content = str(sampling_result["content"])
                else:
                    thought_content = str(sampling_result)
                
                # Add the thought to our collection
                thoughts.append({
                    "number": current_thought,
                    "content": thought_content,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                # Check if we need more thoughts by looking for keywords in the thought
                need_more_thoughts = (
                    "need more analysis" in thought_content.lower() or
                    "further investigation" in thought_content.lower() or
                    "continuing to" in thought_content.lower() or
                    not any(kw in thought_content.lower() for kw in ["conclusion", "final answer", "in conclusion"])
                )
                
                if current_thought == max_thoughts:
                    need_more_thoughts = False
                    
                current_thought += 1
                
            except Exception as e:
                logger.error(f"Error during sequential thinking step {current_thought}: {e}")
                break
                
        # Format the final result
        result = {
            "problem": problem,
            "thoughts": thoughts,
            "total_thoughts": len(thoughts),
            "conclusion": self._extract_conclusion(thoughts)
        }
        
        logger.info(f"Completed sequential thinking process with {len(thoughts)} thoughts")
        return result
    
    def _format_previous_thoughts(self, thoughts: List[Dict[str, Any]]) -> str:
        """Format previous thoughts for inclusion in the prompt."""
        if not thoughts:
            return "No previous thoughts."
            
        formatted = ""
        for thought in thoughts:
            formatted += f"### Thought {thought['number']}:\n{thought['content']}\n\n"
            
        return formatted
    
    def _extract_conclusion(self, thoughts: List[Dict[str, Any]]) -> str:
        """Extract the conclusion from the final thoughts."""
        if not thoughts:
            return "No conclusion reached."
            
        # Look for conclusion markers in the last thought
        last_thought = thoughts[-1]["content"]
        
        # Try to find conclusion section
        if "conclusion" in last_thought.lower():
            for line in last_thought.split("\n"):
                if "conclusion" in line.lower():
                    start_idx = last_thought.lower().find("conclusion")
                    return last_thought[start_idx:]
        
        # If no explicit conclusion, use the last thought
        return f"Final thought: {last_thought}" 