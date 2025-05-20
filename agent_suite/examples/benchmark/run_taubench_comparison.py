#!/usr/bin/env python3
"""
TauBench Agent Comparison Example

This script demonstrates how to use TauBench to compare different agent implementations.
It sets up and evaluates multiple agents on the same tasks to compare their performance.
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import Dict, Any, List

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent_suite.agent_suite.agents.react.agent import ReActAgent
from agent_suite.agent_suite.llm.litellm.litellm import LiteLLM
from agent_suite.agent_suite.benchmarks.comparison.agent_comparison import compare_agents
from agent_suite.examples.simple_react.react_agent_example import ReactAgentExample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom agent wrapper with simplified interface
class SimpleReactAgent:
    """Simple wrapper for the ReAct agent for evaluation."""
    
    def __init__(self, max_iterations: int = 10):
        """Initialize the agent."""
        self.agent = ReActAgent(
            llm=LiteLLM.create_llm(),
            role="You are a helpful AI assistant that answers questions accurately.",
            task="Answer the user's question in a clear and concise manner.",
            guide="Think step by step to arrive at the correct answer.",
            tools=[],  # No tools for this basic agent
            max_iterations=max_iterations
        )
        
    async def arun(self, user_input: str, model: str, **kwargs) -> str:
        """Run the agent asynchronously."""
        return await self.agent.arun(user_input, model)

async def run_comparison(
    model_name: str = "anthropic/claude-3-7-sonnet-20250219",
    output_dir: str = "benchmark_results",
    task_limit: int = 3,
    category_limit: int = 2
):
    """
    Run the TauBench comparison on different agent implementations.
    
    Args:
        model_name: Name of the model to use
        output_dir: Directory to save evaluation results
        task_limit: Maximum number of tasks per category
        category_limit: Maximum number of categories to evaluate
    """
    # Create different agent implementations to compare
    basic_agent = SimpleReactAgent(max_iterations=10)
    react_agent = ReactAgentExample(max_iterations=15)
    
    # Set up the agents to compare
    agents = {
        "Simple ReAct": basic_agent,
        "Enhanced ReAct": react_agent.agent
    }
    
    # Run the comparison
    results = await compare_agents(
        agents_dict=agents,
        model_name=model_name,
        output_dir=output_dir,
        task_limit=task_limit,
        category_limit=category_limit
    )
    
    # Print a summary of the results
    logger.info("\n=== Comparison Results Summary ===")
    for agent_name, agent_results in results.items():
        logger.info(f"\n{agent_name}:")
        logger.info(f"  Overall Score: {agent_results.get('overall_score', 0):.2f}")
        logger.info(f"  Success Rate: {agent_results.get('overall_success_rate', 0):.2%}")
        
        # Print category results
        logger.info("  Category Scores:")
        for category, metrics in agent_results.get("category_metrics", {}).items():
            logger.info(f"    {category}: {metrics.get('overall_score', 0):.2f} ({metrics.get('success_rate', 0):.2%})")
    
    # Log where to find the report
    report_path = os.path.join(output_dir, "taubench_comparison_report.html")
    logger.info(f"\nDetailed comparison report: {report_path}")
    
    return results

def main():
    """Run the TauBench comparison example."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run TauBench comparison on different agent implementations")
    parser.add_argument("--model", type=str, default="anthropic/claude-3-7-sonnet-20250219", help="Model to use for evaluation")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Directory to save results")
    parser.add_argument("--tasks", type=int, default=3, help="Number of tasks per category")
    parser.add_argument("--categories", type=int, default=2, help="Number of categories to evaluate")
    args = parser.parse_args()
    
    # Run the comparison
    asyncio.run(run_comparison(
        model_name=args.model,
        output_dir=args.output,
        task_limit=args.tasks,
        category_limit=args.categories
    ))

if __name__ == "__main__":
    main() 