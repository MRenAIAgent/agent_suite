#!/usr/bin/env python3
"""
Agent Evaluation Runner

This script runs evaluations on your Agent Suite agents using the TauBench framework.
Additional frameworks like AgentBench and AgentBoard will be integrated in the future.
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
from agent_suite.agent_suite.benchmarks.taubench.taubench_eval import TauBenchEvaluation
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

async def run_evaluation(
    agent_type: str = "simple",
    model_name: str = "anthropic/claude-3-7-sonnet-20250219",
    output_dir: str = "benchmark_results",
    task_limit: int = 3,
    category_limit: int = 2
):
    """
    Run agent evaluation using TauBench.
    
    Args:
        agent_type: Type of agent to evaluate ("simple" or "enhanced")
        model_name: Name of the model to use
        output_dir: Directory to save evaluation results
        task_limit: Maximum number of tasks per category to evaluate
        category_limit: Maximum number of categories to evaluate
    
    Returns:
        Dict with evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the agent based on the specified type
    if agent_type == "simple":
        agent = SimpleReactAgent(max_iterations=10)
        logger.info("Using simple ReAct agent")
        # For the simple agent wrapper, we need to use the agent attribute
        evaluation_agent = agent.agent
    elif agent_type == "enhanced":
        agent = ReactAgentExample(max_iterations=15)
        logger.info("Using enhanced ReAct agent with tools")
        # For ReactAgentExample, we need to use its agent attribute
        evaluation_agent = agent.agent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Create TauBench evaluator
    evaluator = TauBenchEvaluation(
        agent=evaluation_agent,
        model=model_name,
        task_limit=task_limit,
        category_limit=category_limit,
        output_dir=output_dir
    )
    
    # Run evaluation
    logger.info(f"Starting evaluation with {category_limit} categories, {task_limit} tasks per category")
    results = await evaluator.run_evaluation()
    
    # Print summary of results
    logger.info("\n=== Evaluation Results Summary ===")
    
    # Print category results
    if "categories" in results:
        for category, category_results in results.get("categories", {}).items():
            total_tasks = category_results.get("total_tasks", 0)
            successful_tasks = category_results.get("successful_tasks", 0)
            success_rate = 0
            if total_tasks > 0:
                success_rate = successful_tasks / total_tasks
            
            logger.info(f"Category: {category}")
            logger.info(f"  Tasks: {successful_tasks}/{total_tasks}")
            logger.info(f"  Success Rate: {success_rate:.2%}")
    
    # Return results
    return results

def main():
    """Run the agent evaluation example."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--agent", type=str, choices=["simple", "enhanced"], default="simple", 
                         help="Type of agent to evaluate")
    parser.add_argument("--model", type=str, default="anthropic/claude-3-7-sonnet-20250219", 
                         help="Model to use for evaluation")
    parser.add_argument("--output", type=str, default="benchmark_results", 
                         help="Directory to save results")
    parser.add_argument("--tasks", type=int, default=3, 
                         help="Number of tasks per category (0 for all)")
    parser.add_argument("--categories", type=int, default=2, 
                         help="Number of categories to evaluate (0 for all)")
    
    args = parser.parse_args()
    
    # Run the evaluation
    asyncio.run(run_evaluation(
        agent_type=args.agent,
        model_name=args.model,
        output_dir=args.output,
        task_limit=args.tasks,
        category_limit=args.categories
    ))

if __name__ == "__main__":
    main() 