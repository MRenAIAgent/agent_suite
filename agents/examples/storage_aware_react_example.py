import asyncio
import os
import sys
import time
import json
from typing import Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.storage_aware_react_agent import StorageAwareReActAgent
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager
from tools.calculator import CalculatorTool
from tools.wikipedia import WikipediaTool


async def run_comparison(query: str, model: str = "gpt-3.5-turbo"):
    """
    Run a comparison between standard storage and token-efficient storage.
    
    Args:
        query: The user query to process
        model: The model to use
    """
    print(f"\n{'=' * 80}")
    print(f"RUNNING COMPARISON FOR QUERY: {query}")
    print(f"{'=' * 80}\n")
    
    # Create tools
    calculator = CalculatorTool()
    wikipedia = WikipediaTool()
    tools = [calculator, wikipedia]
    
    # Set up logging
    log_manager = LogManager()
    
    # Create LLM instance
    llm = LiteLLM.create_llm()
    
    # Define common parameters
    role = "You are an intelligent assistant that breaks down and solves complex problems step-by-step."
    task = "Answer user questions by thinking carefully and using tools when necessary."
    guide = "Always show your reasoning process. Break down complex problems into steps."
    
    # Create agent with standard storage
    start_time = time.time()
    standard_agent = StorageAwareReActAgent(
        llm=llm,
        role=role,
        task=task,
        guide=guide,
        tools=tools,
        log_manager=log_manager,
        max_iterations=5,
        token_efficient=False
    )
    
    # Run standard agent
    print("\n🔄 Running agent with standard storage...")
    standard_result = await standard_agent.arun(query, model)
    standard_time = time.time() - start_time
    standard_summary = standard_agent.get_execution_summary()
    
    # Create agent with token-efficient storage
    start_time = time.time()
    efficient_agent = StorageAwareReActAgent(
        llm=llm,
        role=role,
        task=task,
        guide=guide,
        tools=tools,
        log_manager=log_manager,
        max_iterations=5,
        token_efficient=True,
        max_detailed_steps=3  # Only keep 3 most recent steps in detail
    )
    
    # Run token-efficient agent
    print("\n🔄 Running agent with token-efficient storage...")
    efficient_result = await efficient_agent.arun(query, model)
    efficient_time = time.time() - start_time
    efficient_summary = efficient_agent.get_execution_summary()
    
    # Report results
    print(f"\n{'=' * 80}")
    print(f"COMPARISON RESULTS")
    print(f"{'=' * 80}\n")
    
    print(f"📊 STANDARD STORAGE:")
    print(f"   - Time: {standard_time:.2f}s")
    print(f"   - Iterations: {standard_summary.get('total_iterations', 0)}")
    print(f"   - Answer: {standard_result[:100]}..." if len(standard_result) > 100 else standard_result)
    
    print(f"\n📊 TOKEN-EFFICIENT STORAGE:")
    print(f"   - Time: {efficient_time:.2f}s")
    print(f"   - Iterations: {efficient_summary.get('total_iterations', 0)}")
    print(f"   - Summarized steps: {efficient_summary.get('summarized_steps', 0)}")
    print(f"   - Detailed steps: {efficient_summary.get('detailed_steps', 0)}")
    print(f"   - Answer: {efficient_result[:100]}..." if len(efficient_result) > 100 else efficient_result)
    
    # Calculate improvements
    time_diff = ((standard_time - efficient_time) / standard_time) * 100
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   - Time difference: {time_diff:.1f}% {'faster' if time_diff > 0 else 'slower'} with token-efficient storage")
    
    return {
        "standard": {
            "time": standard_time,
            "result": standard_result,
            "summary": standard_summary
        },
        "efficient": {
            "time": efficient_time,
            "result": efficient_result,
            "summary": efficient_summary
        }
    }


async def main():
    """Run example comparisons with different queries."""
    test_queries = [
        "What is the population of France and what is that number squared?",
        "Who was Albert Einstein and what is the speed of light squared?",
        "What year was the United Nations founded, and what is that year multiplied by 2.5?"
    ]
    
    results = {}
    
    for query in test_queries:
        results[query] = await run_comparison(query)
    
    # Save detailed results to file
    with open("storage_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to storage_comparison_results.json")


if __name__ == "__main__":
    asyncio.run(main()) 