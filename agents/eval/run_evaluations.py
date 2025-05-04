#!/usr/bin/env python
"""
Agent Evaluation Runner

This script runs evaluations on agents using different benchmarking frameworks.
"""

import argparse
import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import logging

from agents.react_agent import ReActAgent
from agents.eval.agentbench.agentbench_eval import AgentBenchEvaluation
from agents.eval.agentboard.agentboard_eval import AgentBoardEvaluation
# Import TauBench evaluation
try:
    from agents.eval.taubench.taubench_eval import TauBenchEvaluation
    TAUBENCH_AVAILABLE = True
except ImportError:
    TAUBENCH_AVAILABLE = False
    logging.warning("TauBench library not found. TauBench evaluation will be skipped.")

from llm.openai.openai_llm import OpenAILLM
from tools.serper_search import SerperSearchTool
from log.logging import LogManager
from dotenv import load_dotenv


def setup_agent(model: str = "gpt-3.5-turbo") -> ReActAgent:
    """
    Set up the agent to be evaluated.
    
    Args:
        model: The model to use for the agent
        
    Returns:
        A configured ReActAgent instance
    """
    # Initialize LLM
    llm = OpenAILLM.create_llm()
    
    # Create tools
    search_tool = SerperSearchTool(query="")
    
    # Create a log manager for the agent
    log_manager = LogManager(verbose=True)
    
    # Create the agent
    agent = ReActAgent(
        llm=llm,
        role="You are a helpful AI assistant.",
        task="Complete tasks accurately and efficiently.",
        guide="""Approach each task methodically and think step-by-step.
               Use available tools when necessary. Provide clear and concise responses.""",
        examples=[],
        tools=[search_tool],
        max_iterations=5,
        log_manager=log_manager
    )
    
    return agent


async def run_evaluations(
    agent: ReActAgent,
    model: str,
    frameworks: List[str] = None,
    output_dir: str = "evaluation_results",
    environment_limit: int = 3,
    task_limit: int = 2,
    turn_limit: int = 5
) -> Dict[str, Any]:
    """
    Run evaluations on the agent using specified frameworks.
    
    Args:
        agent: The agent to evaluate
        model: The model to use for evaluation
        frameworks: List of frameworks to use (default: all)
        output_dir: Directory to save results
        environment_limit: Limit number of environments/domains per framework
        task_limit: Limit number of tasks per environment/domain
        turn_limit: Limit number of turns per task (for multi-turn evaluations)
        
    Returns:
        Dictionary of evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default to all frameworks if none specified
    all_frameworks = ["agentbench", "agentboard"]
    if TAUBENCH_AVAILABLE:
        all_frameworks.append("taubench")
    frameworks = frameworks or all_frameworks
    
    # Initialize results dictionary
    results = {
        "agent_type": agent.__class__.__name__,
        "model": model,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frameworks": {}
    }
    
    # Run AgentBench evaluation if selected
    if "agentbench" in frameworks:
        print(f"\n{'='*50}")
        print(f"Running AgentBench evaluation on {agent.__class__.__name__} with model {model}")
        print(f"{'='*50}")
        
        # Get a subset of environments to test for speed
        environments = ["os", "db", "kg"][:environment_limit]
        
        agentbench_eval = AgentBenchEvaluation(
            agent=agent,
            model=model,
            environments=environments,
            task_limit=task_limit,
            name="agentbench_evaluation"
        )
        
        start_time = time.time()
        agentbench_results = await agentbench_eval.run_evaluation()
        elapsed_time = time.time() - start_time
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"agentbench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, "w") as f:
            json.dump(agentbench_results, f, indent=2)
        
        # Print summary
        summary = agentbench_eval.get_summary()
        print(f"\nAgentBench Evaluation Summary:")
        print(f"- Environments tested: {', '.join(summary.get('environments_tested', []))}")
        print(f"- Total tasks: {summary.get('total_tasks', 0)}")
        print(f"- Overall success rate: {summary.get('overall_success_rate', 0):.2%}")
        print(f"- Elapsed time: {elapsed_time:.2f} seconds")
        print(f"- Detailed results saved to: {results_file}")
        
        # Add to overall results
        results["frameworks"]["agentbench"] = {
            "summary": summary,
            "elapsed_time": elapsed_time,
            "results_file": results_file
        }
    
    # Run AgentBoard evaluation if selected
    if "agentboard" in frameworks:
        print(f"\n{'='*50}")
        print(f"Running AgentBoard evaluation on {agent.__class__.__name__} with model {model}")
        print(f"{'='*50}")
        
        # Get a subset of domains to test for speed
        domains = ["reasoning", "knowledge", "planning"][:environment_limit]
        
        agentboard_eval = AgentBoardEvaluation(
            agent=agent,
            model=model,
            domains=domains,
            task_limit=task_limit,
            turn_limit=turn_limit,
            name="agentboard_evaluation"
        )
        
        start_time = time.time()
        agentboard_results = await agentboard_eval.run_evaluation()
        elapsed_time = time.time() - start_time
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"agentboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, "w") as f:
            json.dump(agentboard_results, f, indent=2)
        
        # Print summary
        summary = agentboard_eval.get_summary()
        print(f"\nAgentBoard Evaluation Summary:")
        print(f"- Domains tested: {', '.join(summary.get('domains_tested', []))}")
        print(f"- Total tasks: {summary.get('total_tasks', 0)}")
        print(f"- Overall success rate: {summary.get('overall_success_rate', 0):.2%}")
        print(f"- Average progress rate: {summary.get('avg_progress_rate', 0):.2%}")
        print(f"- Average grounding accuracy: {summary.get('avg_grounding_accuracy', 0):.2%}")
        print(f"- Elapsed time: {elapsed_time:.2f} seconds")
        print(f"- Detailed results saved to: {results_file}")
        
        # Add to overall results
        results["frameworks"]["agentboard"] = {
            "summary": summary,
            "elapsed_time": elapsed_time,
            "results_file": results_file
        }
    
    # Run TauBench evaluation if selected and available
    if "taubench" in frameworks and TAUBENCH_AVAILABLE:
        print(f"\n{'='*50}")
        print(f"Running TauBench evaluation on {agent.__class__.__name__} with model {model}")
        print(f"{'='*50}")
        
        # Get a subset of categories to test for speed
        categories = ["reasoning", "planning", "tool_use", "qa"][:environment_limit]
        
        try:
            taubench_eval = TauBenchEvaluation(
                agent=agent,
                model=model,
                categories=categories,
                task_limit=task_limit,
                name="taubench_evaluation"
            )
            
            start_time = time.time()
            taubench_results = await taubench_eval.run_evaluation()
            elapsed_time = time.time() - start_time
            
            # Save detailed results
            results_file = os.path.join(output_dir, f"taubench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_file, "w") as f:
                json.dump(taubench_results, f, indent=2)
            
            # Print summary
            summary = taubench_eval.get_summary()
            print(f"\nTauBench Evaluation Summary:")
            print(f"- Categories tested: {', '.join(summary.get('categories_tested', []))}")
            print(f"- Total tasks: {summary.get('total_tasks', 0)}")
            print(f"- Overall success rate: {summary.get('overall_success_rate', 0):.2%}")
            print(f"- Elapsed time: {elapsed_time:.2f} seconds")
            print(f"- Detailed results saved to: {results_file}")
            
            # Add to overall results
            results["frameworks"]["taubench"] = {
                "summary": summary,
                "elapsed_time": elapsed_time,
                "results_file": results_file
            }
        except Exception as e:
            print(f"Error running TauBench evaluation: {e}")
            results["frameworks"]["taubench"] = {
                "error": str(e),
                "status": "failed"
            }
    
    # Save overall results
    overall_results_file = os.path.join(output_dir, f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(overall_results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Evaluation complete!")
    print(f"Overall results saved to: {overall_results_file}")
    print(f"{'='*50}")
    
    return results


def main():
    """Main entry point for the script."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run evaluations on agents")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use for evaluation")
    parser.add_argument("--frameworks", type=str, nargs="+", 
                        choices=["agentbench", "agentboard", "taubench"], 
                        help="Frameworks to use for evaluation")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--env-limit", type=int, default=3, help="Limit number of environments/domains per framework")
    parser.add_argument("--task-limit", type=int, default=2, help="Limit number of tasks per environment/domain")
    parser.add_argument("--turn-limit", type=int, default=5, help="Limit number of turns per task (for multi-turn evaluations)")
    args = parser.parse_args()
    
    # Set up the agent
    agent = setup_agent(model=args.model)
    
    # Run the evaluations
    asyncio.run(run_evaluations(
        agent=agent,
        model=args.model,
        frameworks=args.frameworks,
        output_dir=args.output_dir,
        environment_limit=args.env_limit,
        task_limit=args.task_limit,
        turn_limit=args.turn_limit
    ))


if __name__ == "__main__":
    main() 