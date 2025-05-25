#!/usr/bin/env python
"""
SSH Key Management Evaluation

Script to evaluate an agent's ability to perform SSH key management tasks.
"""

import argparse
import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

from agents.react_agent import ReActAgent
from agents.eval.taubench.taubench_eval import TauBenchEvaluation
from agents.eval.taubench.ssh_key_tasks import get_ssh_key_tasks, validate_ssh_key_task
from llm.openai.openai_llm import OpenAILLM
from tools.serper_search import SerperSearchTool
from log.logging import LogManager
from dotenv import load_dotenv

class SSHKeyEvaluation(TauBenchEvaluation):
    """
    Evaluation specifically for SSH key management tasks.
    """
    
    def __init__(
        self, 
        agent: ReActAgent, 
        model: str,
        task_limit: int = 5,
        name: str = "ssh_key_eval"
    ):
        """
        Initialize the SSH key evaluation.
        
        Args:
            agent: The ReActAgent to evaluate
            model: The model to use for evaluation
            task_limit: Maximum number of tasks to run
            name: Name of the evaluation
        """
        # Call parent constructor with only the "tool_use" category
        super().__init__(
            agent=agent,
            model=model,
            categories=["tool_use"],
            task_limit=task_limit,
            name=name,
            use_reference_model=False
        )
        
        # Override data loader to use our SSH key tasks
        self.ssh_key_tasks = get_ssh_key_tasks()[:task_limit]
    
    async def run_evaluation(self):
        """Run the evaluation on SSH key management tasks."""
        self.start_time = time.time()
        
        # Initialize results for the tool_use category
        self.results["categories"]["tool_use"] = {
            "name": "SSH Key Management",
            "total_tasks": len(self.ssh_key_tasks),
            "successful_tasks": 0,
            "tasks": []
        }
        
        # Run each SSH key task
        for task in self.ssh_key_tasks:
            task_id = task.get("id")
            instruction = task.get("instruction", "")
            reference = task.get("reference", "")
            
            task_result = {
                "id": task_id,
                "instruction": instruction,
                "reference": reference,
                "start_time": time.time()
            }
            
            try:
                # Run the agent on the task
                response = await self.agent.arun(instruction, self.model)
                
                # In a real scenario, we would execute the command and validate
                # For evaluation purposes, we'll compare the response with reference
                # and use validation criteria
                
                # Check if the response correctly addresses the task
                success, details = validate_ssh_key_task(task_id, response)
                
                task_result["success"] = success
                task_result["response"] = response
                task_result["validation_details"] = details
                task_result["iterations"] = getattr(self.agent, "iterations", None)
                
                if success:
                    self.results["categories"]["tool_use"]["successful_tasks"] += 1
                
            except Exception as e:
                task_result["success"] = False
                task_result["error"] = str(e)
            
            task_result["end_time"] = time.time()
            task_result["duration"] = task_result["end_time"] - task_result["start_time"]
            
            self.results["categories"]["tool_use"]["tasks"].append(task_result)
        
        # Calculate category metrics
        category_results = self.results["categories"]["tool_use"]
        if category_results["total_tasks"] > 0:
            category_results["success_rate"] = category_results["successful_tasks"] / category_results["total_tasks"]
            category_results["avg_task_duration"] = sum(task.get("duration", 0) for task in category_results["tasks"]) / category_results["total_tasks"]
        else:
            category_results["success_rate"] = 0
            category_results["avg_task_duration"] = 0
        
        # Calculate overall metrics
        self._calculate_summary_metrics()
        
        self.end_time = time.time()
        self.results["total_time"] = self.end_time - self.start_time
        
        return self.results


def setup_agent(model: str = "gpt-3.5-turbo") -> ReActAgent:
    """
    Set up the agent to be evaluated on SSH key tasks.
    
    Args:
        model: The model to use for the agent
        
    Returns:
        A configured ReActAgent instance
    """
    # Initialize LLM
    llm = OpenAILLM.create_llm()
    
    # Create tools - we need search to find SSH key commands
    search_tool = SerperSearchTool(query="")
    
    # Create a log manager for the agent
    log_manager = LogManager(verbose=True)
    
    # Create the agent with a focus on command-line tasks
    agent = ReActAgent(
        llm=llm,
        role="You are a helpful AI assistant specializing in command-line operations and system administration.",
        task="Assist with SSH key management and configuration tasks accurately and securely.",
        guide="""
        Approach SSH key tasks methodically and securely.
        Be precise with command syntax and parameters.
        Consider security best practices such as proper permissions.
        Provide complete commands that can be executed directly.
        """,
        examples=[],
        tools=[search_tool],
        max_iterations=5,
        log_manager=log_manager
    )
    
    return agent


async def run_ssh_key_evaluation(
    agent: ReActAgent,
    model: str,
    output_dir: str = "evaluation_results",
    task_limit: int = 5
) -> Dict[str, Any]:
    """
    Run SSH key management evaluation on the agent.
    
    Args:
        agent: The agent to evaluate
        model: The model to use for evaluation
        output_dir: Directory to save results
        task_limit: Maximum number of tasks to run
        
    Returns:
        Dictionary of evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "agent_type": agent.__class__.__name__,
        "model": model,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "focus": "SSH Key Management"
    }
    
    print(f"\n{'='*50}")
    print(f"Running SSH Key Management evaluation on {agent.__class__.__name__} with model {model}")
    print(f"{'='*50}")
    
    # Create and run SSH key evaluator
    ssh_eval = SSHKeyEvaluation(
        agent=agent,
        model=model,
        task_limit=task_limit,
        name="ssh_key_evaluation"
    )
    
    start_time = time.time()
    ssh_results = await ssh_eval.run_evaluation()
    elapsed_time = time.time() - start_time
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"ssh_key_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, "w") as f:
        json.dump(ssh_results, f, indent=2)
    
    # Print summary
    summary = ssh_eval.get_summary()
    print(f"\nSSH Key Management Evaluation Summary:")
    print(f"- Total tasks: {summary.get('total_tasks', 0)}")
    print(f"- Successful tasks: {summary.get('successful_tasks', 0)}")
    print(f"- Overall success rate: {summary.get('overall_success_rate', 0):.2%}")
    print(f"- Elapsed time: {elapsed_time:.2f} seconds")
    print(f"- Detailed results saved to: {results_file}")
    
    # Add to overall results
    results["summary"] = summary
    results["elapsed_time"] = elapsed_time
    results["results_file"] = results_file
    
    print(f"\n{'='*50}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*50}")
    
    return results


def main():
    """Main entry point for the script."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SSH key management evaluation on agents")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use for evaluation")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--task-limit", type=int, default=5, help="Maximum number of SSH key tasks to run")
    args = parser.parse_args()
    
    # Set up the agent
    agent = setup_agent(model=args.model)
    
    # Run the evaluation
    asyncio.run(run_ssh_key_evaluation(
        agent=agent,
        model=args.model,
        output_dir=args.output_dir,
        task_limit=args.task_limit
    ))


if __name__ == "__main__":
    main() 