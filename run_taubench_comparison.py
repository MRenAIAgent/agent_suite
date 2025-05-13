#!/usr/bin/env python
"""
TauBench Comparison Script

This script runs TauBench evaluations on multiple agent types for comparison:
1. Custom ReActAgent
2. LangChain ReAct agent
3. LangChain Structured agent

It generates a comparison report showing the performance of each agent type.
"""

import asyncio
import argparse
import os
import json
from datetime import datetime
import time
import logging
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# Import agent components
from agents.react_agent import ReActAgent
from agents.examples.react_agent_example import ReactAgentExample
from log.logging import LogManager

# Import tools from project
from tools.serper_api import SerperSearchTool
from tools.adapter.langchain_tool import convert_langchain_tools

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool as LangChainTool
from langchain.memory import ConversationBufferMemory

# Import TauBench evaluation
from agents.eval.taubench.taubench_eval import TauBenchEvaluation
from agents.eval.taubench.taubench_minimal import load_dataset, Evaluator, calculate_metrics

# Import LangChain tools for examples
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain Agent Wrapper for compatibility with TauBench
class LangChainAgentWrapper:
    """
    Wrapper for LangChain agents to use with TauBench.
    """
    
    def __init__(self, agent_executor, agent_type_name):
        self.agent_executor = agent_executor
        self.agent_type_name = agent_type_name
        self.log_manager = LogManager()
        
    def run(self, task: str) -> str:
        """
        Run the agent on a task.
        
        Args:
            task: Task description
            
        Returns:
            Agent's response
        """
        try:
            # Run the agent on the task
            response = self.agent_executor.run(task)
            
            # Return the response
            return response
        except Exception as e:
            logger.error(f"Error running LangChain {self.agent_type_name} agent: {str(e)}")
            return f"Error: {str(e)}"
            
    async def arun(self, user_input: str, model: str, **kwargs) -> str:
        """
        Async version of run for compatibility with TauBench.
        
        Args:
            user_input: The user's question or request
            model: The LLM model to use (ignored, using predefined model)
            **kwargs: Additional parameters
            
        Returns:
            str: The agent's final response
        """
        try:
            # Run the agent on the task
            return self.run(user_input)
        except Exception as e:
            logger.error(f"Error running LangChain {self.agent_type_name} agent asynchronously: {str(e)}")
            return f"Error: {str(e)}"
    
    # Method to make it look like a ReActAgent for the evaluator
    def __class__(self):
        return type("ReActAgent", (), {})

# Agent Creation Functions
def create_custom_react_agent(model_name="gpt-4"):
    """
    Create the custom ReActAgent.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        ReActAgent: The custom ReActAgent
    """
    # Create and return React Agent example
    return ReactAgentExample(model=model_name)

def create_langchain_react_agent(model_name="gpt-4"):
    """
    Create a LangChain ReAct agent.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        LangChainAgentWrapper: Wrapped LangChain agent
    """
    # Initialize the language model
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize tools - using common search tools
    tools = [
        # Only use LangChain compatible tools
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        DuckDuckGoSearchRun()
    ]
    
    # Initialize the agent
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False
    )
    
    # Return wrapped agent
    return LangChainAgentWrapper(agent_executor, "ReAct")

def create_langchain_structured_agent(model_name="gpt-4"):
    """
    Create a LangChain Structured Tool agent.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        LangChainAgentWrapper: Wrapped LangChain agent
    """
    # Initialize the language model
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    # Initialize tools - using common search tools
    tools = [
        # Only use LangChain compatible tools
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        DuckDuckGoSearchRun()
    ]
    
    # Initialize the agent
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    
    # Return wrapped agent
    return LangChainAgentWrapper(agent_executor, "Structured")

async def evaluate_agent(
    agent_type: str,
    agent,
    model_name: str,
    output_dir: str,
    task_limit: int = 5,
    category_limit: int = 2
):
    """
    Run TauBench evaluation on an agent.
    
    Args:
        agent_type: Name of the agent type
        agent: The agent to evaluate
        model_name: Name of the model to use
        output_dir: Directory to save evaluation results
        task_limit: Maximum number of tasks per category
        category_limit: Maximum number of categories to evaluate
        
    Returns:
        Dict with evaluation results
    """
    # Create agent-specific output directory
    agent_dir = os.path.join(output_dir, agent_type)
    os.makedirs(agent_dir, exist_ok=True)
    
    logger.info(f"Running TauBench evaluation for {agent_type} with {model_name}")
    
    # Create evaluator
    evaluator = TauBenchEvaluation(
        agent=agent,
        model=model_name,
        task_limit=task_limit,
        category_limit=category_limit
    )
    
    # Run evaluation
    start_time = datetime.now()
    results = await evaluator.run_evaluation()
    end_time = datetime.now()
    
    # Calculate elapsed time
    elapsed_time = (end_time - start_time).total_seconds()
    
    # Get summary
    summary = evaluator.get_summary()
    
    # Print summary
    logger.info(f"TauBench Evaluation for {agent_type} completed in {elapsed_time:.2f}s")
    logger.info(f"Total tasks: {summary.get('total_tasks', 0)}")
    logger.info(f"Success rate: {summary.get('overall_success_rate', 0):.2%}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(agent_dir, f"taubench_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(agent_dir, f"evaluation_summary_{timestamp}.json")
    summary_data = {
        "agent_type": agent_type,
        "model": model_name,
        "evaluation_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "frameworks": {
            "taubench": {
                "summary": summary,
                "elapsed_time": elapsed_time,
                "results_file": results_file
            }
        }
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return {
        "agent_type": agent_type,
        "summary": summary,
        "elapsed_time": elapsed_time,
        "results_file": results_file
    }

def generate_comparison_charts(results, output_dir):
    """
    Generate comparison charts from evaluation results.
    
    Args:
        results: List of results from evaluations
        output_dir: Directory to save charts
    """
    # Extract data
    agent_names = [r["agent_type"] for r in results]
    success_rates = [r["summary"]["overall_success_rate"] for r in results]
    execution_times = [r["elapsed_time"] for r in results]
    
    # Set up plots
    plt.figure(figsize=(12, 10))
    
    # Success rate comparison
    plt.subplot(2, 1, 1)
    bars = plt.bar(agent_names, success_rates, color=['blue', 'green', 'orange'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2%}', ha='center', va='bottom')
    
    plt.title('TauBench Success Rate Comparison')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1.1)  # Limit y-axis to 0-110%
    
    # Execution time comparison
    plt.subplot(2, 1, 2)
    bars = plt.bar(agent_names, execution_times, color=['blue', 'green', 'orange'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.2f}s', ha='center', va='bottom')
    
    plt.title('Execution Time Comparison')
    plt.ylabel('Time (seconds)')
    
    # Save the combined chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'taubench_comparison.png'))
    
    # Generate per-category success rate comparison
    plt.figure(figsize=(12, 8))
    
    # Get all categories
    all_categories = set()
    for r in results:
        all_categories.update(r["summary"]["category_success_rates"].keys())
    all_categories = sorted(list(all_categories))
    
    # Extract per-category success rates
    category_data = {agent: [] for agent in agent_names}
    category_indices = np.arange(len(all_categories))
    
    for i, agent in enumerate(agent_names):
        for category in all_categories:
            rate = results[i]["summary"]["category_success_rates"].get(category, 0)
            category_data[agent].append(rate)
    
    # Set width of bars
    bar_width = 0.25
    
    # Plot bars for each agent
    for i, (agent, rates) in enumerate(category_data.items()):
        offset = (i - 1) * bar_width
        bars = plt.bar(category_indices + offset, rates, bar_width,
                       label=agent, alpha=0.7)
        
        # Add values on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.2%}', ha='center', va='bottom', fontsize=8)
    
    plt.title('Success Rate by Category')
    plt.xlabel('Category')
    plt.ylabel('Success Rate')
    plt.xticks(category_indices, all_categories)
    plt.ylim(0, 1.1)  # Limit y-axis to 0-110%
    plt.legend()
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_comparison.png'))
    
    logger.info(f"Comparison charts saved to {output_dir}")

def generate_comparison_report(results, output_dir):
    """
    Generate a comparison report as HTML.
    
    Args:
        results: List of results from evaluations
        output_dir: Directory to save the report
    """
    # Extract data
    agent_names = [r["agent_type"] for r in results]
    success_rates = [r["summary"]["overall_success_rate"] for r in results]
    execution_times = [r["elapsed_time"] for r in results]
    
    # Get all categories
    all_categories = set()
    for r in results:
        all_categories.update(r["summary"]["category_success_rates"].keys())
    all_categories = sorted(list(all_categories))
    
    # Create HTML report
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TauBench Agent Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .chart { margin: 20px 0; max-width: 100%; }
            .summary { margin: 20px 0; }
            .winner { background-color: #d4edda; }
        </style>
    </head>
    <body>
        <h1>TauBench Agent Comparison</h1>
        <p>Evaluation Time: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <h2>Overall Performance Summary</h2>
        <table>
            <tr>
                <th>Agent</th>
                <th>Success Rate</th>
                <th>Execution Time</th>
            </tr>
    """
    
    # Add rows for each agent
    best_rate_idx = success_rates.index(max(success_rates))
    fastest_idx = execution_times.index(min(execution_times))
    
    for i, agent in enumerate(agent_names):
        rate_class = " class='winner'" if i == best_rate_idx else ""
        time_class = " class='winner'" if i == fastest_idx else ""
        
        html += f"""
            <tr>
                <td>{agent}</td>
                <td{rate_class}>{success_rates[i]:.2%}</td>
                <td{time_class}>{execution_times[i]:.2f}s</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Success Rate by Category</h2>
        <table>
            <tr>
                <th>Category</th>
    """
    
    # Add column headers for each agent
    for agent in agent_names:
        html += f"<th>{agent}</th>"
    
    html += "</tr>"
    
    # Add rows for each category
    for category in all_categories:
        html += f"<tr><td>{category}</td>"
        
        for i, agent in enumerate(agent_names):
            rate = results[i]["summary"]["category_success_rates"].get(category, 0)
            html += f"<td>{rate:.2%}</td>"
        
        html += "</tr>"
    
    html += """
        </table>
        
        <h2>Comparison Charts</h2>
        <div class="chart">
            <img src="taubench_comparison.png" alt="TauBench Performance Comparison" width="800">
        </div>
        <div class="chart">
            <img src="category_comparison.png" alt="Category Success Rate Comparison" width="800">
        </div>
        
        <h2>Detailed Results</h2>
    """
    
    # Add links to detailed results for each agent
    for i, agent in enumerate(agent_names):
        html += f"""
        <div class="summary">
            <h3>{agent}</h3>
            <p>Success Rate: {success_rates[i]:.2%}</p>
            <p>Execution Time: {execution_times[i]:.2f}s</p>
            <p>Detailed Results: <a href="{results[i]['results_file']}">{results[i]['results_file']}</a></p>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    # Save the report
    report_file = os.path.join(output_dir, 'taubench_comparison_report.html')
    with open(report_file, 'w') as f:
        f.write(html)
    
    logger.info(f"Comparison report saved to {report_file}")

async def run_comparison(
    model_name: str,
    output_dir: str,
    task_limit: int = 5,
    category_limit: int = 2,
    agent_types: List[str] = ["custom_react", "langchain_react", "langchain_structured"]
):
    """
    Run TauBench evaluations on multiple agent types and generate comparison.
    
    Args:
        model_name: Name of the model to use
        output_dir: Directory to save evaluation results
        task_limit: Maximum number of tasks per category
        category_limit: Maximum number of categories to evaluate
        agent_types: List of agent types to evaluate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results list
    all_results = []
    
    # Evaluate each agent type
    for agent_type in agent_types:
        try:
            # Create agent based on type
            if agent_type == "custom_react":
                agent = create_custom_react_agent(model_name)
                agent_name = "Custom ReAct"
            elif agent_type == "langchain_react":
                agent = create_langchain_react_agent(model_name)
                agent_name = "LangChain ReAct"
            elif agent_type == "langchain_structured":
                agent = create_langchain_structured_agent(model_name)
                agent_name = "LangChain Structured"
            else:
                logger.warning(f"Unknown agent type: {agent_type}, skipping...")
                continue
            
            # Run evaluation
            result = await evaluate_agent(
                agent_type=agent_name,
                agent=agent,
                model_name=model_name,
                output_dir=output_dir,
                task_limit=task_limit,
                category_limit=category_limit
            )
            
            # Add to results
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Error evaluating {agent_type}: {str(e)}")
    
    # Generate comparison if we have results for multiple agents
    if len(all_results) > 1:
        try:
            generate_comparison_charts(all_results, output_dir)
            generate_comparison_report(all_results, output_dir)
        except Exception as e:
            logger.error(f"Error generating comparison: {str(e)}")
    else:
        logger.warning("Not enough agents evaluated to generate comparison")

async def main():
    """Main function to parse arguments and run comparison."""
    parser = argparse.ArgumentParser(description="Run TauBench comparison across agent types")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="Name of the model to use")
    parser.add_argument("--output-dir", type=str, default="taubench_comparison",
                       help="Directory to save evaluation results")
    parser.add_argument("--task-limit", type=int, default=5,
                       help="Maximum number of tasks per category")
    parser.add_argument("--category-limit", type=int, default=2,
                       help="Maximum number of categories to evaluate")
    parser.add_argument("--agents", nargs="+", 
                       default=["custom_react", "langchain_react", "langchain_structured"],
                       help="Agent types to evaluate")
    
    args = parser.parse_args()
    
    await run_comparison(
        model_name=args.model,
        output_dir=args.output_dir,
        task_limit=args.task_limit,
        category_limit=args.category_limit,
        agent_types=args.agents
    )

if __name__ == "__main__":
    asyncio.run(main()) 