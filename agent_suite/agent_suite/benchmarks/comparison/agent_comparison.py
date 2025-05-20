#!/usr/bin/env python
"""
Agent Comparison Utilities

This module provides utilities for comparing different agent implementations
using benchmarks like TauBench.
"""

import asyncio
import os
import json
from datetime import datetime
import time
import logging
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from ...agents.react.agent import ReActAgent
from ...benchmarks.taubench.taubench_eval import TauBenchEvaluation
from ...benchmarks.taubench.taubench_minimal import load_dataset, Evaluator, calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_comparison_charts(results, output_dir):
    """
    Generate comparison charts from results.
    
    Args:
        results: Dictionary of results for each agent type
        output_dir: Directory to save charts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for charts
    agent_names = list(results.keys())
    num_agents = len(agent_names)
    
    # Extract categories and scores
    categories = []
    category_scores = {agent: [] for agent in agent_names}
    
    # Collect all categories and scores
    for agent in agent_names:
        agent_results = results[agent]
        
        for category, scores in agent_results.get("category_metrics", {}).items():
            if category not in categories:
                categories.append(category)
            
            # Store category score for this agent
            category_scores[agent].append(scores.get("overall_score", 0))
    
    # Ensure all agents have scores for all categories (fill with zeros if missing)
    for agent in agent_names:
        if len(category_scores[agent]) < len(categories):
            category_scores[agent].extend([0] * (len(categories) - len(category_scores[agent])))
    
    # Generate category comparison chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.2
    index = np.arange(len(categories))
    
    # Plot bars for each agent
    for i, agent in enumerate(agent_names):
        ax.bar(index + i * bar_width, category_scores[agent], bar_width, label=agent)
    
    # Add labels and title
    ax.set_xlabel('Category')
    ax.set_ylabel('Score')
    ax.set_title('Agent Performance by Category')
    ax.set_xticks(index + bar_width * (num_agents - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels above bars
    for i, agent in enumerate(agent_names):
        for j, score in enumerate(category_scores[agent]):
            ax.text(j + i * bar_width, score + 0.05, f"{score:.2f}", ha='center')
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_comparison.png"))
    
    # Generate overall comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract overall scores
    overall_scores = []
    for agent in agent_names:
        score = results[agent].get("overall_score", 0)
        overall_scores.append(score)
    
    # Plot bars
    bars = ax.bar(agent_names, overall_scores, color=['#2C7FB8', '#7FCDBB', '#D95F02'])
    
    # Add labels and title
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Score')
    ax.set_title('Overall Agent Performance Comparison')
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05, f"{height:.2f}", ha='center')
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "taubench_comparison.png"))
    
    plt.close('all')  # Close all figures to free memory
    
    return {
        "category_chart": os.path.join(output_dir, "category_comparison.png"),
        "overall_chart": os.path.join(output_dir, "taubench_comparison.png")
    }

def generate_comparison_report(results, output_dir):
    """
    Generate HTML comparison report.
    
    Args:
        results: Dictionary of results for each agent type
        output_dir: Directory to save the report
    
    Returns:
        str: Path to the generated report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate charts and get their paths
    chart_paths = generate_comparison_charts(results, output_dir)
    
    # Create report HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TauBench Agent Comparison Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            h1, h2, h3 {
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .chart {
                margin: 30px 0;
                text-align: center;
            }
            .chart img {
                max-width: 100%;
                height: auto;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .highlightRow {
                background-color: #e6f7ff !important;
            }
            .timestamp {
                font-size: 0.8em;
                color: #666;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TauBench Agent Comparison Report</h1>
            <div class="timestamp">Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
            
            <h2>Overall Performance Comparison</h2>
            <div class="chart">
                <img src="taubench_comparison.png" alt="Overall Performance Comparison Chart">
            </div>
            
            <h2>Performance by Category</h2>
            <div class="chart">
                <img src="category_comparison.png" alt="Performance by Category Chart">
            </div>
            
            <h2>Detailed Results</h2>
    """
    
    # Add detailed results for each agent
    for agent_type, agent_results in results.items():
        html_content += f"""
            <h3>{agent_type}</h3>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Score</th>
                    <th>Correct</th>
                    <th>Incorrect</th>
                    <th>Success Rate</th>
                </tr>
        """
        
        # Add rows for each category
        for category, metrics in agent_results.get("category_metrics", {}).items():
            html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{metrics.get("overall_score", 0):.2f}</td>
                    <td>{metrics.get("correct", 0)}</td>
                    <td>{metrics.get("incorrect", 0)}</td>
                    <td>{metrics.get("success_rate", 0):.2%}</td>
                </tr>
            """
        
        # Add overall results row
        html_content += f"""
                <tr class="highlightRow">
                    <td><strong>Overall</strong></td>
                    <td><strong>{agent_results.get("overall_score", 0):.2f}</strong></td>
                    <td><strong>{agent_results.get("total_correct", 0)}</strong></td>
                    <td><strong>{agent_results.get("total_incorrect", 0)}</strong></td>
                    <td><strong>{agent_results.get("overall_success_rate", 0):.2%}</strong></td>
                </tr>
            </table>
        """
    
    # Close HTML document
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML content to file
    report_path = os.path.join(output_dir, "taubench_comparison_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    return report_path

async def compare_agents(
    agents_dict: Dict[str, ReActAgent],
    model_name: str,
    output_dir: str,
    task_limit: int = 5,
    category_limit: int = 2
):
    """
    Compare multiple agents using TauBench.
    
    Args:
        agents_dict: Dictionary mapping agent names to agent instances
        model_name: Name of the model to use
        output_dir: Directory to save evaluation results
        task_limit: Maximum number of tasks per category
        category_limit: Maximum number of categories to evaluate
        
    Returns:
        Dict[str, Any]: Comparison results
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results for each agent
    results = {}
    
    # Evaluate each agent
    for agent_name, agent in agents_dict.items():
        logger.info(f"Evaluating {agent_name}...")
        
        # Create a directory for this agent's results
        agent_dir = os.path.join(output_dir, agent_name.replace(" ", "_"))
        os.makedirs(agent_dir, exist_ok=True)
        
        # Run TauBench evaluation
        evaluation = TauBenchEvaluation(
            agent=agent,
            model=model_name,
            task_limit=task_limit,
            category_limit=category_limit,
            output_dir=agent_dir
        )
        
        # Run evaluation
        start_time = time.time()
        eval_results = await evaluation.run_evaluation()
        end_time = time.time()
        
        # Calculate total correct and incorrect across all categories
        total_correct = sum(metrics.get("correct", 0) for metrics in eval_results.get("category_metrics", {}).values())
        total_incorrect = sum(metrics.get("incorrect", 0) for metrics in eval_results.get("category_metrics", {}).values())
        
        # Calculate overall score and success rate
        if total_correct + total_incorrect > 0:
            overall_success_rate = total_correct / (total_correct + total_incorrect)
        else:
            overall_success_rate = 0
            
        # Calculate overall score as average of category scores
        category_scores = [metrics.get("overall_score", 0) for metrics in eval_results.get("category_metrics", {}).values()]
        if category_scores:
            overall_score = sum(category_scores) / len(category_scores)
        else:
            overall_score = 0
            
        # Add overall metrics to results
        eval_results["overall_score"] = overall_score
        eval_results["overall_success_rate"] = overall_success_rate
        eval_results["total_correct"] = total_correct
        eval_results["total_incorrect"] = total_incorrect
        eval_results["evaluation_time"] = end_time - start_time
        
        # Store results
        results[agent_name] = eval_results
        
        # Save results to file
        with open(os.path.join(agent_dir, "evaluation_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
            
        logger.info(f"Finished evaluating {agent_name}")
        logger.info(f"Overall score: {overall_score:.2f}")
        logger.info(f"Success rate: {overall_success_rate:.2%}")
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Generate comparison report
    report_path = generate_comparison_report(results, output_dir)
    logger.info(f"Comparison report generated: {report_path}")
    
    return results 