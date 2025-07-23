#!/usr/bin/env python3
"""
Logging Demonstration Script

This script demonstrates the enhanced logging capabilities of the agent_suite,
showing how to log thoughts, actions, results, and final answers, and how to
export logs to different formats. Now includes latency tracking features.
"""

import sys
import os
import time
import random
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from log.logging import LogManager
from log.export import export_logs

def simulate_llm_call():
    """Simulate a call to an LLM with realistic latency."""
    # Sleep for a random time between 1-3 seconds to simulate LLM processing
    latency = random.uniform(1, 3)
    time.sleep(latency)
    return latency

def simulate_action_execution():
    """Simulate execution of a tool action with realistic latency."""
    # Sleep for a random time between 0.2-1.5 seconds to simulate action execution
    latency = random.uniform(0.2, 1.5)
    time.sleep(latency)
    return latency

def simulate_agent_execution():
    """Simulate an agent execution with thoughts, actions, and results, including latency."""
    # Create a log manager
    log_manager = LogManager()
    
    # Start timing the total execution
    log_manager.start_execution()
    
    # Simulate LLM call for initial reasoning
    llm_latency = simulate_llm_call()
    log_manager.log_llm_latency(llm_latency)
    
    # Simulate an agent thinking
    log_manager.log_thought("I need to analyze Tesla's stock performance. I'll start by searching for recent data.")
    
    # Simulate an action (search tool)
    action_step = log_manager.log_action("search", {"query": "Tesla stock performance 2024"})
    
    # Simulate executing the action and getting a result
    simulate_action_execution()
    log_manager.log_result(action_step, "Found Tesla's stock has shown volatility in 2024 with significant price swings.")
    
    # Simulate another LLM call for next thought
    llm_latency = simulate_llm_call()
    log_manager.log_llm_latency(llm_latency)
    
    # Simulate another thought
    log_manager.log_thought("I should look for analyst predictions and financial metrics to make a more informed forecast.")
    
    # Simulate another action (search for metrics)
    action_step = log_manager.log_action("search", {"query": "Tesla financial metrics revenue growth 2025 forecast"})
    
    # Simulate executing the action and getting a result
    simulate_action_execution()
    log_manager.log_result(action_step, "Analysts project continued growth in Tesla's revenue through 2025, with estimates ranging from 15-25% annual growth.")
    
    # Simulate another LLM call for next thought
    llm_latency = simulate_llm_call()
    log_manager.log_llm_latency(llm_latency)
    
    # Simulate a final thought
    log_manager.log_thought("Based on the financial data and analyst projections, I can now provide a comprehensive forecast for Tesla's stock in 2025.")
    
    # Simulate a tool action (financial analyzer)
    action_step = log_manager.log_action("financial_analyzer", {"ticker": "TSLA", "projection_year": 2025})
    
    # Simulate executing the action and getting a complex result (slower execution)
    simulate_action_execution()  # Longer latency for complex action
    
    # Simulate complex result
    result_data = {
        "projected_price_range": {"low": 180, "high": 320},
        "key_factors": [
            "EV market competition",
            "Battery technology advancements",
            "Global economic conditions",
            "Production scalability"
        ],
        "confidence_level": "moderate"
    }
    log_manager.log_result(action_step, result_data)
    
    # Simulate final LLM call for generating final answer
    llm_latency = simulate_llm_call()
    log_manager.log_llm_latency(llm_latency)
    
    # Simulate final answer
    final_answer = """
    Tesla Stock Prediction for 2025:
    
    Based on comprehensive analysis of historical data, financial metrics, and industry trends, Tesla's stock is likely to trade in a range of $180-$320 by the end of 2025. 
    
    Key factors influencing this prediction:
    1. Increasing competition in the EV market may pressure margins
    2. Potential breakthroughs in battery technology could provide upside
    3. Global economic conditions will impact luxury vehicle demand
    4. Production scalability remains critical for meeting growth targets
    
    Investors should monitor quarterly deliveries, margin trends, and progress on new models and technologies as leading indicators for stock performance.
    
    This prediction comes with a moderate confidence level due to the inherent volatility of Tesla's stock and the rapidly evolving EV market landscape.
    """
    
    log_manager.log_final_answer(final_answer)
    
    return log_manager


if __name__ == "__main__":
    print("\n===== Agent Logging Demonstration with Latency Tracking =====\n")
    
    # Simulate an agent execution
    log_manager = simulate_agent_execution()
    
    # Print the execution summary
    log_manager.print_execution_summary()
    
    # Get latency metrics
    latency_metrics = log_manager.get_latency_metrics()
    
    # Print latency summary
    print("\n===== Latency Metrics Summary =====\n")
    print(f"Total execution time: {latency_metrics.get('total', 0):.2f}s")
    
    if 'llm_total' in latency_metrics:
        print(f"Total LLM processing time: {latency_metrics.get('llm_total', 0):.2f}s")
        print(f"Average LLM latency: {latency_metrics.get('llm_avg', 0):.2f}s per call")
    
    if 'action_total' in latency_metrics:
        print(f"Total action execution time: {latency_metrics.get('action_total', 0):.2f}s")
        print(f"Average action latency: {latency_metrics.get('action_avg', 0):.2f}s per action")
    
    if 'total' in latency_metrics and 'llm_total' in latency_metrics and 'action_total' in latency_metrics:
        overhead = latency_metrics['total'] - (latency_metrics['llm_total'] + latency_metrics['action_total'])
        overhead_percent = (overhead / latency_metrics['total']) * 100 if latency_metrics['total'] > 0 else 0
        print(f"Overhead: {overhead:.2f}s ({overhead_percent:.1f}% of total time)")
    
    # Export logs to different formats
    print("\n===== Exporting Logs with Latency Data =====\n")
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get the execution data
    current_execution = log_manager.get_current_execution()
    
    # Export to different formats with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export to JSON
    json_path = export_logs(current_execution, "json", os.path.join(logs_dir, f"demo_execution_{timestamp}.json"))
    print(f"Exported JSON log to: {json_path}")
    
    # Export to Markdown
    md_path = export_logs(current_execution, "markdown", os.path.join(logs_dir, f"demo_execution_{timestamp}.md"))
    print(f"Exported Markdown log to: {md_path}")
    
    # Export to HTML
    html_path = export_logs(current_execution, "html", os.path.join(logs_dir, f"demo_execution_{timestamp}.html"))
    print(f"Exported HTML log to: {html_path}")
    
    print("\nYou can open these files to view detailed logs of the agent's execution.")
    print("\nTry opening the HTML file in your web browser to see the latency visualization chart.")
    
    # Show how to access logs programmatically
    print("\n===== Accessing Logs Programmatically =====\n")
    
    print(f"Total thought steps: {log_manager.get_thought_count()}")
    print(f"Total action steps: {len(current_execution['actions'])}")
    
    print("\nFirst thought:")
    if current_execution["thoughts"]:
        first_thought = current_execution["thoughts"][0]
        print(f"- Step {first_thought['step']}: {first_thought['content']}")
    
    print("\nExample action input and result:")
    if current_execution["actions"]:
        example_action = current_execution["actions"][-1]  # Last action
        action_name = example_action.get("name", "Unknown")
        action_input = example_action.get("input", {})
        print(f"- Action: {action_name}")
        print(f"- Input: {action_input}")
        
        # Find corresponding result
        for result in current_execution["results"]:
            if result.get("related_action_step") == example_action["step"]:
                print(f"- Result: {result['content']}")
                break 