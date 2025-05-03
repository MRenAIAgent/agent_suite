#!/usr/bin/env python3
"""
Logging Demonstration Script

This script demonstrates the enhanced logging capabilities of the agent_suite,
showing how to log thoughts, actions, results, and final answers, and how to
export logs to different formats.
"""

import sys
import os
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from log.logging import LogManager
from log.export import export_logs

def simulate_agent_execution():
    """Simulate an agent execution with thoughts, actions, and results."""
    # Create a log manager
    log_manager = LogManager()
    
    # Simulate an agent thinking
    log_manager.log_thought("I need to analyze Tesla's stock performance. I'll start by searching for recent data.")
    
    # Simulate an action
    action_step = log_manager.log_action("search", {"query": "Tesla stock performance 2024"})
    
    # Simulate a result
    log_manager.log_result(action_step, "Found Tesla's stock has shown volatility in 2024 with significant price swings.")
    
    # Simulate another thought
    log_manager.log_thought("I should look for analyst predictions and financial metrics to make a more informed forecast.")
    
    # Simulate another action
    action_step = log_manager.log_action("search", {"query": "Tesla financial metrics revenue growth 2025 forecast"})
    
    # Simulate another result
    log_manager.log_result(action_step, "Analysts project continued growth in Tesla's revenue through 2025, with estimates ranging from 15-25% annual growth.")
    
    # Simulate a final thought
    log_manager.log_thought("Based on the financial data and analyst projections, I can now provide a comprehensive forecast for Tesla's stock in 2025.")
    
    # Simulate a tool action
    action_step = log_manager.log_action("financial_analyzer", {"ticker": "TSLA", "projection_year": 2025})
    
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
    print("\n===== Agent Logging Demonstration =====\n")
    
    # Simulate an agent execution
    log_manager = simulate_agent_execution()
    
    # Print the execution summary
    log_manager.print_execution_summary()
    
    # Export logs to different formats
    print("\n===== Exporting Logs =====\n")
    
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
    print("\nTry opening the HTML file in your web browser for a nicely formatted view of the execution.")
    
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