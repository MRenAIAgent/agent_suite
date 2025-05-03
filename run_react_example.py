#!/usr/bin/env python3
"""
Simple script to run the ReactAgentExample with enhanced logging.
This script demonstrates the use of the enhanced logging capabilities to track agent thoughts,
actions, and results in a structured and readable format.
"""

from agents.examples.react_agent_example import ReactAgentExample
from log.export import export_logs
import os

if __name__ == "__main__":
    # Create and run the agent
    agent = ReactAgentExample(max_iterations=30)
    
    # Example query
    query = "Comprehensive report on tesla stock prediction in 2025"
    
    print(f"\n=== Query: {query} ===\n")
    response = agent.run(query)
    print(f"\nResponse:\n{response}")
    
    # Access the log data programmatically
    log_manager = agent.agent.log_manager
    
    # Get statistics about the agent execution
    current_execution = log_manager.get_current_execution()
    thought_count = log_manager.get_thought_count()
    
    print("\n=== Execution Statistics ===")
    print(f"- Total thought steps: {thought_count}")
    print(f"- Total action steps: {len(current_execution['actions'])}")
    
    # Example of how to access and use the logged data programmatically
    print("\n=== Example: Accessing the First Thought ===")
    if current_execution["thoughts"]:
        first_thought = current_execution["thoughts"][0]
        print(f"First thought (Step {first_thought['step']}): {first_thought['content'][:100]}...")
    
    print("\n=== Example: Accessing the Last Action and its Result ===")
    if current_execution["actions"]:
        last_action = current_execution["actions"][-1]
        print(f"Last action (Step {last_action['step']}): {last_action['name']}")
        
        # Find the corresponding result for this action
        for result in current_execution["results"]:
            if result.get("related_action_step") == last_action["step"]:
                print(f"Result: {str(result['content'])[:100]}...")
                break
    
    # Export the logs to different formats
    print("\n=== Exporting Logs ===")
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Export to JSON
    json_path = export_logs(current_execution, "json", os.path.join(logs_dir, "last_execution.json"))
    print(f"Exported JSON log to: {json_path}")
    
    # Export to Markdown
    md_path = export_logs(current_execution, "markdown", os.path.join(logs_dir, "last_execution.md"))
    print(f"Exported Markdown log to: {md_path}")
    
    # Export to HTML
    html_path = export_logs(current_execution, "html", os.path.join(logs_dir, "last_execution.html"))
    print(f"Exported HTML log to: {html_path}")
    
    print("\nYou can open these files to view detailed logs of the agent's execution.") 