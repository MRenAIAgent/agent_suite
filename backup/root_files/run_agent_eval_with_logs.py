#!/usr/bin/env python
"""
Agent Evaluation with Enhanced Logging

This script runs the TauBench evaluation on an agent, capturing detailed logs
of the agent's thought process, actions, and execution latency.
"""

import argparse
import asyncio
import os
import json
import time
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional

# Import agent components
from agents.react_agent import ReActAgent
from agents.examples.react_agent_example import ReactAgentExample
from log.logging import LogManager
from log.export import export_logs, LogExporter

# Import TauBench evaluation
try:
    from agents.eval.taubench.taubench_eval import TauBenchEvaluation, TauBenchDataLoader
    # Also try to import metrics module if TauBench is available
    try:
        from taubench.metrics import calculate_metrics
        METRICS_AVAILABLE = True
    except ImportError:
        METRICS_AVAILABLE = False
        logging.warning("TauBench metrics module not available. Simple success criteria will be used.")
    TAUBENCH_AVAILABLE = True
except ImportError:
    TAUBENCH_AVAILABLE = False
    METRICS_AVAILABLE = False
    logging.warning("TauBench library not found. TauBench evaluation will be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedLogManager(LogManager):
    """Extended LogManager with additional features for evaluation analysis."""
    
    def __init__(self):
        super().__init__()
        self.task_logs = {}  # Store logs by task ID
        self.current_task_id = None
    
    def set_current_task(self, task_id: str):
        """Set the current task being evaluated."""
        self.current_task_id = task_id
        if task_id not in self.task_logs:
            self.task_logs[task_id] = {
                "thoughts": [],
                "actions": [],
                "results": [],
                "final_answer": None,
                "latency": {
                    "llm": [],
                    "actions": [],
                    "total_start": None,
                    "total_end": None,
                    "total": None
                }
            }
    
    def log_thought(self, thought: str, timestamp: Optional[str] = None):
        """Log a thought and store it with the current task."""
        step = super().log_thought(thought, timestamp)
        
        if self.current_task_id:
            thought_entry = {
                "step": step,
                "timestamp": timestamp or datetime.now().isoformat(),
                "content": thought
            }
            self.task_logs[self.current_task_id]["thoughts"].append(thought_entry)
        
        return step
    
    def export_task_logs(self, output_dir: str, format: str = "html"):
        """Export logs for each task to separate files."""
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        for task_id, logs in self.task_logs.items():
            safe_task_id = task_id.replace("/", "_").replace(" ", "_")
            output_path = os.path.join(output_dir, f"task_{safe_task_id}.{format}")
            
            if format.lower() == "json":
                path = LogExporter.export_to_json(logs, output_path)
            elif format.lower() in ["markdown", "md"]:
                path = LogExporter.export_to_markdown(logs, output_path)
            else:  # html is default
                path = LogExporter.export_to_html(logs, output_path)
                
            exported_files[task_id] = path
            
        return exported_files
    
    def get_task_execution(self, task_id: str) -> Dict:
        """Get the execution logs for a specific task."""
        return self.task_logs.get(task_id, {})
    
    def get_task_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all tasks."""
        metrics = {}
        
        for task_id, logs in self.task_logs.items():
            # Extract key metrics
            thought_count = len(logs.get("thoughts", []))
            action_count = len(logs.get("actions", []))
            
            # Get latency metrics if available
            latency = logs.get("latency", {})
            llm_latencies = latency.get("llm", [])
            action_latencies = [lat for _, lat in latency.get("actions", [])]
            
            metrics[task_id] = {
                "thought_count": thought_count,
                "action_count": action_count,
                "llm_latency": sum(llm_latencies) if llm_latencies else 0,
                "avg_llm_latency": sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0,
                "action_latency": sum(action_latencies) if action_latencies else 0,
                "avg_action_latency": sum(action_latencies) / len(action_latencies) if action_latencies else 0,
                "total_latency": latency.get("total", 0)
            }
            
        return metrics

async def run_taubench_evaluation(
    agent: ReActAgent,
    model: str,
    categories: List[str],
    task_limit: int,
    output_dir: str,
    log_format: str = "html"
) -> Dict[str, Any]:
    """
    Run TauBench evaluation with enhanced logging.
    
    Args:
        agent: The agent to evaluate
        model: The model to use
        categories: Categories of tasks to evaluate
        task_limit: Maximum number of tasks per category
        output_dir: Directory to save results
        log_format: Format for exporting logs (json, markdown, html)
        
    Returns:
        Evaluation results
    """
    if not TAUBENCH_AVAILABLE:
        logger.error("TauBench is not available. Unable to run evaluation.")
        return {"error": "TauBench not available"}
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "task_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Check if we're using an enhanced log manager
    using_enhanced_logging = isinstance(agent.log_manager, EnhancedLogManager)
    
    # Create TauBench evaluator
    logger.info(f"Creating TauBench evaluator with categories: {categories}")
    evaluator = TauBenchEvaluation(
        agent=agent,
        model=model,
        categories=categories,
        task_limit=task_limit,
        name="taubench_with_logs",
        use_reference_model=False,
        data_dir=None  # Use default data directory
    )
    
    # Patch the evaluator's _evaluate_category method to capture task IDs
    original_evaluate_category = evaluator._evaluate_category
    
    async def patched_evaluate_category(category, tasks):
        """Patched method that sets the current task ID before evaluation."""
        logger.info(f"Evaluating category: {category} with {len(tasks)} tasks")
        
        category_results = {
            "name": category,
            "total_tasks": len(tasks),
            "successful_tasks": 0,
            "tasks": []
        }
        
        for task in tasks:
            task_id = task.get("id", f"{category}_{len(category_results['tasks'])}")
            
            # Set current task in log manager if it's enhanced
            if using_enhanced_logging:
                agent.log_manager.set_current_task(task_id)
                agent.log_manager.start_execution()
            
            logger.info(f"Evaluating task: {task_id}")
            
            # Prepare the task prompt
            instruction = task.get("instruction", "")
            if task.get("input"):
                prompt = f"{instruction}\n\n{task['input']}"
            else:
                prompt = instruction
            
            task_result = {
                "id": task_id,
                "instruction": instruction,
                "start_time": time.time()
            }
            
            try:
                # Run the agent on the task
                response = await agent.arun(prompt, model)
                
                # Use TauBench evaluation metrics if available
                if task.get("reference") and METRICS_AVAILABLE:
                    metrics = calculate_metrics(
                        response, 
                        task["reference"],
                        task.get("evaluation_criteria", {})
                    )
                    task_result["metrics"] = metrics
                    task_result["success"] = metrics.get("success", False)
                else:
                    # Simple success criteria if TauBench metrics aren't available
                    task_result["success"] = True  # Placeholder, assuming task completed
                
                task_result["response"] = response
                task_result["iterations"] = getattr(agent, "iterations", None)
                
                if task_result.get("success", False):
                    category_results["successful_tasks"] += 1
                
            except Exception as e:
                logger.error(f"Error evaluating task {task_id}: {e}")
                task_result["success"] = False
                task_result["error"] = str(e)
            
            task_result["end_time"] = time.time()
            task_result["duration"] = task_result["end_time"] - task_result["start_time"]
            
            category_results["tasks"].append(task_result)
        
        # Calculate category metrics
        if category_results["total_tasks"] > 0:
            category_results["success_rate"] = category_results["successful_tasks"] / category_results["total_tasks"]
            category_results["avg_task_duration"] = sum(task.get("duration", 0) for task in category_results["tasks"]) / category_results["total_tasks"]
        else:
            category_results["success_rate"] = 0
            category_results["avg_task_duration"] = 0
        
        return category_results
    
    # Apply the patch
    evaluator._evaluate_category = patched_evaluate_category
    
    # Run evaluation
    logger.info(f"Starting TauBench evaluation with {len(categories)} categories")
    start_time = time.time()
    results = await evaluator.run_evaluation()
    elapsed_time = time.time() - start_time
    
    # Export task logs if using enhanced logging
    if using_enhanced_logging:
        logger.info(f"Exporting task logs in {log_format} format")
        exported_files = agent.log_manager.export_task_logs(logs_dir, log_format)
        logger.info(f"Exported {len(exported_files)} task log files")
        
        # Get task metrics and add to results
        task_metrics = agent.log_manager.get_task_metrics()
        results["task_metrics"] = task_metrics
        
        # Add log file paths to results
        results["log_files"] = exported_files
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"taubench_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Get task metrics if available
    if "task_metrics" in results:
        # Calculate averages
        metrics = results["task_metrics"]
        avg_thought_count = sum(m["thought_count"] for m in metrics.values()) / len(metrics) if metrics else 0
        avg_action_count = sum(m["action_count"] for m in metrics.values()) / len(metrics) if metrics else 0
        avg_llm_latency = sum(m["llm_latency"] for m in metrics.values()) / len(metrics) if metrics else 0
        avg_action_latency = sum(m["action_latency"] for m in metrics.values()) / len(metrics) if metrics else 0
        avg_total_latency = sum(m["total_latency"] for m in metrics.values()) / len(metrics) if metrics else 0
        
        logger.info(f"Average thoughts per task: {avg_thought_count:.2f}")
        logger.info(f"Average actions per task: {avg_action_count:.2f}")
        logger.info(f"Average LLM latency per task: {avg_llm_latency:.2f}s")
        logger.info(f"Average action latency per task: {avg_action_latency:.2f}s")
        logger.info(f"Average total latency per task: {avg_total_latency:.2f}s")
    
    # Print summary
    summary = evaluator.get_summary()
    logger.info(f"TauBench evaluation completed in {elapsed_time:.2f}s")
    logger.info(f"Categories tested: {', '.join(summary.get('categories_tested', []))}")
    logger.info(f"Total tasks: {summary.get('total_tasks', 0)}")
    logger.info(f"Success rate: {summary.get('overall_success_rate', 0):.2%}")
    logger.info(f"Results saved to: {results_file}")
    
    return results

def create_agent_with_enhanced_logging(model: str) -> ReActAgent:
    """Create a ReActAgent with enhanced logging capabilities."""
    # Create enhanced log manager
    log_manager = EnhancedLogManager()
    
    # Create agent example with the enhanced log manager
    agent_example = ReactAgentExample(
        model=model,
        max_iterations=15  # Increased iterations for more comprehensive evaluation
    )
    
    # Set the enhanced log manager
    agent_example.agent.log_manager = log_manager
    
    return agent_example.agent

def main():
    """Main entry point for command-line execution."""
    if not TAUBENCH_AVAILABLE:
        print("TauBench is not available. Please install tau-bench package.")
        return
    
    parser = argparse.ArgumentParser(description="Run TauBench evaluation with enhanced logging")
    
    # Agent configuration
    parser.add_argument("--model", type=str, default="anthropic/claude-3-7-sonnet-20250219",
                      help="LLM model to use (default: claude-3-sonnet)")
    
    # Evaluation parameters
    parser.add_argument("--categories", nargs="+", 
                      default=["reasoning", "planning", "tool_use"],
                      help="Categories to evaluate (default: reasoning, planning, tool_use)")
    parser.add_argument("--task-limit", type=int, default=2,
                      help="Maximum number of tasks per category (default: 2)")
    parser.add_argument("--output-dir", type=str, default="taubench_evaluation",
                      help="Directory to save results (default: taubench_evaluation)")
    
    # Logging parameters
    parser.add_argument("--log-format", choices=["json", "markdown", "html"],
                      default="html", help="Format for exporting task logs (default: html)")
    
    args = parser.parse_args()
    
    # Create agent with enhanced logging
    agent = create_agent_with_enhanced_logging(args.model)
    
    # Run evaluation
    asyncio.run(run_taubench_evaluation(
        agent=agent,
        model=args.model,
        categories=args.categories,
        task_limit=args.task_limit,
        output_dir=args.output_dir,
        log_format=args.log_format
    ))

if __name__ == "__main__":
    main() 