#!/usr/bin/env python
"""
Agent Evaluation Runner

This script runs evaluations on your agent using AgentBench, AgentBoard, and TauBench frameworks.
It provides detailed metrics on agent performance across different domains and tasks.
"""

import argparse
import asyncio
import os
import json
from datetime import datetime
import time
import logging
from typing import List, Dict, Any

# Import agent components
from agents.react_agent import ReActAgent
from agents.examples.react_agent_example import ReactAgentExample
from log.logging import LogManager
from log.export import export_logs

# Import evaluation frameworks
from agents.eval.agentbench.agentbench_eval import AgentBenchEvaluation
from agents.eval.agentboard.agentboard_eval import AgentBoardEvaluation

# Import TauBench evaluation if available
try:
    from agents.eval.taubench.taubench_eval import TauBenchEvaluation
    TAUBENCH_AVAILABLE = True
except ImportError:
    TAUBENCH_AVAILABLE = False
    logging.warning("TauBench library not found. TauBench evaluation will be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_evaluation(
    agent: ReActAgent,
    model: str,
    frameworks: List[str],
    output_dir: str,
    env_limit: int,
    task_limit: int,
    turn_limit: int,
    export_format: str
) -> Dict[str, Any]:
    """
    Run evaluations on the agent using the specified frameworks.
    
    Args:
        agent: The agent to evaluate
        model: The model used by the agent
        frameworks: Which frameworks to use for evaluation (agentbench, agentboard, taubench)
        output_dir: Directory to save evaluation results
        env_limit: Maximum number of environments to test per framework
        task_limit: Maximum number of tasks to test per environment
        turn_limit: Maximum number of turns for multi-turn evaluations
        export_format: Format to export detailed results (json, markdown, html)
        
    Returns:
        Dict with evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "agent_type": agent.__class__.__name__,
        "model": model,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frameworks": {}
    }
    
    # Create log directory for detailed logs
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Run AgentBench evaluation
    if "agentbench" in frameworks:
        logger.info(f"Starting AgentBench evaluation on {model}...")
        
        # Select environments to test
        environments = ["os", "db", "kg"][:env_limit]
        
        # Create evaluator
        agentbench_eval = AgentBenchEvaluation(
            agent=agent,
            model=model,
            environments=environments,
            task_limit=task_limit,
            name="agentbench_evaluation"
        )
        
        # Run evaluation
        start_time = time.time()
        try:
            agentbench_results = await agentbench_eval.run_evaluation()
            elapsed_time = time.time() - start_time
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f"agentbench_results_{timestamp}.json")
            with open(results_file, "w") as f:
                json.dump(agentbench_results, f, indent=2)
            
            # Get summary
            summary = agentbench_eval.get_summary()
            
            # Print summary
            logger.info(f"AgentBench Evaluation completed in {elapsed_time:.2f}s")
            logger.info(f"Environments tested: {', '.join(summary.get('environments_tested', []))}")
            logger.info(f"Total tasks: {summary.get('total_tasks', 0)}")
            logger.info(f"Success rate: {summary.get('overall_success_rate', 0):.2%}")
            
            # Add to results
            results["frameworks"]["agentbench"] = {
                "summary": summary,
                "elapsed_time": elapsed_time,
                "results_file": results_file
            }
            
            # Export logs if available
            if hasattr(agent, "log_manager") and hasattr(agent.log_manager, "get_current_execution"):
                log_file = os.path.join(log_dir, f"agentbench_logs_{timestamp}.{export_format}")
                export_logs(agent.log_manager.get_current_execution(), export_format, log_file)
                logger.info(f"Detailed logs exported to {log_file}")
                
        except Exception as e:
            logger.error(f"Error in AgentBench evaluation: {e}")
            results["frameworks"]["agentbench"] = {
                "error": str(e),
                "status": "failed"
            }
    
    # Run AgentBoard evaluation
    if "agentboard" in frameworks:
        logger.info(f"Starting AgentBoard evaluation on {model}...")
        
        # Select domains to test
        domains = ["reasoning", "knowledge", "planning", "navigation", "coding"][:env_limit]
        
        # Create evaluator
        agentboard_eval = AgentBoardEvaluation(
            agent=agent,
            model=model,
            domains=domains,
            task_limit=task_limit,
            turn_limit=turn_limit,
            name="agentboard_evaluation"
        )
        
        # Run evaluation
        start_time = time.time()
        try:
            agentboard_results = await agentboard_eval.run_evaluation()
            elapsed_time = time.time() - start_time
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f"agentboard_results_{timestamp}.json")
            with open(results_file, "w") as f:
                json.dump(agentboard_results, f, indent=2)
            
            # Get summary
            summary = agentboard_eval.get_summary()
            
            # Print summary
            logger.info(f"AgentBoard Evaluation completed in {elapsed_time:.2f}s")
            logger.info(f"Domains tested: {', '.join(summary.get('domains_tested', []))}")
            logger.info(f"Total tasks: {summary.get('total_tasks', 0)}")
            logger.info(f"Success rate: {summary.get('overall_success_rate', 0):.2%}")
            logger.info(f"Average progress rate: {summary.get('avg_progress_rate', 0):.2%}")
            
            # Add to results
            results["frameworks"]["agentboard"] = {
                "summary": summary,
                "elapsed_time": elapsed_time,
                "results_file": results_file
            }
            
            # Export logs if available
            if hasattr(agent, "log_manager") and hasattr(agent.log_manager, "get_current_execution"):
                log_file = os.path.join(log_dir, f"agentboard_logs_{timestamp}.{export_format}")
                export_logs(agent.log_manager.get_current_execution(), export_format, log_file)
                logger.info(f"Detailed logs exported to {log_file}")
                
        except Exception as e:
            logger.error(f"Error in AgentBoard evaluation: {e}")
            results["frameworks"]["agentboard"] = {
                "error": str(e),
                "status": "failed"
            }
    
    # Run TauBench evaluation if available
    if "taubench" in frameworks and TAUBENCH_AVAILABLE:
        logger.info(f"Starting TauBench evaluation on {model}...")
        
        # Select categories to test (up to env_limit)
        categories = ["reasoning", "planning", "tool_use"][:env_limit]
        
        # Create evaluator
        try:
            logger.info("Creating TauBenchEvaluation instance...")
            taubench_eval = TauBenchEvaluation(
                agent=agent,
                model=model,
                categories=categories,
                task_limit=task_limit,
                category_limit=env_limit,
                name="taubench_evaluation"
            )
            
            # Run evaluation
            logger.info("Starting TauBench evaluation...")
            start_time = time.time()
            try:
                logger.info("Calling taubench_eval.run_evaluation()...")
                taubench_results = await taubench_eval.run_evaluation()
                elapsed_time = time.time() - start_time
                
                # Save detailed results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = os.path.join(output_dir, f"taubench_results_{timestamp}.json")
                logger.info(f"Saving results to {results_file}...")
                with open(results_file, "w") as f:
                    json.dump(taubench_results, f, indent=2)
                
                # Get summary
                logger.info("Getting summary...")
                try:
                    summary = taubench_eval.get_summary()
                    logger.info(f"Got summary: {summary}")
                except Exception as e:
                    logger.error(f"Error getting summary: {e}", exc_info=True)
                    summary = {"error": str(e)}
                
                # Print summary
                logger.info(f"TauBench Evaluation completed in {elapsed_time:.2f}s")
                logger.info(f"Categories tested: {', '.join(summary.get('categories_tested', []))}")
                logger.info(f"Total tasks: {summary.get('total_tasks', 0)}")
                logger.info(f"Success rate: {summary.get('overall_success_rate', 0):.2%}")
                
                # Add to results
                results["frameworks"]["taubench"] = {
                    "summary": summary,
                    "elapsed_time": elapsed_time,
                    "results_file": results_file
                }
                
                # Export logs if available
                if hasattr(agent, "log_manager") and hasattr(agent.log_manager, "get_current_execution"):
                    try:
                        log_file = os.path.join(log_dir, f"taubench_logs_{timestamp}.{export_format}")
                        export_logs(agent.log_manager.get_current_execution(), export_format, log_file)
                        logger.info(f"Detailed logs exported to {log_file}")
                    except Exception as e:
                        logger.error(f"Error exporting logs: {e}", exc_info=True)
                        # Continue execution even if log export fails
                    
            except Exception as e:
                logger.error(f"Error in TauBench evaluation: {e}", exc_info=True)
                results["frameworks"]["taubench"] = {
                    "error": str(e),
                    "status": "failed"
                }
        except Exception as e:
            logger.error(f"Error initializing TauBench evaluation: {e}", exc_info=True)
            results["frameworks"]["taubench"] = {
                "error": str(e),
                "status": "initialization_failed"
            }
    
    # Save overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation complete. Summary saved to {summary_file}")
    return results

def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(description="Run evaluations on agent using multiple frameworks")
    
    # Framework selection
    parser.add_argument("--frameworks", nargs="+", choices=["agentbench", "agentboard", "taubench", "all"],
                      default=["all"], help="Evaluation frameworks to use")
    
    # Agent configuration
    parser.add_argument("--model", type=str, default="anthropic/claude-3-7-sonnet-20250219",
                      help="LLM model to use (default: claude-3-7-sonnet)")
    
    # Evaluation parameters
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                      help="Directory to save results")
    parser.add_argument("--env-limit", type=int, default=2,
                      help="Limit number of environments/domains per framework")
    parser.add_argument("--task-limit", type=int, default=2,
                      help="Limit number of tasks per environment")
    parser.add_argument("--turn-limit", type=int, default=5,
                      help="Limit number of turns per task for multi-turn evaluations")
    
    # Export format
    parser.add_argument("--export-format", choices=["json", "markdown", "html"],
                      default="html", help="Format for exporting detailed logs")
    
    args = parser.parse_args()
    
    # Expand 'all' to include all frameworks
    frameworks = []
    if "all" in args.frameworks:
        frameworks = ["agentbench", "agentboard"]
        if TAUBENCH_AVAILABLE:
            frameworks.append("taubench")
    else:
        frameworks = args.frameworks
    
    # Create agent for evaluation
    logger.info(f"Creating agent with model {args.model}")
    agent_example = ReactAgentExample(model=args.model, max_iterations=10)
    agent = agent_example.agent
    
    # Run evaluation
    asyncio.run(run_evaluation(
        agent=agent,
        model=args.model,
        frameworks=frameworks,
        output_dir=args.output_dir,
        env_limit=args.env_limit,
        task_limit=args.task_limit,
        turn_limit=args.turn_limit,
        export_format=args.export_format
    ))

if __name__ == "__main__":
    main() 