"""
TauBench Evaluation

Implementation of agent evaluation using the TauBench framework.
Adapted for use with ReActAgent in the agent_suite.
"""

import asyncio
import time
import json
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

# Import our minimal TauBench implementation
try:
    # First try to import the external tau-bench library
    import taubench
    from taubench.datasets import load_dataset
    from taubench.evaluation import Evaluator
    from taubench.metrics import calculate_metrics
    TAUBENCH_AVAILABLE = True
except ImportError:
    # If not available, use our minimal implementation
    from .taubench_minimal import load_dataset, Evaluator, calculate_metrics
    TAUBENCH_AVAILABLE = True
    logging.info("Using minimal TauBench implementation")

# Base evaluation class
class BaseEvaluation:
    """Base class for evaluation methods."""
    
    def __init__(self, agent, name="evaluation"):
        """Initialize the evaluation.
        
        Args:
            agent: The agent to evaluate
            name: Name for this evaluation
        """
        self.agent = agent
        self.name = name
        self.results = {}
    
    async def run_evaluation(self):
        """Run the evaluation.
        
        Returns:
            Evaluation results
        """
        raise NotImplementedError("Subclasses must implement run_evaluation")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results.
        
        Returns:
            Summary of evaluation results
        """
        return self.results

from ...agents.react.agent import ReActAgent

# Define TauBench task categories
TASK_CATEGORIES = [
    "reasoning",          # Logical reasoning tasks
    "planning",           # Sequential planning tasks
    "tool_use",           # Tool manipulation and usage
    "code_generation",    # Code writing tasks
    "qa",                 # Question answering
    "math",               # Mathematical problem solving
    "science",            # Scientific reasoning
    "instruction",        # Following complex instructions
]

class TauBenchDataLoader:
    """
    Data loader for TauBench datasets.
    
    This class handles loading and preprocessing of TauBench task datasets.
    """
    
    def __init__(self, data_dir: Optional[str] = None, cache: bool = True):
        """
        Initialize the TauBench data loader.
        
        Args:
            data_dir: Directory containing TauBench datasets (None uses default)
            cache: Whether to cache loaded datasets in memory
        """
        if not TAUBENCH_AVAILABLE:
            raise ImportError("TauBench library is required to use TauBenchDataLoader")
        
        self.data_dir = data_dir
        self.cache = cache
        self.loaded_datasets = {}
    
    def load_dataset(self, category: str, version: str = "latest", split: str = "test") -> List[Dict[str, Any]]:
        """
        Load a TauBench dataset for a specific category.
        
        Args:
            category: Task category to load
            version: Dataset version to use
            split: Dataset split to load ('train', 'validation', or 'test')
            
        Returns:
            List of task instances
        """
        cache_key = f"{category}_{version}_{split}"
        
        # Return cached dataset if available
        if self.cache and cache_key in self.loaded_datasets:
            return self.loaded_datasets[cache_key]
        
        # Load dataset using TauBench's load_dataset function
        try:
            dataset = load_dataset(
                category, 
                version=version, 
                split=split,
                data_dir=self.data_dir
            )
            
            # Convert to standardized format for our evaluation
            standardized_tasks = self._standardize_dataset(dataset, category)
            
            # Cache if enabled
            if self.cache:
                self.loaded_datasets[cache_key] = standardized_tasks
                
            return standardized_tasks
        
        except Exception as e:
            logging.error(f"Error loading TauBench dataset {category}: {e}")
            # Return empty list when dataset cannot be loaded
            return []
    
    def _standardize_dataset(self, dataset: Any, category: str) -> List[Dict[str, Any]]:
        """
        Convert TauBench dataset to standardized format for evaluation.
        
        Args:
            dataset: The raw TauBench dataset
            category: The task category
            
        Returns:
            Standardized tasks list
        """
        standardized_tasks = []
        
        # Process according to TauBench format, which may vary by category
        for idx, item in enumerate(dataset):
            try:
                # Our minimal implementation already provides the correct format
                if isinstance(item, dict) and "id" in item and "instruction" in item:
                    # Just make sure category is set
                    if "category" not in item:
                        item["category"] = category
                    standardized_tasks.append(item)
                else:
                    # Extract task data based on category for other formats
                    task = {
                        "id": item.get("id", f"{category}_{idx}"),
                        "category": category,
                        "instruction": item.get("instruction", ""),
                        "input": item.get("input", ""),
                        "reference": item.get("reference", ""),
                        "metadata": item.get("metadata", {}),
                        "tools": item.get("tools", []),
                        "evaluation_criteria": item.get("evaluation_criteria", {})
                    }
                    
                    standardized_tasks.append(task)
            except Exception as e:
                logging.warning(f"Error processing task {idx} in {category}: {e}")
        
        return standardized_tasks
    
    def list_available_categories(self) -> List[str]:
        """
        List all available TauBench categories.
        
        Returns:
            List of available categories
        """
        if not TAUBENCH_AVAILABLE:
            return []
            
        # This would use the TauBench API to list available categories
        # For now, return the predefined list
        return TASK_CATEGORIES
    
    def sample_tasks(self, category: str, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Sample a limited number of tasks from a category.
        
        Args:
            category: Task category to sample from
            num_samples: Number of samples to take
            
        Returns:
            Sampled tasks
        """
        tasks = self.load_dataset(category)
        
        if not tasks:
            return []
            
        # Take up to num_samples, but don't fail if fewer are available
        return tasks[:min(num_samples, len(tasks))]


class TauBenchEvaluation(BaseEvaluation):
    """
    Evaluation for agents using the TauBench framework.
    
    TauBench is a specialized framework for evaluating language model agents
    across a diverse set of tasks focusing on reasoning, planning, and tool use.
    """
    
    def __init__(
        self, 
        agent: ReActAgent, 
        model: str,
        categories: List[str] = None,
        task_limit: int = 5,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        name: str = "taubench_eval",
        use_reference_model: bool = False,
        reference_model: Optional[str] = None,
        category_limit: int = 5
    ):
        """
        Initialize the TauBench evaluation.
        
        Args:
            agent: The ReActAgent to evaluate
            model: The model to use for evaluation
            categories: List of task categories to evaluate (defaults to a subset)
            task_limit: Maximum number of tasks per category
            data_dir: Directory containing TauBench datasets (None uses default)
            output_dir: Directory to save evaluation results (None uses default)
            name: Name for this evaluation run
            use_reference_model: Whether to run comparative evaluation with a reference model
            reference_model: Model to use as reference for comparative evaluation (if enabled)
            category_limit: Maximum number of categories to evaluate
        """
        super().__init__(agent, name)
        
        if not TAUBENCH_AVAILABLE:
            raise ImportError("TauBench library is required to use TauBenchEvaluation")
        
        self.model = model
        self.categories = categories or TASK_CATEGORIES
        self.task_limit = task_limit
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.use_reference_model = use_reference_model
        self.reference_model = reference_model
        self.category_limit = category_limit
        
        # Initialize evaluator
        self.evaluator = Evaluator()
        
        # Initialize results
        self.results = {
            "model": model,
            "agent_type": agent.__class__.__name__,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "categories": {},
            "summary": {},
            "comparative": {} if use_reference_model else None
        }
    
    async def run_evaluation(self):
        """
        Run the TauBench evaluation on the agent.
        
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        # Initialize results with metadata
        results = {
            "model": self.model,
            "agent_type": self.agent.__class__.__name__,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "categories": {},
            "summary": {}
        }
        
        # Get list of categories to evaluate
        categories_to_evaluate = self.categories[:self.task_limit] if self.categories else TASK_CATEGORIES[:self.task_limit]
        print(f"Categories to evaluate: {categories_to_evaluate}")
        
        # Create data loader
        data_loader = TauBenchDataLoader(self.data_dir)
        total_tasks = 0
        
        for category in categories_to_evaluate:
            print(f"Loading tasks for category: {category}")
            
            # Load dataset for this category
            tasks = data_loader.load_dataset(category)
            print(f"Loaded {len(tasks)} tasks for category {category}")
            
            # Limit number of tasks if specified
            if self.task_limit > 0 and len(tasks) > self.task_limit:
                tasks = tasks[:self.task_limit]
                print(f"Limited to {len(tasks)} tasks")
            
            # Skip if no tasks
            if not tasks:
                print(f"No tasks found for category: {category}")
                continue
            
            # Run evaluation for this category
            category_results = await self._evaluate_category(category, tasks)
            print(f"Completed evaluation for category: {category}")
            
            # Store results for this category
            results["categories"][category] = category_results
            total_tasks += len(tasks)
        
        # Calculate overall metrics and summary statistics
        if total_tasks > 0:
            print(f"Calculating summary metrics for {total_tasks} tasks")
            # Store the results before calculating metrics
            self.results = results
            summary = self._calculate_summary_metrics()
            results["summary"] = summary
        else:
            print("No tasks were evaluated")
            results["summary"] = {
                "error": "No tasks were evaluated"
            }
        
        # Store completion time
        elapsed_time = time.time() - start_time
        results["elapsed_time"] = elapsed_time
        
        self.results = results
        return results
    
    async def _evaluate_category(self, category: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the agent on a category of tasks.
        
        Args:
            category: The task category
            tasks: List of tasks to evaluate
            
        Returns:
            Dictionary with evaluation results for this category
        """
        print(f"Evaluating {len(tasks)} tasks in category: {category}")
        
        # Initialize results for this category
        category_results = {
            "total_tasks": len(tasks),
            "tasks": [],
            "successful_tasks": 0,
            "success_rate": 0.0
        }
        
        # Process each task
        for task in tasks:
            task_id = task.get("id", f"{category}_{len(category_results['tasks'])}")
            instruction = task.get("instruction", "")
            task_input = task.get("input", "")
            reference = task.get("reference", "")
            criteria = task.get("evaluation_criteria", {})
            
            # Combine instruction and input if both exist
            if task_input:
                prompt = f"{instruction}\n\n{task_input}"
            else:
                prompt = instruction
            
            task_result = {
                "id": task_id,
                "prompt": prompt,
                "reference": reference
            }
            
            try:
                # Run the agent on this task
                response = await self.agent.arun(prompt, self.model)
                task_result["response"] = response
                
                # Evaluate the response
                metrics = calculate_metrics(response, reference, criteria)
                task_result["metrics"] = metrics
                
                # Update success counter
                if metrics.get("success", False):
                    category_results["successful_tasks"] += 1
                    
            except Exception as e:
                task_result["error"] = str(e)
                task_result["metrics"] = {"success": False, "error": True}
            
            # Add to category results
            category_results["tasks"].append(task_result)
        
        # Calculate category success rate
        if category_results["total_tasks"] > 0:
            category_results["success_rate"] = category_results["successful_tasks"] / category_results["total_tasks"]
        
        return category_results
    
    async def _run_comparative_evaluation(self):
        """Run comparative evaluation against a reference model."""
        if not self.use_reference_model or not self.reference_model:
            return
            
        logging.info(f"Running comparative evaluation against {self.reference_model}...")
        
        # Create a new data loader for the reference evaluation
        data_loader = TauBenchDataLoader(data_dir=self.data_dir)
        
        # Initialize comparative results
        self.results["comparative"] = {
            "reference_model": self.reference_model,
            "categories": {},
            "summary": {}
        }
        
        # Process each category that was evaluated for the primary model
        for category, results in self.results["categories"].items():
            # Skip categories with errors
            if "error" in results:
                continue
                
            # Get the same tasks that were used for the primary model
            task_ids = [task["id"] for task in results.get("tasks", [])]
            
            if not task_ids:
                continue
                
            # Load all tasks for this category
            all_tasks = data_loader.load_dataset(category)
            
            # Filter to the same tasks that were evaluated for the primary model
            reference_tasks = [task for task in all_tasks if task["id"] in task_ids]
            
            if not reference_tasks:
                continue
                
            # Evaluate the reference model on these tasks
            # Implement reference model evaluation logic here
            
            # For now, just log that this would be done
            logging.info(f"Would evaluate reference model {self.reference_model} on {len(reference_tasks)} tasks for {category}")
            
            # Store placeholder results
            self.results["comparative"]["categories"][category] = {
                "total_tasks": len(reference_tasks),
                "tasks": []
            }
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary metrics across all evaluated categories.
        
        Returns:
            Dictionary with summary metrics
        """
        print("Calculating summary metrics...")
        
        if not self.results or not self.results.get("categories"):
            return {"error": "No results available"}
        
        categories = self.results.get("categories", {})
        
        # Initialize counters
        total_tasks = 0
        successful_tasks = 0
        categories_tested = []
        category_success_rates = {}
        
        # Process each category
        for category, results in categories.items():
            # Skip categories that had errors
            if "error" in results:
                continue
                
            categories_tested.append(category)
            
            # Get task-level metrics
            tasks = results.get("tasks", [])
            category_success = 0
            
            for task in tasks:
                metrics = task.get("metrics", {})
                if metrics.get("success"):
                    category_success += 1
            
            # Calculate category success rate
            if tasks:
                category_success_rate = category_success / len(tasks)
                category_success_rates[category] = category_success_rate
                
                # Update totals
                total_tasks += len(tasks)
                successful_tasks += category_success
        
        # Calculate overall metrics
        if total_tasks > 0:
            overall_success_rate = successful_tasks / total_tasks
        else:
            overall_success_rate = 0
        
        # Create summary
        summary = {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "overall_success_rate": overall_success_rate,
            "categories_tested": categories_tested,
            "category_success_rates": category_success_rates
        }
        
        return summary
    
    def _calculate_comparative_summary(self):
        """Calculate summary metrics for comparative evaluation."""
        comparative = self.results.get("comparative", {})
        
        if not comparative or "categories" not in comparative:
            return
            
        # Calculate overall performance comparison
        overall_diff = {}
        metric_counts = {}
        
        for category, results in comparative.get("categories", {}).items():
            for comparison in results.get("comparisons", []):
                for metric, diff in comparison.get("performance_diff", {}).items():
                    if metric not in overall_diff:
                        overall_diff[metric] = 0
                        metric_counts[metric] = 0
                    
                    overall_diff[metric] += diff
                    metric_counts[metric] += 1
        
        # Calculate average differences
        avg_diffs = {}
        for metric, total in overall_diff.items():
            count = metric_counts[metric]
            if count > 0:
                avg_diffs[metric] = total / count
        
        # Add to summary
        self.results["summary"]["comparative"] = {
            "reference_model": comparative.get("reference_model", ""),
            "avg_metric_differences": avg_diffs,
            "better_than_reference": {
                metric: diff > 0 for metric, diff in avg_diffs.items()
            }
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation results.
        
        Returns:
            Dictionary with summary information
        """
        if not self.results:
            return {"status": "No results available"}
            
        # If summary already exists, return it
        if self.results.get("summary") and isinstance(self.results["summary"], dict):
            return self.results["summary"]
            
        # Try to calculate summary
        try:
            summary = self._calculate_summary_metrics()
            return summary
        except Exception as e:
            # Return a minimal summary if there's an error
            categories_list = list(self.results.get("categories", {}).keys())
            
            return {
                "status": "Error calculating full summary",
                "error": str(e),
                "categories_tested": categories_list,
                "total_categories": len(categories_list)
            } 