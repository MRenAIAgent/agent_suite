"""
AgentBench Evaluation

Implementation of agent evaluation using the AgentBench framework.
Adapted for use with ReactAgent in the agent_suite.
"""

import asyncio
import time
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from agents.eval import BaseEvaluation
from agents.react_agent import ReActAgent

# Define AgentBench environments to test
ENVIRONMENTS = [
    "os",        # Operating System
    "db",        # Database
    "kg",        # Knowledge Graph
    "dcg",       # Digital Card Game 
    "ltp"        # Lateral Thinking Puzzles
]

class AgentBenchEvaluation(BaseEvaluation):
    """
    Evaluation for agents using the AgentBench framework.
    
    AgentBench tests agents across diverse environments including OS interactions,
    database operations, knowledge graph queries, game playing, and puzzle solving.
    """
    
    def __init__(
        self, 
        agent: ReActAgent, 
        model: str,
        environments: List[str] = None,
        task_limit: int = 5,
        name: str = "agentbench_eval"
    ):
        """
        Initialize the AgentBench evaluation.
        
        Args:
            agent: The ReActAgent to evaluate
            model: The model to use for evaluation
            environments: List of environments to test (defaults to all)
            task_limit: Maximum number of tasks per environment
            name: Name of the evaluation
        """
        super().__init__(agent, name)
        self.model = model
        self.environments = environments or ENVIRONMENTS
        self.task_limit = task_limit
        self.results = {
            "agent_type": agent.__class__.__name__,
            "model": model,
            "environments": {},
            "overall_metrics": {}
        }
        
    async def run_evaluation(self):
        """Run the evaluation on all selected environments."""
        self.start_time = time.time()
        
        # Load tasks for each environment
        for env in self.environments:
            try:
                tasks = self._load_environment_tasks(env)
                env_results = await self._evaluate_environment(env, tasks)
                self.results["environments"][env] = env_results
            except Exception as e:
                self.results["environments"][env] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        # Calculate overall metrics
        self._calculate_overall_metrics()
        
        self.end_time = time.time()
        return self.results
    
    def _load_environment_tasks(self, environment: str) -> List[Dict[str, Any]]:
        """
        Load tasks for a specific environment.
        
        This function simulates loading tasks from AgentBench.
        In a real implementation, this would load from datasets.
        
        Args:
            environment: The environment to load tasks for
            
        Returns:
            List of tasks for the environment
        """
        # Simulated tasks for each environment
        # In a real implementation, these would be loaded from actual AgentBench datasets
        
        simulated_tasks = {
            "os": [
                {"id": "os_1", "instruction": "List all files in the current directory and count how many are Python files."},
                {"id": "os_2", "instruction": "Create a new directory called 'test_dir' and add a text file with the current date."},
                {"id": "os_3", "instruction": "Find all processes using more than 100MB of memory."},
                {"id": "os_4", "instruction": "Check if port 8080 is in use, and if not, suggest a command to start a web server there."},
                {"id": "os_5", "instruction": "Search through log files for errors and summarize them."}
            ],
            "db": [
                {"id": "db_1", "instruction": "Write a SQL query to find the top 5 customers by total purchase amount."},
                {"id": "db_2", "instruction": "Create a table to store user profiles with appropriate fields and constraints."},
                {"id": "db_3", "instruction": "Optimize a query that's running slowly by adding appropriate indexes."},
                {"id": "db_4", "instruction": "Join data from products and orders tables to find the most popular products."},
                {"id": "db_5", "instruction": "Write a transaction that safely transfers funds between two accounts."}
            ],
            "kg": [
                {"id": "kg_1", "instruction": "Find all entities connected to 'Artificial Intelligence' in the knowledge graph."},
                {"id": "kg_2", "instruction": "Add a new relationship between 'Machine Learning' and 'Neural Networks'."},
                {"id": "kg_3", "instruction": "Query for all researchers who have published papers on both 'NLP' and 'Computer Vision'."},
                {"id": "kg_4", "instruction": "Create a subgraph of all technologies developed after 2020."},
                {"id": "kg_5", "instruction": "Find the shortest path between 'Transformers' and 'Image Generation'."}
            ],
            "dcg": [
                {"id": "dcg_1", "instruction": "Evaluate this hand of cards and decide the best card to play next."},
                {"id": "dcg_2", "instruction": "Create a strategy to defeat an opponent who focuses on defensive cards."},
                {"id": "dcg_3", "instruction": "Choose which cards to discard to optimize your hand for the next round."},
                {"id": "dcg_4", "instruction": "Calculate the probability of drawing a specific card combo within the next 3 turns."},
                {"id": "dcg_5", "instruction": "Analyze the opponent's play pattern and suggest a counter-strategy."}
            ],
            "ltp": [
                {"id": "ltp_1", "instruction": "Solve this lateral thinking puzzle: A man walks into a bar and asks for a glass of water. The bartender pulls out a gun and points it at him. The man says 'Thank you' and walks out. Why?"},
                {"id": "ltp_2", "instruction": "A woman lives on the 10th floor. Every morning she takes the elevator down to the ground floor. In the evening, she takes the elevator to the 7th floor and walks up the stairs to reach her apartment on the 10th floor. Why?"},
                {"id": "ltp_3", "instruction": "A man is found dead in a field. Next to him is an unopened package. There is no other evidence as to how he died. How did he die?"},
                {"id": "ltp_4", "instruction": "Two people enter a room. Later, three people come out. How is this possible?"},
                {"id": "ltp_5", "instruction": "A man pushes his car in front of a hotel and tells the owner he's bankrupt. Why?"}
            ]
        }
        
        tasks = simulated_tasks.get(environment, [])
        return tasks[:self.task_limit]  # Limit number of tasks as specified
    
    async def _evaluate_environment(self, environment: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the agent on a specific environment.
        
        Args:
            environment: The environment being evaluated
            tasks: List of tasks for the environment
            
        Returns:
            Evaluation results for the environment
        """
        env_results = {
            "name": environment,
            "total_tasks": len(tasks),
            "successful_tasks": 0,
            "tasks": []
        }
        
        for task in tasks:
            task_id = task.get("id", f"task_{len(env_results['tasks'])}")
            instruction = task.get("instruction", "")
            
            task_result = {
                "id": task_id,
                "instruction": instruction,
                "start_time": time.time()
            }
            
            try:
                # Run the agent on the task
                response = await self.agent.arun(instruction, self.model)
                
                # In a real implementation, we would validate the agent's response
                # against ground truth or expected outputs from AgentBench
                # Here we're just recording the response
                
                task_result["success"] = True  # Placeholder, real evaluation would determine true/false
                task_result["response"] = response
                task_result["iterations"] = getattr(self.agent, "iterations", None)
                env_results["successful_tasks"] += 1
                
            except Exception as e:
                task_result["success"] = False
                task_result["error"] = str(e)
            
            task_result["end_time"] = time.time()
            task_result["duration"] = task_result["end_time"] - task_result["start_time"]
            
            env_results["tasks"].append(task_result)
        
        # Calculate environment-specific metrics
        env_results["success_rate"] = env_results["successful_tasks"] / env_results["total_tasks"] if env_results["total_tasks"] > 0 else 0
        env_results["avg_task_duration"] = sum(task["duration"] for task in env_results["tasks"]) / len(env_results["tasks"]) if env_results["tasks"] else 0
        
        return env_results
    
    def _calculate_overall_metrics(self):
        """Calculate overall metrics across all environments."""
        environments = self.results.get("environments", {})
        if not environments:
            self.results["overall_metrics"] = {
                "status": "No environments evaluated"
            }
            return
        
        total_tasks = sum(env.get("total_tasks", 0) for env in environments.values())
        successful_tasks = sum(env.get("successful_tasks", 0) for env in environments.values())
        
        self.results["overall_metrics"] = {
            "total_environments": len(environments),
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "overall_success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "avg_environment_success_rate": sum(env.get("success_rate", 0) for env in environments.values()) / len(environments) if environments else 0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        if not self.results.get("environments"):
            return {"status": "No results available"}
        
        summary = {
            "name": self.name,
            "agent_type": self.agent.__class__.__name__,
            "model": self.model,
            "environments_tested": list(self.results.get("environments", {}).keys()),
            "total_tasks": self.results.get("overall_metrics", {}).get("total_tasks", 0),
            "overall_success_rate": self.results.get("overall_metrics", {}).get("overall_success_rate", 0),
            "environment_performance": {}
        }
        
        for env_name, env_data in self.results.get("environments", {}).items():
            summary["environment_performance"][env_name] = {
                "success_rate": env_data.get("success_rate", 0),
                "avg_task_duration": env_data.get("avg_task_duration", 0)
            }
        
        return summary 