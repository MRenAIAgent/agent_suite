#!/usr/bin/env python
"""
Run TauBench evaluation with LangChain agents.
"""

import asyncio
import argparse
import os
import json
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# Import TauBench evaluation
from agents.eval.taubench.taubench_eval import TauBenchEvaluation
from agents.eval.taubench.taubench_minimal import load_dataset, Evaluator, calculate_metrics

# Import tool setup
from tools.weather import WeatherTool
from tools.calculator import CalculatorTool
from tools.translation import TranslationTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LangChainAgentWrapper:
    """
    Wrapper for LangChain agents to use with TauBench.
    """
    
    def __init__(self, agent_executor, agent_type_name):
        self.agent_executor = agent_executor
        self.agent_type_name = agent_type_name
        
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
    
    # Initialize tools
    tools = [
        WeatherTool().as_langchain_tool(),
        CalculatorTool().as_langchain_tool(),
        TranslationTool().as_langchain_tool()
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
    
    # Initialize tools
    tools = [
        WeatherTool().as_langchain_tool(),
        CalculatorTool().as_langchain_tool(),
        TranslationTool().as_langchain_tool()
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

async def run_evaluation(
    agent_type: str,
    model_name: str,
    output_dir: str,
    task_limit: int = 5,
    category_limit: int = 2
):
    """
    Run TauBench evaluation on a LangChain agent.
    
    Args:
        agent_type: Type of agent to evaluate ('react' or 'structured')
        model_name: Name of the model to use
        output_dir: Directory to save evaluation results
        task_limit: Maximum number of tasks per category
        category_limit: Maximum number of categories to evaluate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create agent based on agent_type
    if agent_type == "react":
        agent = create_langchain_react_agent(model_name)
        agent_name = "LangChain ReAct"
    elif agent_type == "structured":
        agent = create_langchain_structured_agent(model_name)
        agent_name = "LangChain Structured"
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    logger.info(f"Running TauBench evaluation for {agent_name} with {model_name}")
    
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
    logger.info(f"TauBench Evaluation completed in {elapsed_time:.2f}s")
    logger.info(f"Total tasks: {summary.get('total_tasks', 0)}")
    logger.info(f"Success rate: {summary.get('overall_success_rate', 0):.2%}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"taubench_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
    summary_data = {
        "agent_type": agent_name,
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
    logger.info(f"Summary saved to {summary_file}")

async def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Run TauBench evaluation with LangChain agents")
    parser.add_argument("--agent-type", type=str, choices=["react", "structured"], default="react",
                       help="Type of LangChain agent to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="Name of the model to use")
    parser.add_argument("--output-dir", type=str, default="taubench_results_langchain",
                       help="Directory to save evaluation results")
    parser.add_argument("--task-limit", type=int, default=5,
                       help="Maximum number of tasks per category")
    parser.add_argument("--category-limit", type=int, default=2,
                       help="Maximum number of categories to evaluate")
    
    args = parser.parse_args()
    
    await run_evaluation(
        agent_type=args.agent_type,
        model_name=args.model,
        output_dir=args.output_dir,
        task_limit=args.task_limit,
        category_limit=args.category_limit
    )

if __name__ == "__main__":
    asyncio.run(main()) 