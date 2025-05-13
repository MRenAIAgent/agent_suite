#!/usr/bin/env python
"""
LangChain ReAct Agent Evaluation Script

This script evaluates a LangChain ReAct agent using the TauBench evaluation framework.
"""

import asyncio
import json
import os
import logging
import argparse
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import TauBench evaluation
from agents.eval.taubench.taubench_eval import TauBenchEvaluation
from agents.eval.taubench.taubench_minimal import load_dataset

# Import LangChain components
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Try to import additional LangChain tools
try:
    from langchain_community.utilities import WikipediaAPIWrapper
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


class LangChainReActAgent:
    """LangChain ReAct Agent wrapper for TauBench evaluation."""
    
    def __init__(self, model: str = "gpt-4", verbose: bool = True):
        """Initialize the LangChain ReAct agent."""
        self.model = model
        self.verbose = verbose
        self.llm = ChatOpenAI(temperature=0, model=model)
        
        # Set up tools
        tools = self._setup_tools()
        logger.info(f"Initialized {len(tools)} tools for LangChain ReAct agent")
        
        # Set up memory
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Create ReAct agent
        self.agent = initialize_agent(
            tools, 
            self.llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=verbose,
            memory=memory
        )
        logger.info(f"LangChain ReAct agent initialized with model: {model}")
    
    def _setup_tools(self) -> List[Tool]:
        """Set up tools for the agent."""
        tools = []
        
        # Add Wikipedia tool if available
        if WIKIPEDIA_AVAILABLE:
            try:
                wiki = WikipediaAPIWrapper()
                tools.append(
                    Tool(
                        name="Wikipedia",
                        func=wiki.run,
                        description="Search for information on Wikipedia."
                    )
                )
                logger.info("Added Wikipedia tool")
            except Exception as e:
                logger.warning(f"Failed to initialize Wikipedia tool: {e}")
        
        # Add Search tool if available
        if SEARCH_AVAILABLE:
            try:
                search = DuckDuckGoSearchRun()
                tools.append(
                    Tool(
                        name="Search",
                        func=search.run,
                        description="Search the internet for information."
                    )
                )
                logger.info("Added DuckDuckGo search tool")
            except Exception as e:
                logger.warning(f"Failed to initialize DuckDuckGo search tool: {e}")
        
        # Always add Calculator tool
        tools.append(
            Tool(
                name="Calculator",
                func=lambda x: eval(x),
                description="Useful for performing calculations."
            )
        )
        logger.info("Added Calculator tool")
        
        # Add mock Search tool if real one is not available
        if not SEARCH_AVAILABLE and not WIKIPEDIA_AVAILABLE:
            tools.append(
                Tool(
                    name="MockSearch",
                    func=lambda x: f"Searched for: {x}\nThis is a mock search result.",
                    description="Search the web for information."
                )
            )
            logger.info("Added mock search tool")
        
        return tools
    
    async def arun(self, prompt: str, model_override: Optional[str] = None) -> str:
        """Run the agent on a prompt."""
        try:
            # If model_override is provided, create a new instance with that model
            if model_override and model_override != self.model:
                logger.info(f"Using model override: {model_override}")
                temp_agent = LangChainReActAgent(model=model_override, verbose=self.verbose)
                return await temp_agent.arun(prompt)
            
            # Run the agent asynchronously
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.agent.run(prompt))
            return result
        except Exception as e:
            logger.error(f"Error running LangChain agent: {e}")
            return f"Error: {str(e)}"


async def evaluate_agent(agent, categories: List[str], task_limit: int, output_dir: str):
    """Evaluate the agent using TauBench."""
    # Create evaluator
    evaluator = TauBenchEvaluation(
        agent=agent,
        model=agent.model,
        categories=categories,
        task_limit=task_limit,
        name="langchain_react_eval"
    )
    
    # Run evaluation
    start_time = datetime.now()
    results = await evaluator.run_evaluation()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"langchain_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Get summary
    summary = evaluator.get_summary()
    
    # Print summary
    logger.info(f"Evaluation completed in {elapsed_time:.2f}s")
    logger.info(f"Categories tested: {', '.join(summary.get('categories_tested', []))}")
    logger.info(f"Total tasks: {summary.get('total_tasks', 0)}")
    logger.info(f"Success rate: {summary.get('overall_success_rate', 0):.2%}")
    
    # Generate category breakdown
    logger.info("\n--- Category Breakdown ---")
    category_rates = summary.get("category_success_rates", {})
    for category, rate in category_rates.items():
        logger.info(f"{category.capitalize()}: {rate:.2%}")
    
    return results, summary, results_file


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate LangChain ReAct agent with TauBench")
    
    parser.add_argument("--model", type=str, default="gpt-4",
                      help="LLM model to use (default: gpt-4)")
    parser.add_argument("--categories", nargs="+", 
                      default=["reasoning", "planning", "tool_use"],
                      help="Categories to evaluate")
    parser.add_argument("--task-limit", type=int, default=3,
                      help="Limit number of tasks per category")
    parser.add_argument("--output-dir", type=str, default="langchain_eval_results",
                      help="Directory to save results")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output from the LangChain agent")
    
    args = parser.parse_args()
    
    try:
        # Initialize LangChain ReAct agent
        logger.info(f"Initializing LangChain ReAct agent with model {args.model}...")
        langchain_agent = LangChainReActAgent(model=args.model, verbose=args.verbose)
        
        # Evaluate agent
        logger.info(f"Starting evaluation on categories: {', '.join(args.categories)}")
        results, summary, results_file = await evaluate_agent(
            agent=langchain_agent,
            categories=args.categories,
            task_limit=args.task_limit,
            output_dir=args.output_dir
        )
        
        # Print final summary
        logger.info("\n===== EVALUATION SUMMARY =====")
        logger.info(f"Model: {args.model}")
        logger.info(f"Overall success rate: {summary.get('overall_success_rate', 0):.2%}")
        logger.info(f"Successful tasks: {summary.get('successful_tasks', 0)}/{summary.get('total_tasks', 0)}")
        logger.info(f"Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return 1


if __name__ == "__main__":
    exitcode = asyncio.run(main())
    sys.exit(exitcode) 