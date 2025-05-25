#!/usr/bin/env python
"""
Agent Comparison Script

This script compares a LangChain ReAct agent with a custom React agent
using the TauBench evaluation framework, and displays comparative metrics.
"""

import asyncio
import json
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom React agent
from agents.react_agent import ReActAgent
from agents.examples.react_agent_example import ReactAgentExample

# Import TauBench evaluation
from benchmark.agent.code.taubench.taubench_eval import TauBenchEvaluation, TauBenchDataLoader
from benchmark.agent.code.taubench.taubench_minimal import load_dataset

# Import LangChain components
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

# Try to import additional LangChain tools if available
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.tools.human.tool import HumanInputRun
    ADDITIONAL_TOOLS_AVAILABLE = True
except ImportError:
    ADDITIONAL_TOOLS_AVAILABLE = False

# LangChain ReAct Agent wrapper to match our interface
class LangChainReActWrapper:
    """Wrapper for LangChain agent to match our interface."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize the LangChain agent."""
        self.model = model
        self.llm = ChatOpenAI(temperature=0, model=model)
        
        # Set up tools
        tools = []
        
        # Add Wikipedia search
        try:
            wikipedia = WikipediaAPIWrapper()
            tools.append(
                Tool(
                    name="Wikipedia",
                    func=wikipedia.run,
                    description="A tool for searching information on Wikipedia."
                )
            )
        except Exception as e:
            logger.warning(f"Could not initialize Wikipedia tool: {e}")
            # Add a mock Wikipedia tool as fallback
            tools.append(
                Tool(
                    name="Wikipedia",
                    func=lambda x: f"Searched Wikipedia for: {x}\nThis is a mock result.",
                    description="A tool for searching information on Wikipedia."
                )
            )
        
        # Add search tool (either real or mock)
        if ADDITIONAL_TOOLS_AVAILABLE:
            try:
                search = DuckDuckGoSearchRun()
                tools.append(
                    Tool(
                        name="Search",
                        func=search.run,
                        description="Search the internet for current information."
                    )
                )
            except Exception as e:
                logger.warning(f"Could not initialize DuckDuckGo search: {e}")
                self._add_mock_search_tool(tools)
        else:
            self._add_mock_search_tool(tools)
        
        # Set up memory
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Create ReAct agent
        try:
            self.agent = initialize_agent(
                tools, 
                self.llm, 
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                verbose=True,
                memory=memory
            )
            logger.info(f"LangChain ReAct agent initialized with {len(tools)} tools")
        except Exception as e:
            logger.error(f"Error initializing LangChain agent: {e}")
            raise
    
    def _add_mock_search_tool(self, tools_list):
        """Add a mock search tool to the tools list."""
        tools_list.append(
            Tool(
                name="Search",
                func=lambda x: f"Searched for: {x}\nThis is a mock search result for demonstration purposes.",
                description="Search the web for information."
            )
        )
        
    async def arun(self, prompt: str, model_override: Optional[str] = None) -> str:
        """Run the agent on a prompt."""
        # Use model_override if provided
        if model_override and model_override != self.model:
            # Create new instance with the requested model
            logger.info(f"Using model override: {model_override}")
            wrapper = LangChainReActWrapper(model=model_override)
            return await wrapper.arun(prompt)
        
        # Run the agent
        try:
            # Convert to async
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.agent.run(prompt))
            return result
        except Exception as e:
            logger.error(f"Error running LangChain agent: {e}")
            return f"Error: {str(e)}"

async def evaluate_agent(agent, name: str, model: str, output_dir: str, categories: List[str], task_limit: int = 3):
    """Evaluate an agent using TauBench."""
    logger.info(f"Evaluating {name} with model {model}...")
    
    eval_dir = os.path.join(output_dir, name.lower().replace(" ", "_"))
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = TauBenchEvaluation(
        agent=agent,
        model=model,
        categories=categories,
        task_limit=task_limit,
        name=f"{name}_eval"
    )
    
    # Run evaluation
    start_time = datetime.now()
    results = await evaluator.run_evaluation()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(eval_dir, f"{name}_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Get summary
    summary = evaluator.get_summary()
    
    # Print summary
    logger.info(f"{name} Evaluation completed in {elapsed_time:.2f}s")
    logger.info(f"Categories tested: {', '.join(summary.get('categories_tested', []))}")
    logger.info(f"Total tasks: {summary.get('total_tasks', 0)}")
    logger.info(f"Success rate: {summary.get('overall_success_rate', 0):.2%}")
    
    return results, summary, results_file

def create_comparison_visualizations(custom_results: Dict, langchain_results: Dict, output_dir: str):
    """Create visualizations comparing the two agents."""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract metrics
    custom_summary = custom_results.get("summary", {})
    langchain_summary = langchain_results.get("summary", {})
    
    # Create overall success rate comparison
    plt.figure(figsize=(10, 6))
    agents = ["Custom React", "LangChain React"]
    success_rates = [
        custom_summary.get("overall_success_rate", 0) * 100,
        langchain_summary.get("overall_success_rate", 0) * 100
    ]
    
    plt.bar(agents, success_rates, color=['#3498db', '#e74c3c'])
    plt.title('Overall Success Rate Comparison')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    
    # Add percentage labels on bars
    for i, v in enumerate(success_rates):
        plt.text(i, v + 3, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "overall_success_rate.png"))
    
    # Create category-level comparison
    custom_cat_rates = custom_summary.get("category_success_rates", {})
    langchain_cat_rates = langchain_summary.get("category_success_rates", {})
    
    # Find common categories
    common_categories = set(custom_cat_rates.keys()) & set(langchain_cat_rates.keys())
    
    if common_categories:
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        categories = list(common_categories)
        custom_rates = [custom_cat_rates.get(cat, 0) * 100 for cat in categories]
        langchain_rates = [langchain_cat_rates.get(cat, 0) * 100 for cat in categories]
        
        # Set up bar positions
        x = range(len(categories))
        width = 0.35
        
        # Create grouped bar chart
        plt.bar([i - width/2 for i in x], custom_rates, width, label='Custom React', color='#3498db')
        plt.bar([i + width/2 for i in x], langchain_rates, width, label='LangChain React', color='#e74c3c')
        
        plt.title('Success Rate by Category')
        plt.ylabel('Success Rate (%)')
        plt.xticks(x, categories)
        plt.ylim(0, 100)
        plt.legend()
        
        # Add percentage labels on bars
        for i, v in enumerate(custom_rates):
            plt.text(i - width/2, v + 3, f"{v:.1f}%", ha='center', va='bottom')
            
        for i, v in enumerate(langchain_rates):
            plt.text(i + width/2, v + 3, f"{v:.1f}%", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "category_success_rates.png"))
    
    # Create detailed comparison table
    comparison_data = {
        "Metric": ["Overall Success Rate", "Total Tasks", "Successful Tasks"],
        "Custom React": [
            f"{custom_summary.get('overall_success_rate', 0):.2%}",
            custom_summary.get("total_tasks", 0),
            custom_summary.get("successful_tasks", 0)
        ],
        "LangChain React": [
            f"{langchain_summary.get('overall_success_rate', 0):.2%}",
            langchain_summary.get("total_tasks", 0),
            langchain_summary.get("successful_tasks", 0)
        ]
    }
    
    # Add category metrics
    for category in common_categories:
        comparison_data["Metric"].append(f"{category.capitalize()} Success Rate")
        comparison_data["Custom React"].append(f"{custom_cat_rates.get(category, 0):.2%}")
        comparison_data["LangChain React"].append(f"{langchain_cat_rates.get(category, 0):.2%}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(comparison_data)
    csv_file = os.path.join(vis_dir, "agent_comparison.csv")
    df.to_csv(csv_file, index=False)
    
    # Also save as HTML for easy viewing
    html_file = os.path.join(vis_dir, "agent_comparison.html")
    df.to_html(html_file, index=False)
    
    return vis_dir

def generate_html_report(
    custom_results: Dict, 
    langchain_results: Dict, 
    custom_summary: Dict, 
    langchain_summary: Dict, 
    vis_dir: str,
    output_dir: str,
    model: str
):
    """Generate a detailed HTML report from the comparison results."""
    logger.info("Generating HTML comparison report...")
    
    # Load the template
    try:
        with open("report_template.html", "r") as f:
            template = f.read()
    except FileNotFoundError:
        logger.error("Report template not found. Skipping HTML report generation.")
        return None
    
    # Get metrics
    custom_success_rate = custom_summary.get("overall_success_rate", 0)
    langchain_success_rate = langchain_summary.get("overall_success_rate", 0)
    
    # Determine winner
    custom_winner_class = "winner" if custom_success_rate > langchain_success_rate else ""
    langchain_winner_class = "winner" if langchain_success_rate > custom_success_rate else ""
    
    # Format category list
    categories_tested = ", ".join(custom_summary.get("categories_tested", []))
    
    # Calculate total tasks
    total_tasks = custom_summary.get("total_tasks", 0)
    
    # Get successful tasks counts
    custom_successful = custom_summary.get("successful_tasks", 0)
    langchain_successful = langchain_summary.get("successful_tasks", 0)
    
    # Create table rows
    table_rows = ""
    
    # Add overall metrics
    table_rows += f"""
        <tr>
            <td>Overall Success Rate</td>
            <td class="{custom_winner_class}">{custom_success_rate:.2%}</td>
            <td class="{langchain_winner_class}">{langchain_success_rate:.2%}</td>
            <td>{(custom_success_rate - langchain_success_rate) * 100:.1f}%</td>
        </tr>
    """
    
    # Add category metrics
    custom_cat_rates = custom_summary.get("category_success_rates", {})
    langchain_cat_rates = langchain_summary.get("category_success_rates", {})
    
    # Find common categories
    common_categories = set(custom_cat_rates.keys()) & set(langchain_cat_rates.keys())
    
    for category in common_categories:
        custom_rate = custom_cat_rates.get(category, 0)
        langchain_rate = langchain_cat_rates.get(category, 0)
        diff = (custom_rate - langchain_rate) * 100
        
        # Determine winner for this category
        cat_custom_winner = "winner" if custom_rate > langchain_rate else ""
        cat_langchain_winner = "winner" if langchain_rate > custom_rate else ""
        
        table_rows += f"""
            <tr>
                <td>{category.capitalize()} Success Rate</td>
                <td class="{cat_custom_winner}">{custom_rate:.2%}</td>
                <td class="{cat_langchain_winner}">{langchain_rate:.2%}</td>
                <td>{diff:.1f}%</td>
            </tr>
        """
    
    # Create category sections
    category_sections = ""
    
    for category in common_categories:
        category_sections += f"""
            <div class="category-section">
                <h3>{category.capitalize()} Tasks</h3>
                <p>Success Rate: Custom React: {custom_cat_rates.get(category, 0):.2%} vs LangChain React: {langchain_cat_rates.get(category, 0):.2%}</p>
                
                <!-- Task details would go here in a full implementation -->
                <p>Detailed task responses are available in the JSON results files.</p>
            </div>
        """
    
    # Format image paths
    overall_chart_path = os.path.join("visualizations", "overall_success_rate.png")
    category_chart_path = os.path.join("visualizations", "category_success_rates.png")
    
    # Replace placeholders in template
    report_html = template.replace("{{GENERATION_DATE}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report_html = report_html.replace("{{MODEL_NAME}}", model)
    report_html = report_html.replace("{{CATEGORIES_TESTED}}", categories_tested)
    report_html = report_html.replace("{{TOTAL_TASKS}}", str(total_tasks))
    
    report_html = report_html.replace("{{CUSTOM_WINNER_CLASS}}", custom_winner_class)
    report_html = report_html.replace("{{LANGCHAIN_WINNER_CLASS}}", langchain_winner_class)
    
    report_html = report_html.replace("{{CUSTOM_SUCCESS_RATE}}", f"{custom_success_rate:.2%}")
    report_html = report_html.replace("{{LANGCHAIN_SUCCESS_RATE}}", f"{langchain_success_rate:.2%}")
    
    report_html = report_html.replace("{{CUSTOM_SUCCESSFUL_TASKS}}", str(custom_successful))
    report_html = report_html.replace("{{LANGCHAIN_SUCCESSFUL_TASKS}}", str(langchain_successful))
    
    report_html = report_html.replace("{{OVERALL_CHART_PATH}}", overall_chart_path)
    report_html = report_html.replace("{{CATEGORY_CHART_PATH}}", category_chart_path)
    
    report_html = report_html.replace("{{TABLE_ROWS}}", table_rows)
    report_html = report_html.replace("{{CATEGORY_SECTIONS}}", category_sections)
    
    # Save the report
    report_path = os.path.join(output_dir, "comparison_report.html")
    with open(report_path, "w") as f:
        f.write(report_html)
    
    logger.info(f"HTML comparison report saved to {report_path}")
    return report_path

async def main():
    """Main entry point for the comparison script."""
    parser = argparse.ArgumentParser(description="Compare LangChain ReAct agent with custom React agent")
    
    # Evaluation parameters
    parser.add_argument("--model", type=str, default="gpt-4",
                      help="LLM model to use (default: gpt-4)")
    parser.add_argument("--categories", nargs="+", 
                      default=["reasoning", "planning", "tool_use"],
                      help="Categories to evaluate")
    parser.add_argument("--task-limit", type=int, default=3,
                      help="Limit number of tasks per category")
    parser.add_argument("--output-dir", type=str, default="agent_comparison",
                      help="Directory to save results")
    parser.add_argument("--skip-langchain", action="store_true",
                      help="Skip LangChain agent evaluation (evaluate only custom agent)")
    parser.add_argument("--langchain-only", action="store_true",
                      help="Evaluate only the LangChain agent (skip custom agent)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize and evaluate agents based on command line flags
        if args.langchain_only and args.skip_langchain:
            logger.error("Cannot specify both --skip-langchain and --langchain-only")
            return 1
        
        custom_results = None
        custom_summary = None
        custom_file = None
        langchain_results = None
        langchain_summary = None
        langchain_file = None
        
        # Evaluate custom agent if not skipped
        if not args.langchain_only:
            logger.info("Initializing custom React agent...")
            custom_example = ReactAgentExample(model=args.model, max_iterations=10)
            custom_agent = custom_example.agent
            
            custom_results, custom_summary, custom_file = await evaluate_agent(
                agent=custom_agent,
                name="Custom_React",
                model=args.model,
                output_dir=args.output_dir,
                categories=args.categories,
                task_limit=args.task_limit
            )
            
            logger.info(f"Custom React success rate: {custom_summary.get('overall_success_rate', 0):.2%}")
            logger.info(f"Custom results: {custom_file}")
        
        # Evaluate LangChain agent if not skipped
        if not args.skip_langchain:
            try:
                logger.info("Initializing LangChain ReAct agent...")
                langchain_agent = LangChainReActWrapper(model=args.model)
                
                langchain_results, langchain_summary, langchain_file = await evaluate_agent(
                    agent=langchain_agent,
                    name="LangChain_React",
                    model=args.model,
                    output_dir=args.output_dir,
                    categories=args.categories,
                    task_limit=args.task_limit
                )
                
                logger.info(f"LangChain React success rate: {langchain_summary.get('overall_success_rate', 0):.2%}")
                logger.info(f"LangChain results: {langchain_file}")
                
            except Exception as e:
                logger.error(f"Error evaluating LangChain agent: {e}")
                if args.langchain_only:
                    return 1
        
        # Generate comparison if both agents were evaluated
        if custom_results and langchain_results:
            # Create comparison visualizations
            vis_dir = create_comparison_visualizations(
                custom_results=custom_results,
                langchain_results=langchain_results,
                output_dir=args.output_dir
            )
            
            # Generate HTML report
            report_path = generate_html_report(
                custom_results=custom_results,
                langchain_results=langchain_results,
                custom_summary=custom_summary,
                langchain_summary=langchain_summary,
                vis_dir=vis_dir,
                output_dir=args.output_dir,
                model=args.model
            )
            
            # Print summary comparison
            logger.info("===== AGENT COMPARISON SUMMARY =====")
            logger.info(f"Custom React success rate: {custom_summary.get('overall_success_rate', 0):.2%}")
            logger.info(f"LangChain React success rate: {langchain_summary.get('overall_success_rate', 0):.2%}")
            logger.info(f"Visualizations saved to: {vis_dir}")
            logger.info(f"HTML report saved to: {report_path}")
            
    except Exception as e:
        logger.error(f"Error in evaluation process: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exitcode = asyncio.run(main())
    sys.exit(exitcode) 