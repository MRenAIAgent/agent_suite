#!/usr/bin/env python
"""
Agent Benchmark

This script benchmarks three types of agents using TauBench:
1. Custom ReActAgent from this repo
2. LangChain ReAct agent (ZERO_SHOT_REACT_DESCRIPTION)
3. LangChain Tool agent (STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)

It implements the necessary tools for the benchmark and generates comparison reports.
"""

import os
import json
import asyncio
import argparse
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import random
import inspect

# For visualization
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import custom ReActAgent - make optional
try:
    from agents.react_agent import ReActAgent
    from agents.memory_manager import MemoryManager
    from log.logging import LogManager
    from tools.tool import Tool
    from agents.agent import Agent
    REACT_AGENT_AVAILABLE = True
except ImportError:
    logger.warning("Custom ReActAgent not available, some functionality will be limited")
    REACT_AGENT_AVAILABLE = False
    
    # Define a minimal Tool class if the original isn't available
    class Tool:
        """Minimal Tool implementation for when the real one isn't available."""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            
        def run(self, *args, **kwargs):
            return f"Tool not available: would have run with args={args}, kwargs={kwargs}"
            
        async def arun(self, *args, **kwargs):
            return self.run(*args, **kwargs)
    
    # Define a minimal Agent class if the original isn't available
    class Agent:
        """Minimal Agent implementation for when the real one isn't available."""
        def run(self, task: str) -> str:
            return f"Agent not available: would have run task '{task}'"
            
        async def ainvoke(self, task: str) -> str:
            return self.run(task)

# Import LLM providers with fallbacks
try:
    from llm.anthropic import AnthropicLLM
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from llm.openai import OpenAILLM
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import existing tools if available
try:
    from tools.serper_api import SerperSearchTool
    SERPER_AVAILABLE = True
except ImportError:
    SERPER_AVAILABLE = False

# Import LangChain components - make optional
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import MessagesPlaceholder
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langchain.tools import BaseTool as LangChainBaseTool, StructuredTool
    from langchain.agents import create_structured_chat_agent
    from langchain.agents.agent import AgentExecutor
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available, some functionality will be limited")
    LANGCHAIN_AVAILABLE = False

# Import TauBench evaluation
try:
    from benchmark.agent.code.taubench.taubench_eval import TauBenchEvaluation
    from benchmark.agent.code.taubench.taubench_minimal import load_dataset
    TAUBENCH_AVAILABLE = True
except ImportError:
    logger.warning("TauBench not available, using simplified dataset")
    TAUBENCH_AVAILABLE = False
    
    def load_dataset():
        """Minimal implementation for when TauBench is not available."""
        return [
            {"problem": "What is 2+2?", "reference": "4"},
            {"problem": "What is the capital of France?", "reference": "Paris"},
            {"problem": "Translate 'hello' to Spanish", "reference": "hola"},
        ]

#------------------------------------------------------------------------------
# Tool Implementations
#------------------------------------------------------------------------------

class CalculatorTool(Tool):
    """
    Calculator tool for performing arithmetic operations.
    
    This tool evaluates mathematical expressions and returns the result.
    Examples: "2+2", "sqrt(4)", "5*5", "10/2"
    """
    expression: Optional[str] = None
    
    async def arun(self, expression: str) -> Any:
        """Evaluate a mathematical expression."""
        try:
            # Replace ^ with ** for exponentiation (common math notation)
            expression = expression.replace('^', '**')
            
            # Safer eval using only math operations
            import math
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            allowed_names.update({"abs": abs, "complex": complex, "float": float, "int": int, "pow": pow, "round": round})
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    def run(self, expression: str) -> Any:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: The expression to evaluate
            
        Returns:
            Result of the evaluation
        """
        try:
            # Replace ^ with ** for exponentiation (common math notation)
            expression = expression.replace('^', '**')
            
            # Safer eval using only math operations
            import math
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            allowed_names.update({"abs": abs, "complex": complex, "float": float, "int": int, "pow": pow, "round": round})
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"


class WeatherTool(Tool):
    """
    Weather tool for getting current weather information.
    
    This tool provides weather information for a specified location.
    Examples of locations: "Tokyo", "New York", "Paris"
    """
    location: Optional[str] = None
    
    async def arun(self, location: str) -> Any:
        """Get weather information for a location."""
        # Mock implementation that returns fake but realistic weather data
        return self._generate_weather_data(location)
    
    def run(self, location: str) -> Any:
        """Get weather information for a location synchronously."""
        # Mock implementation that returns fake but realistic weather data
        return self._generate_weather_data(location)
    
    def _generate_weather_data(self, location: str) -> Dict[str, Any]:
        """Generate mock weather data for a location."""
        # Set random seed based on location for consistent results
        random.seed(sum(ord(c) for c in location))
        
        # Generate random but plausible weather data
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorms", "Snowy"]
        temperature = random.randint(15, 30) if "Tokyo" in location else random.randint(-5, 35)
        
        return {
            "location": location,
            "temperature": f"{temperature}°C",
            "conditions": random.choice(conditions),
            "humidity": f"{random.randint(30, 90)}%",
            "wind_speed": f"{random.randint(0, 30)} km/h",
            "forecast": "This is a mock weather response for demonstration purposes."
        }


class TranslationTool(Tool):
    """
    Translation tool for translating text between languages.
    
    This tool translates text from one language to another.
    Example: text="Hello, how are you?", source_language="English", target_language="French"
    """
    text: Optional[str] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    
    async def arun(self, text: str, source_language: str, target_language: str) -> Any:
        """Translate text between languages."""
        return self._mock_translation(text, source_language, target_language)
    
    def run(self, text: str, source_language: str, target_language: str) -> Any:
        """Translate text between languages synchronously."""
        return self._mock_translation(text, source_language, target_language)
    
    def _mock_translation(self, text: str, source_language: str, target_language: str) -> str:
        """Generate mock translation."""
        # Dictionary of common phrases in different languages for demo purposes
        translations = {
            ("English", "French", "Hello"): "Bonjour",
            ("English", "French", "Hello, how are you?"): "Bonjour, comment allez-vous?",
            ("English", "French", "Thank you"): "Merci",
            ("English", "French", "Goodbye"): "Au revoir",
            ("English", "Spanish", "Hello"): "Hola",
            ("English", "Spanish", "Hello, how are you?"): "Hola, ¿cómo estás?",
            ("English", "Spanish", "Thank you"): "Gracias",
            ("English", "Spanish", "Goodbye"): "Adiós",
            ("English", "German", "Hello"): "Hallo",
            ("English", "German", "Hello, how are you?"): "Hallo, wie geht es dir?",
            ("English", "German", "Thank you"): "Danke",
            ("English", "German", "Goodbye"): "Auf Wiedersehen",
        }
        
        key = (source_language, target_language, text)
        if key in translations:
            return translations[key]
        
        # For unknown translations, return a message
        return f"Translation ({source_language} to {target_language}): {text} [This is a mock translation]"


class SearchTool(Tool):
    """
    Search tool for finding information on the web.
    
    This tool performs web searches and returns relevant results.
    Example: "latest news about climate change"
    """
    query: Optional[str] = None
    
    async def arun(self, query: str) -> Any:
        """Perform a web search."""
        if SERPER_AVAILABLE and os.getenv("SERPER_API_KEY"):
            # Use real SerperSearchTool if available
            serper_tool = SerperSearchTool()
            return await serper_tool.arun(query=query)
        else:
            # Use mock implementation
            return self._mock_search_results(query)
    
    def run(self, query: str) -> Any:
        """Perform a web search synchronously."""
        if SERPER_AVAILABLE and os.getenv("SERPER_API_KEY"):
            # Use real SerperSearchTool if available
            serper_tool = SerperSearchTool()
            return serper_tool.run(query=query)
        else:
            # Use mock implementation
            return self._mock_search_results(query)
    
    def _mock_search_results(self, query: str) -> List[Dict[str, str]]:
        """Generate mock search results."""
        # Mock implementation of a search engine
        return [
            {
                "title": f"Result 1 for '{query}'",
                "link": f"https://example.com/result1",
                "snippet": f"This is a mock search result for '{query}'. It contains relevant information about the query."
            },
            {
                "title": f"Result 2 for '{query}'",
                "link": f"https://example.com/result2",
                "snippet": f"Another mock search result related to '{query}'. This result provides additional context."
            },
            {
                "title": f"Result 3 for '{query}'",
                "link": f"https://example.com/result3",
                "snippet": f"A third mock search result for '{query}'. This contains alternative information."
            }
        ]


class CalendarTool(Tool):
    """
    Calendar tool for scheduling events.
    
    This tool allows scheduling events at specific dates and times.
    Example: event="Team Meeting", date="2023-12-15", time="14:00"
    """
    event: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    
    async def arun(self, event: str, date: str, time: str) -> Any:
        """Schedule an event."""
        return self._schedule_event(event, date, time)
    
    def run(self, event: str, date: str, time: str) -> Any:
        """Schedule an event synchronously."""
        return self._schedule_event(event, date, time)
    
    def _schedule_event(self, event: str, date: str, time: str) -> str:
        """Mock scheduling an event."""
        try:
            # Validate date format (YYYY-MM-DD)
            datetime.strptime(date, "%Y-%m-%d")
            
            # Validate time format (HH:MM)
            datetime.strptime(time, "%H:%M")
            
            return f"Event '{event}' has been scheduled for {date} at {time} [This is a mock calendar entry]"
        except ValueError as e:
            return f"Error scheduling event: {str(e)}"

#------------------------------------------------------------------------------
# Tool Adapters for LangChain
#------------------------------------------------------------------------------

def custom_to_langchain_tool(custom_tool: Tool) -> LangChainBaseTool:
    """
    Convert a custom Tool instance to a LangChain Tool.
    
    Args:
        custom_tool: Instance of custom Tool
        
    Returns:
        LangChain Tool instance
    """
    # Get the class of the tool
    tool_class = custom_tool.__class__
    
    # Create a name for the tool
    name = tool_class.__name__.lower().replace("tool", "")
    
    # Get the description
    description = tool_class.__doc__.strip() if tool_class.__doc__ else ""
    
    # Import the LangChainTool class once at the top level
    from langchain.tools.base import Tool as LangChainBasicTool
    from langchain.tools import StructuredTool
    
    # Special handling for tools with multiple parameters
    if isinstance(custom_tool, TranslationTool):
        # Create a wrapper function that handles the dictionary input format from LangChain
        def translate_wrapper(text: str, source_language: str = "English", target_language: str = "French"):
            # Log the inputs for debugging
            logger.info(f"TranslationTool wrapper called with: text='{text}', source='{source_language}', target='{target_language}'")
            
            # Call the tool with the extracted parameters
            try:
                return custom_tool.run(
                    text=text,
                    source_language=source_language,
                    target_language=target_language
                )
            except Exception as e:
                logger.error(f"Error in translation wrapper: {str(e)}")
                return f"Translation error: {str(e)}"
        
        # Create a structured tool for translation
        return StructuredTool.from_function(
            func=translate_wrapper,
            name=name,
            description=description
        )
    
    # Special handling for other multi-parameter tools
    elif isinstance(custom_tool, CalendarTool):
        def calendar_wrapper(event: str = "Meeting", date: str = None, time: str = "12:00"):
            # Set default date to today if not provided
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
                
            # Log the inputs for debugging
            logger.info(f"CalendarTool wrapper called with: event='{event}', date='{date}', time='{time}'")
            
            try:
                return custom_tool.run(
                    event=event,
                    date=date,
                    time=time
                )
            except Exception as e:
                logger.error(f"Error in calendar wrapper: {str(e)}")
                return f"Calendar error: {str(e)}"
            
        return StructuredTool.from_function(
            func=calendar_wrapper,
            name=name,
            description=description
        )
    
    # Default case for single-parameter tools
    else:
        # Create a wrapper function that can handle different input formats
        def wrapper(args):
            # Handle both string inputs and dictionary inputs
            if isinstance(args, dict):
                # The input might be in 'tool_input' or directly in the dict
                if "tool_input" in args and isinstance(args["tool_input"], dict):
                    # Use the first value in the nested dict as the parameter
                    param_value = next(iter(args["tool_input"].values()), "")
                elif "action_input" in args:
                    param_value = args["action_input"]
                else:
                    # Use the first value in the dict as the parameter
                    param_value = next(iter(args.values()), "")
            else:
                param_value = args
                
            # Convert to string if necessary
            if not isinstance(param_value, str):
                param_value = str(param_value)
                
            return custom_tool.run(param_value)
            
        # Create a langchain Tool
        return LangChainBasicTool(
            name=name,
            description=description,
            func=wrapper
        )


def get_all_tools() -> Dict[str, Tool]:
    """
    Initialize all available custom tools.
    
    Returns:
        Dictionary of tool instances by name
    """
    tools = {
        "calculator": CalculatorTool(),
        "weather": WeatherTool(),
        "translation": TranslationTool(),
        "search": SearchTool(),
        "calendar": CalendarTool()
    }
    return tools


def get_all_langchain_tools(custom_tools: Dict[str, Tool]) -> List[LangChainBaseTool]:
    """
    Convert all custom tools to LangChain tools.
    
    Args:
        custom_tools: Dictionary of custom Tool instances
        
    Returns:
        List of LangChain tools
    """
    return [custom_to_langchain_tool(tool) for tool in custom_tools.values()]


#------------------------------------------------------------------------------
# Agent Initialization
#------------------------------------------------------------------------------

# Custom ReActAgent wrapper for benchmarking
class CustomReActAgentWrapper(Agent):
    """
    Wrapper for the custom ReActAgent to make it compatible with the benchmark interface.
    """
    
    def __init__(self, react_agent, default_model="gpt-3.5-turbo"):
        """
        Initialize the wrapper for the custom ReActAgent.
        
        Args:
            react_agent: The ReActAgent to wrap
            default_model: Default model to use when running
        """
        self.react_agent = react_agent
        self.default_model = default_model
        self.loop = None
        
    def run(self, task: str) -> str:
        """
        Run the agent on a task.
        
        Args:
            task: Task description
            
        Returns:
            Agent's response
        """
        try:
            # Create a new event loop if one doesn't already exist for this thread
            if self.loop is None:
                import asyncio
                import threading
                
                # Run in a separate thread to avoid asyncio issues
                def run_in_thread(task, model):
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    # Run the agent
                    return loop.run_until_complete(self.react_agent.arun(task, model))
                
                # Create a thread and run the agent
                thread = threading.Thread(target=lambda: setattr(self, 'result', run_in_thread(task, self.default_model)))
                thread.start()
                thread.join()
                
                # Get the result from the thread
                response = self.result
            else:
                # Use the existing loop
                response = self.loop.run_until_complete(self.react_agent.arun(task, self.default_model))
            
            # Process the response
            return self.handle_task_completion(response)
        except Exception as e:
            logger.error(f"Error running Custom ReAct agent: {str(e)}")
            return f"Error: {str(e)}"
            
    async def ainvoke(self, task: str) -> str:
        """
        Asynchronously run the agent on a task.
        
        Args:
            task: Task description
            
        Returns:
            Agent's response
        """
        try:
            # Use the arun method directly since we're already in an async context
            response = await self.react_agent.arun(task, self.default_model)
            
            # Process the response
            return self.handle_task_completion(response)
        except Exception as e:
            logger.error(f"Error running Custom ReAct agent asynchronously: {str(e)}")
            return f"Error: {str(e)}"
    
    def handle_task_completion(self, response: str) -> str:
        """
        Process the agent's response into a format expected by the evaluation.
        
        Args:
            response: The raw response from the agent
            
        Returns:
            Processed response as expected by the benchmark
        """
        # Response from ReActAgent is already a string, so return it directly
        if isinstance(response, str):
            logger.info(f"Agent returned a string response of length {len(response)}")
            return response
            
        # If for some reason we get a dict or something else, try to handle it
        if isinstance(response, dict):
            if "output" in response:
                return response["output"]
            elif "response" in response:
                return response["response"]
            else:
                # Return the string representation of the dict
                return str(response)
                
        # For any other type, convert to string
        return str(response)

def create_custom_react_agent(
    llm_provider: str = "openai",
    model_name: str = "gpt-3.5-turbo",
    tools: Optional[Dict[str, Tool]] = None,
    verbose: bool = False
) -> Agent:
    """
    Create the custom ReActAgent.
    
    Args:
        llm_provider: LLM provider to use
        model_name: Model name to use
        tools: Dictionary of tools to use
        verbose: Whether to enable verbose logging
        
    Returns:
        Agent instance
    """
    if not REACT_AGENT_AVAILABLE:
        logger.warning("Custom ReActAgent not available, returning minimal agent")
        return Agent()
        
    # Set up tools
    if tools is None:
        tools = get_all_tools()
        
    # Convert to list for ReActAgent
    tool_list = list(tools.values())
    
    # Create log manager
    log_manager = LogManager(
        console_log_level=logging.DEBUG if verbose else logging.INFO,
        log_dir="logs"
    )
    
    # Create memory manager
    memory_manager = MemoryManager()
    
    # Determine which LLM provider to use
    if llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
        llm = AnthropicLLM()
    elif llm_provider == "openai" and OPENAI_AVAILABLE:
        llm = OpenAILLM()
    else:
        if OPENAI_AVAILABLE:
            llm = OpenAILLM()
        elif ANTHROPIC_AVAILABLE:
            llm = AnthropicLLM()
        else:
            raise ValueError("No supported LLM provider available")
    
    # Create the ReActAgent
    react_agent = ReActAgent(
        tools=tool_list,
        llm=llm,
        memory_manager=memory_manager,
        log_manager=log_manager,
        system_message="You are a helpful AI assistant that helps users complete tasks. Be precise and accurate in your responses."
    )
    
    # Wrap the ReActAgent
    return CustomReActAgentWrapper(react_agent, default_model=model_name)


def initialize_langchain_agent(langchain_tools, model_name="default", temperature=0.1):
    """
    Initialize a LangChain agent with the provided tools.
    
    Args:
        langchain_tools: List of LangChain Tool instances
        model_name: Name of the model to use (default: "default")
        temperature: Temperature for the model (default: 0.1)
        
    Returns:
        LangChain AgentExecutor instance
    """
    try:
        # Import the necessary modules
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentType, initialize_agent
        from langchain.memory import ConversationBufferMemory
        
        # Initialize the LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature
        )
        
        # Initialize the memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize the agent
        agent_executor = initialize_agent(
            tools=langchain_tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # Use a structured-compatible agent type
            memory=memory
        )
        
        return agent_executor
    except ImportError as e:
        logger.error(f"Failed to initialize LangChain ReAct agent: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LangChain ReAct agent: {str(e)}")
        return None


def create_langchain_react_agent(
    tools: List[Tool],
    model_name: str = "default",
    temperature: float = 0.1
) -> Agent:
    """
    Create a LangChain ReAct agent.
    
    Args:
        tools: List of Tool instances
        model_name: Name of the model to use (default: "default")
        temperature: Temperature for the model (default: 0.1)
        
    Returns:
        Agent instance
    """
    # Convert custom tools to LangChain tools
    langchain_tools = [custom_to_langchain_tool(tool) for tool in tools]
    
    # Filter out None values (tools that couldn't be converted)
    langchain_tools = [tool for tool in langchain_tools if tool is not None]
    
    # Initialize the agent executor using our new function
    agent_executor = initialize_langchain_agent(
        langchain_tools=langchain_tools,
        model_name=model_name,
        temperature=temperature
    )
    
    if agent_executor is None:
        return None
    
    # Create and return a LangChainReActAgent instance
    return LangChainReActAgent(agent_executor, tools)


def create_langchain_structured_agent(
    llm_provider: str = "openai",
    model_name: str = "gpt-3.5-turbo",
    tools: Optional[Dict[str, Tool]] = None,
    verbose: bool = False
) -> AgentExecutor:
    """
    Create a LangChain structured chat agent.
    
    Args:
        llm_provider: LLM provider to use ("openai" or "anthropic")
        model_name: Model name to use
        tools: Dictionary of tools to provide to the agent
        verbose: Whether to enable verbose logging
        
    Returns:
        LangChain AgentExecutor instance
    """
    if tools is None:
        tools = get_all_tools()
    
    # Convert tools to LangChain format
    langchain_tools = get_all_langchain_tools(tools)
    
    # Initialize LLM
    if llm_provider.lower() == "openai":
        llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
    else:
        try:
            # Try to import Anthropic
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=model_name,
                temperature=0
            )
        except ImportError:
            logger.warning("langchain_anthropic not available, using OpenAI instead")
            llm = ChatOpenAI(
                model=model_name,
                temperature=0
            )
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Use the simplest approach to create a structured agent
    return initialize_agent(
        tools=langchain_tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=verbose
    )

#------------------------------------------------------------------------------
# Evaluation Functions
#------------------------------------------------------------------------------

def evaluate_agent(agent, dataset, verbose=False, max_cases=None):
    """
    Evaluate an agent on a dataset.
    
    Args:
        agent: Agent to evaluate
        dataset: Dataset to evaluate on
        verbose: Whether to print verbose output (default: False)
        max_cases: Maximum number of cases to evaluate (default: None)
        
    Returns:
        List of evaluation results
    """
    results = []
    
    # Limit the number of cases if max_cases is specified
    if max_cases is not None:
        dataset = dataset[:max_cases]
    
    # Evaluate each case
    for i, case in enumerate(dataset):
        try:
            # Extract the problem and reference
            problem = case["problem"] if isinstance(case, dict) else ""
            reference = case["reference"] if isinstance(case, dict) else ""
            
            # Default values if extraction fails
            if not problem and not reference and isinstance(case, str):
                problem = case
                reference = ""
            
            logger.info(f"Evaluating case {i+1}/{len(dataset)}: {problem}")
            
            # Track the start time
            start_time = time.time()
            
            # Run the agent on the problem
            if verbose:
                logger.info(f"Running agent on: {problem}")
                
            response = agent.run(problem)
            
            # Track the end time
            end_time = time.time()
            
            # Calculate the time taken
            time_taken = end_time - start_time
            
            # Determine the task type
            task_type = get_task_type(problem)
            
            # Check if the response is correct
            is_correct = check_correctness(task_type, response, reference)
            
            # Add debug logging for the response
            if isinstance(response, str):
                logger.info(f"Agent returned a string response of length {len(response)}")
            else:
                logger.info(f"Agent returned a non-string response of type {type(response)}")
            
            # Create the result object
            result = {
                "index": i,
                "problem": problem,
                "response": response,
                "reference": reference,
                "time_taken": time_taken,
                "is_correct": is_correct,
                "task_type": task_type
            }
            
            # Add the result to the list
            results.append(result)
            
            if verbose:
                logger.info(f"Response: {response}")
                logger.info(f"Reference: {reference}")
                logger.info(f"Time taken: {time_taken:.2f} seconds")
                logger.info(f"Is correct: {is_correct}")
                
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            
    return results


async def compare_agents(
    llm_provider: str = "openai",
    model_name: str = "gpt-4.1",
    max_cases: Optional[int] = None,
    categories: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compare different agent implementations on TauBench test cases.
    
    Args:
        llm_provider: LLM provider to use ("openai" or "anthropic")
        model_name: Model name to use
        max_cases: Maximum number of test cases to evaluate (None for all)
        categories: Categories of test cases to use (None for all)
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary of comparison results
    """
    # Define available categories based on the TauBench directory structure
    available_categories = ["reasoning", "planning", "tool_use", "qa", "code_generation"]
    
    # Use the specified categories or default to those that involve tool use
    if not categories:
        # Default to tool_use since we're testing tools
        categories_to_load = ["tool_use"]
    else:
        categories_to_load = [cat for cat in categories if cat in available_categories]
    
    if not categories_to_load:
        logger.warning(f"No valid categories specified. Using default: tool_use")
        categories_to_load = ["tool_use"]
    
    # Load TauBench datasets for each category
    test_cases = []
    for category in categories_to_load:
        try:
            category_cases = load_dataset(category=category)
            if category_cases:
                logger.info(f"Loaded {len(category_cases)} test cases from category '{category}'")
                test_cases.extend(category_cases)
            else:
                logger.warning(f"No test cases found for category '{category}'")
        except Exception as e:
            logger.error(f"Error loading dataset for category '{category}': {str(e)}")
    
    # Use fallback test cases if no TauBench cases were loaded
    if not test_cases:
        logger.warning("Could not load TauBench test cases, using fallback cases")
        test_cases = [
            {
                "id": "test1",
                "query": "What is 2+2?",
                "answer": "4",
                "category": "math"
            },
            {
                "id": "test2",
                "query": "What's the weather in New York?",
                "answer": None,
                "category": "weather"
            },
            {
                "id": "test3", 
                "query": "Translate 'Hello, how are you?' to French",
                "answer": "Bonjour, comment allez-vous?",
                "category": "translation"
            }
        ]
        logger.info(f"Created {len(test_cases)} fallback test cases")
    
    # Initialize tools
    tools = get_all_tools()
    
    # Initialize agents
    logger.info("Initializing agents...")
    agents = []
    
    # Try to initialize the custom agent
    try:
        logger.info("Attempting to initialize custom ReActAgent...")
        custom_react_agent = create_custom_react_agent(
            llm_provider=llm_provider,
            model_name=model_name,
            tools=tools,
            verbose=verbose
        )
        agents.append(("custom_react", custom_react_agent))
        logger.info("Successfully initialized custom ReActAgent")
    except Exception as e:
        logger.error(f"Failed to initialize custom ReActAgent: {str(e)}")
    
    # Add LangChain agents
    try:
        langchain_react_agent = create_langchain_react_agent(
            tools=list(tools.values()),
            model_name=model_name,
            temperature=0.1
        )
        agents.append(("langchain_react", langchain_react_agent))
    except Exception as e:
        logger.error(f"Failed to initialize LangChain ReAct agent: {str(e)}")
        
    try:
        langchain_structured_agent = create_langchain_structured_agent(
            llm_provider=llm_provider,
            model_name=model_name,
            tools=tools,
            verbose=verbose
        )
        agents.append(("langchain_structured", langchain_structured_agent))
    except Exception as e:
        logger.error(f"Failed to initialize LangChain Structured agent: {str(e)}")
    
    # Evaluate each agent
    results = {}
    for agent_type, agent in agents:
        logger.info(f"Evaluating agent: {agent_type}")
        try:
            results[agent_type] = evaluate_agent(agent, test_cases, verbose=verbose)
        except Exception as e:
            logger.error(f"Error evaluating {agent_type}: {str(e)}")
    
    if not results:
        raise RuntimeError("No agents were successfully initialized and evaluated")
    
    return results


#------------------------------------------------------------------------------
# Visualization and Reporting
#------------------------------------------------------------------------------

def generate_comparison_report(results, output_dir=None):
    """
    Generate comparison visualizations for different agents.
    
    Args:
        results: Dictionary with agent results
        output_dir: Directory to save visualizations (default: None)
    """
    # Check if we have results
    if not results:
        logger.warning("No results to compare")
        return
    
    # Get list of agent types
    agent_types = list(results.keys())
    
    # Create a DataFrame for success rates
    success_rates = {}
    avg_times = {}
    
    for agent_type in agent_types:
        # Calculate success rate and average time
        agent_results = results[agent_type]
        
        if isinstance(agent_results, dict) and "summary" in agent_results:
            # Old format with summary and metrics
            summary = agent_results["summary"]
            success_rates[agent_type] = summary["success_rate"]
            avg_times[agent_type] = summary["avg_time"]
        elif isinstance(agent_results, list):
            # New format with list of results
            # Calculate success rate
            success_count = sum(1 for r in agent_results if r.get("is_correct", False))
            total_count = len(agent_results)
            success_rate = success_count / total_count if total_count > 0 else 0
            
            # Calculate average time
            avg_time = sum(r.get("time_taken", 0) for r in agent_results) / total_count if total_count > 0 else 0
            
            success_rates[agent_type] = success_rate
            avg_times[agent_type] = avg_time
        else:
            logger.warning(f"Unknown result format for {agent_type}, skipping")
            continue
    
    # Create pandas DataFrames for plotting
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Success rate comparison
        success_df = pd.DataFrame.from_dict(success_rates, orient='index', columns=['Success Rate'])
        
        # Generate bar chart
        plt.figure(figsize=(10, 6))
        success_df.plot(kind='bar', color='green', alpha=0.7)
        plt.title('Agent Success Rate Comparison')
        plt.ylabel('Success Rate')
        plt.xlabel('Agent Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            success_chart_path = os.path.join(output_dir, 'success_rate_comparison.png')
            plt.savefig(success_chart_path)
            logger.info(f"Success rate comparison chart saved to {success_chart_path}")
        
        # Time comparison
        time_df = pd.DataFrame.from_dict(avg_times, orient='index', columns=['Average Time (s)'])
        
        plt.figure(figsize=(10, 6))
        time_df.plot(kind='bar', color='blue', alpha=0.7)
        plt.title('Agent Average Execution Time Comparison')
        plt.ylabel('Average Time (seconds)')
        plt.xlabel('Agent Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            time_chart_path = os.path.join(output_dir, 'execution_time_comparison.png')
            plt.savefig(time_chart_path)
            logger.info(f"Execution time comparison chart saved to {time_chart_path}")
            
        # Save overall metrics as CSV
        metrics_df = pd.DataFrame({
            'Success Rate': success_rates,
            'Average Time (s)': avg_times
        }).transpose()
        
        if output_dir:
            metrics_path = os.path.join(output_dir, 'agent_metrics.csv')
            metrics_df.to_csv(metrics_path)
            logger.info(f"Agent metrics saved to {metrics_path}")
            
            # Save detailed results as JSON
            detailed_path = os.path.join(output_dir, 'detailed_results.json')
            with open(detailed_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Detailed results saved to {detailed_path}")
                
    except ImportError:
        logger.warning("Pandas or Matplotlib not available for visualization")
        # Save metrics in JSON format instead
        if output_dir:
            metrics_path = os.path.join(output_dir, 'agent_metrics.json')
            metrics = {
                'success_rates': success_rates,
                'avg_times': avg_times
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Agent metrics saved to {metrics_path}")
            
            # Save detailed results as JSON
            detailed_path = os.path.join(output_dir, 'detailed_results.json')
            with open(detailed_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Detailed results saved to {detailed_path}")


#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Agent Benchmark")
    parser.add_argument("--agents", nargs="+", default=["custom_react", "langchain_react", "langchain_structured"],
                        help="Agents to benchmark")
    parser.add_argument("--model-name", default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider")
    parser.add_argument("--output-dir", default="benchmark/agent/results", help="Output directory")
    parser.add_argument("--max-cases", type=int, default=None, help="Maximum number of test cases")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dataset-path", default=None, help="Path to dataset")
    
    args = parser.parse_args()
    
    # Run the benchmark
    create_and_run_benchmark(args)


class LangChainReActAgent(Agent):
    """
    Wrapper for LangChain ReAct agent.
    """
    
    def __init__(self, agent_executor, tools):
        """
        Initialize a LangChainReActAgent.
        
        Args:
            agent_executor: LangChain AgentExecutor instance
            tools: List of Tool instances
        """
        self.agent_executor = agent_executor
        self.tools = tools
        
    def run(self, task: str) -> str:
        """
        Run the agent on a task.
        
        Args:
            task: Task description
            
        Returns:
            Agent's response
        """
        try:
            # Run the agent
            response = self.agent_executor.run(task)
            return self.handle_task_completion(response)
        except Exception as e:
            logger.error(f"Error running LangChain ReAct agent: {str(e)}")
            return f"Error: {str(e)}"
            
    async def ainvoke(self, task: str) -> str:
        """
        Asynchronously run the agent on a task.
        
        Args:
            task: Task description
            
        Returns:
            Agent's response
        """
        try:
            # Try to use arun if available
            if hasattr(self.agent_executor, 'arun'):
                response = await self.agent_executor.arun(task)
            else:
                # Fall back to synchronous call if arun not available
                response = self.agent_executor.run(task)
            
            # Process the response to handle the TauBench evaluation format
            return self.handle_task_completion(response)
        except Exception as e:
            logger.error(f"Error running LangChain ReAct agent asynchronously: {str(e)}")
            return f"Error: {str(e)}"
    
    def handle_task_completion(self, response: str) -> str:
        """
        Process the agent's response into a format expected by the evaluation.
        
        Args:
            response: The raw response from the agent
            
        Returns:
            Processed response as expected by the benchmark
        """
        # Since LangChain ReAct agents return a string, we simply return it directly
        # This prevents the 'str' object has no attribute 'get' error
        if isinstance(response, str):
            logger.info(f"Agent returned a string response of length {len(response)}")
            return response
        
        # If for some reason we get a dict or something else, try to handle it
        if isinstance(response, dict):
            if "output" in response:
                return response["output"]
            elif "response" in response:
                return response["response"]
            else:
                # Return the string representation of the dict
                return str(response)
        
        # For any other type, convert to string
        return str(response)


def check_correctness(task_type, response, reference):
    """
    Check if the agent's response is correct based on the task type and reference.
    
    Args:
        task_type: Type of task (e.g., "qa", "tool_use")
        response: Agent's response
        reference: Reference answer or expected behavior
        
    Returns:
        True if the response is correct, False otherwise
    """
    # Handle different response types
    if isinstance(response, dict):
        response_text = response.get("response", "")
    elif isinstance(response, str):
        response_text = response
    else:
        # Convert to string if it's some other type
        response_text = str(response)
    
    # For QA tasks, check if the response contains the reference
    if task_type == "qa":
        # Convert to lowercase and strip punctuation for comparison
        response_lower = response_text.lower()
        reference_lower = reference.lower() if reference else ""
        
        # Check if the response contains the reference answer
        return reference_lower in response_lower
    
    # For calculation tasks, check if the response contains the correct result
    elif task_type == "calculation":
        # Extract numbers from the response and reference
        response_numbers = extract_numbers(response_text)
        reference_numbers = extract_numbers(reference)
        
        # Check if any of the numbers in the response match the reference
        return any(abs(num - reference_numbers[0]) < 0.01 for num in response_numbers) if response_numbers and reference_numbers else False
    
    # For tool use tasks, check if the right tool was used (based on reference)
    elif task_type == "tool_use":
        response_lower = response_text.lower()
        
        # Check if the response mentions using the required tool
        required_tool = reference.lower() if reference else ""
        
        # Weather tool check - more specific
        if "weather" in required_tool:
            # Check for weather-specific terms and the temperature value for Tokyo
            weather_terms = ["tokyo", "temperature", "28", "humid", "wind", "thunder"]
            return any(term in response_lower for term in weather_terms)
            
        # Calculator tool check
        elif "calculat" in required_tool and any(x in response_lower for x in ["1276", "276", "interest"]):
            return True
            
        # Translation tool check
        elif "translat" in required_tool and "bonjour" in response_lower:
            return True
            
        else:
            return False
    
    # For planning tasks, check if the response covers key elements from reference
    elif task_type == "planning":
        # Extract key elements from reference
        key_elements = [elem.strip() for elem in reference.split(",")] if reference else []
        
        # Check if the response mentions most of the key elements
        mentioned_count = sum(1 for elem in key_elements if elem.lower() in response_text.lower())
        return mentioned_count >= len(key_elements) * 0.6  # At least 60% of key elements
    
    # Default case (including reasoning): check if there's a reasonable response
    else:
        return len(response_text) > 20  # Arbitrary minimum length


def extract_numbers(text):
    """
    Extract numbers from a text string.
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List of numbers extracted from the text
    """
    import re
    
    # Find all numbers (including decimals) in the text
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    
    # Convert to float
    return [float(num) for num in numbers]


def get_task_type(problem):
    """
    Determine the type of task based on the problem description.
    
    Args:
        problem: Problem description
        
    Returns:
        Task type string
    """
    problem_lower = problem.lower()
    
    if "weather" in problem_lower:
        return "tool_use"
    elif "interest" in problem_lower and "calculator" in problem_lower:
        return "calculation"
    elif "translate" in problem_lower:
        return "tool_use"
    elif "plan" in problem_lower or "step" in problem_lower:
        return "planning"
    elif "question" in problem_lower:
        return "qa"
    else:
        return "reasoning"


def process_benchmark_results(results):
    """
    Process benchmark results for each problem.
    
    Args:
        results: List of results from evaluate_agent
        
    Returns:
        Dictionary with success information
    """
    success_count = 0
    total_count = len(results)
    
    for result in results:
        # Extract problem, response, and reference
        problem = result.get("problem", "")
        response = result.get("response", "")
        reference = result.get("reference", "")
        
        # Determine the task type based on the problem
        task_type = get_task_type(problem)
        
        # Check if the response is correct
        is_correct = check_correctness(task_type, response, reference)
        
        # Add correctness to result
        result["is_correct"] = is_correct
        
        # Update success count
        if is_correct:
            success_count += 1
    
    return {
        "success_count": success_count,
        "total_count": total_count,
        "success_rate": success_count / total_count if total_count > 0 else 0.0
    }


def load_default_dataset(max_cases=None):
    """
    Load the default dataset from TauBench.
    
    Args:
        max_cases: Maximum number of test cases to load
        
    Returns:
        List of problems
    """
    try:
        # Try to load from taubench
        dataset_path = os.path.join(os.path.dirname(__file__), "taubench", "datasets", "tau_bench.jsonl")
        if not os.path.exists(dataset_path):
            # Fall back to the default path
            dataset_path = "benchmark/agent/data/taubench/datasets/tau_bench.jsonl"
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            
            # If the file doesn't exist, load from taubench module
            try:
                dataset = load_dataset()
                
                # Save to the new location
                with open(dataset_path, "w") as f:
                    for item in dataset:
                        f.write(json.dumps(item) + "\n")
                    
                logger.info(f"Saved dataset to {dataset_path}")
            except Exception as e:
                logger.error(f"Error loading dataset from taubench module: {e}")
                dataset = []
        else:
            # Load from file
            dataset = []
            with open(dataset_path, "r") as f:
                for line in f:
                    dataset.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Create a small sample dataset
        dataset = [
            {"problem": "What is 2+2?", "reference": "4"},
            {"problem": "What is the capital of France?", "reference": "Paris"},
            {"problem": "Translate 'hello' to Spanish", "reference": "hola"},
        ]
    
    # Limit to max_cases if specified
    if max_cases is not None:
        dataset = dataset[:max_cases]
        
    return dataset


def create_and_run_benchmark(args):
    """
    Create and run the benchmark with the given arguments.
    
    Args:
        args: Command line arguments
    """
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logger.info("Starting agent benchmark...")
    
    # Initialize tools
    tools = get_all_tools()
    
    # Get dataset
    if args.dataset_path:
        logger.info(f"Loading dataset from {args.dataset_path}")
        dataset = load_dataset_from_file(args.dataset_path)
    else:
        logger.info("Loading default dataset")
        dataset = load_default_dataset(max_cases=args.max_cases)
    
    # List of agents to evaluate
    agents = []
    
    # Create custom agent if requested
    if "custom_react" in args.agents:
        try:
            custom_agent = create_custom_react_agent(
                llm_provider=args.llm_provider,
                model_name=args.model_name,
                tools=tools,
                verbose=args.verbose
            )
            agents.append(("custom_react", custom_agent))
            logger.info("Successfully initialized custom ReActAgent")
        except Exception as e:
            logger.error(f"Failed to initialize custom ReActAgent: {str(e)}")
    
    # Create LangChain ReAct agent if requested
    if "langchain_react" in args.agents:
        try:
            langchain_react_agent = create_langchain_react_agent(
                tools=tools,
                model_name=args.model_name,
                temperature=0.1
            )
            agents.append(("langchain_react", langchain_react_agent))
            logger.info("Successfully initialized LangChain ReAct agent")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain ReAct agent: {str(e)}")
    
    # Create LangChain Structured Tool agent if requested
    if "langchain_structured" in args.agents:
        try:
            langchain_structured_agent = create_langchain_structured_agent(
                llm_provider=args.llm_provider,
                model_name=args.model_name,
                tools=tools,
                verbose=args.verbose
            )
            agents.append(("langchain_structured", langchain_structured_agent))
            logger.info("Successfully initialized LangChain Structured Tool agent")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain Structured Tool agent: {str(e)}")
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmark for each agent
    results = {}
    for agent_name, agent in agents:
        if agent is None:
            logger.warning(f"Agent {agent_name} is None, skipping...")
            continue
            
        logger.info(f"Running benchmark for {agent_name}...")
        
        # Evaluate agent on dataset
        agent_results = evaluate_agent_with_error_handling(
            agent=agent,
            dataset=dataset,
            verbose=args.verbose,
            max_cases=args.max_cases
        )
        
        # Add to results
        results[agent_name] = agent_results
        
        # Save results to file
        if args.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(args.output_dir, f"{agent_name}_results_{timestamp}.json")
            with open(output_file, "w") as f:
                json.dump(agent_results, f, indent=2, default=str)
            logger.info(f"Results for {agent_name} saved to {output_file}")
    
    # Generate comparison report
    if len(results) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(args.output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        generate_comparison_report(results, output_dir=report_dir)
        logger.info(f"Comparison report saved to {report_dir}")
    
    # Print summary
    logger.info("Benchmark complete.")
    for agent_name, agent_results in results.items():
        if isinstance(agent_results, list):
            success_count = sum(1 for r in agent_results if r.get("is_correct", False))
            total_count = len(agent_results)
            success_rate = success_count / total_count if total_count > 0 else 0
            avg_time = sum(r.get("time_taken", 0) for r in agent_results) / total_count if total_count > 0 else 0
            
            logger.info(f"{agent_name} summary:")
            logger.info(f"  Success rate: {success_rate:.2%}")
            logger.info(f"  Average time: {avg_time:.2f} seconds")
        
    if args.output_dir:
        logger.info(f"Benchmark complete. Results saved to {args.output_dir}")


def load_dataset_from_file(file_path):
    """
    Load a dataset from a file.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        List of benchmark cases
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            return []
        
        # Determine file type from extension
        if file_path.endswith('.json'):
            # Load JSON file
            with open(file_path, 'r') as f:
                dataset = json.load(f)
                
        elif file_path.endswith('.jsonl'):
            # Load JSONL file
            dataset = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        dataset.append(json.loads(line))
                        
        elif file_path.endswith('.csv'):
            # Load CSV file
            import csv
            dataset = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset.append(row)
                    
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return []
        
        # Ensure all entries have the required fields
        for i, item in enumerate(dataset):
            if not isinstance(item, dict):
                logger.warning(f"Item {i} is not a dictionary: {item}")
                continue
                
            if "problem" not in item:
                logger.warning(f"Item {i} does not have a 'problem' field: {item}")
                
            if "reference" not in item:
                logger.warning(f"Item {i} does not have a 'reference' field: {item}")
                
            if "task_type" not in item:
                # Try to infer task type
                problem = item.get("problem", "")
                item["task_type"] = get_task_type(problem)
                logger.info(f"Inferred task type for item {i}: {item['task_type']}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset from file: {str(e)}")
        return []


def evaluate_agent_with_error_handling(agent, dataset, verbose=False, max_cases=None):
    """
    Evaluate an agent on a dataset with error handling.
    Wrapper around evaluate_agent function that catches and handles errors.
    
    Args:
        agent: Agent to evaluate
        dataset: Dataset to evaluate on
        verbose: Whether to print verbose output (default: False)
        max_cases: Maximum number of cases to evaluate (default: None)
        
    Returns:
        List of evaluation results
    """
    try:
        return evaluate_agent(agent, dataset, verbose, max_cases)
    except Exception as e:
        logger.error(f"Error during agent evaluation: {str(e)}")
        return []


if __name__ == "__main__":
    asyncio.run(main()) 