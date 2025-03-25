import asyncio
import unittest
from parameterized import parameterized
from pydantic import Field
from llm.litellm.litellm import LiteLLM
from tools.tool import Tool
from tools.animal_type import AnimalType
from tools.serper_api import SerperSearchTool
from agents.react_agent import ReActAgent
from agents.react_agent_fc.react_agent_fc import ReactAgentFC


class TestTool(Tool):
    """Simple test tool for testing ReactAgent."""
    query: str = Field(description="The query to process")
    
    def run(self, query: str):
        return f"Result for: {query}"
    
    async def arun(self, query: str):
        return self.run(query=query)


# Define test parameters separately
TEST_INPUTS = [
    # "What is smallest positive integer plus 1?",
    # "How many world champions title does 2022 world cup holder have totally?"
    "give me a concrete plan to visit wild flowers in southen california in April 1 in 2025"
]

AGENT_PATTERNS = [
    "standard",  # Regular ReactAgent
    # "fc"         # ReactAgentFC with function calling
]

MODELS = [
    "openai/gpt-4o-mini"
]


class TestReactAgent(unittest.TestCase):
    """Test cases for ReactAgent class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.llm = LiteLLM.create_llm()
        self.system_prompt = "You are a helpful assistant."
        self.test_tool = AnimalType(animal_name="dog")
        self.serper_tool = SerperSearchTool(query="")
 
        self.agent = ReActAgent(
            llm=self.llm,
            role="You are kindergardener, like to be a story teller.",
            task="Tell a story for kindergardener kids",
            guide="""Story should be short and simple, and easy to understand. 
                appropreiate for kindergardener kids""",
            examples=[],
            tools=[self.serper_tool],
            max_iterations=5
        )
        self.agent_fc = ReactAgentFC(
            llm=self.llm,
            role="You are kindergardener, like to be a story teller.",
            task="Tell a story for kindergardener kids",
            guide="""Story should be short and simple, and easy to understand. 
                appropreiate for kindergardener kids""",
            tools=[self.serper_tool],
            max_iterations=3
        )
        
    def test_initialization(self):
        """Test that the agent initializes correctly."""
        # ReactAgent doesn't expose system_prompt directly, it's in prompt_manager
        self.assertEqual(len(self.agent.tools), 1)
        self.assertEqual(self.agent.max_iterations, 5)
        self.assertIsNotNone(self.agent.agent_pattern)
        self.assertIsNotNone(self.agent.execution_pattern)

    @parameterized.expand([
        (input_text, agent_pattern, model)
        for input_text in TEST_INPUTS
        for agent_pattern in AGENT_PATTERNS
        for model in MODELS
    ])
    def test_agent(self, input_text, agent_pattern, model):
        """Test agent with various inputs and configurations."""
        # Determine if this is an async test based on the input
        is_async = input_text != "Tell me something interesting" and input_text != "what type of animal is cat?"
        
        # Create a description for the test
        description = f"Testing {agent_pattern} agent with model {model} on input: {input_text[:30]}..."
        
        # Print test description
        print(f"\nRunning test: {description}")
        
        # Get the appropriate agent
        agent = self.agent_fc if agent_pattern == "fc" else self.agent
        
        # Run the test
        if is_async:
            result = asyncio.run(agent.arun(input_text, model))
        else:
            result = agent.run(input_text, model)
        
        # Print result for debugging
        print(f"Result:\n{result}")
        
        # Verify result
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    # Fix for the "don't know how to make test from: None" error
    # Pass an empty list as argv to prevent unittest from trying to parse sys.argv
    unittest.main(argv=['first-arg-is-ignored'])
