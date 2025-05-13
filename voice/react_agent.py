from typing import List
from langchain_community.agents import AgentExecutor, create_react_agent
from langchain_community.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.tools.base import BaseTool
from langchain_community.schema import AgentAction, AgentFinish
from langchain_core.schema.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field
from serpapi import GoogleSearch

import os
from dotenv import load_dotenv

load_dotenv()

class SearchTool(BaseTool):
    """Tool that adds the capability to search using SerpAPI."""
    name: str = "search"
    description: str = "useful for when you need to search the internet to answer questions"
    return_direct: bool = False

    def _run(self, query: str) -> str:
        """Use the tool."""
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY")
        })
        results = search.get_dict()
        if "answer_box" in results:
            return results["answer_box"]["answer"]
        elif "organic_results" in results and len(results["organic_results"]) > 0:
            return results["organic_results"][0]["snippet"]
        return "No good search result found"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchTool does not support async")

class ConversationalAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.tools = [SearchTool()]
        
        prompt = """You are a friendly conversational AI assistant that can search the internet and maintain context.
        You engage naturally in back-and-forth dialogue while providing accurate information.
        Use the search tool when you need to look up facts or current information.
        
        Keep track of the conversation history and refer back to previous context when relevant.
        Be concise but personable in your responses."""
        
        self.agent = create_conversational_retrieval_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
            verbose=True
        )

    def run(self, query: str, chat_history: List = None) -> str:
        """
        Run the conversational agent on a query
        
        Args:
            query (str): The user's question or request
            chat_history (List): Optional list of previous conversation messages
            
        Returns:
            str: The agent's response
        """
        try:
            if chat_history is None:
                chat_history = []
                
            response = self.agent.invoke({
                "input": query,
                "chat_history": chat_history
            })
            return response["output"]
        except Exception as e:
            return f"Error running conversational agent: {str(e)}"

    async def arun(self, query: str, chat_history: List = None) -> str:
        """
        Run the conversational agent asynchronously
        
        Args:
            query (str): The user's question or request
            chat_history (List): Optional list of previous conversation messages
            
        Returns:
            str: The agent's response
        """
        try:
            if chat_history is None:
                chat_history = []
                
            response = await self.agent.ainvoke({
                "input": query, 
                "chat_history": chat_history
            })
            return response["output"]
        except Exception as e:
            return f"Error running conversational agent: {str(e)}"


class ReactAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.tools = [SearchTool()]
        

        template = '''Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}'''

        prompt = PromptTemplate.from_template(template)
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, query: str) -> str:
        """
        Run the agent on a query
        
        Args:
            query (str): The user's question or request
            
        Returns:
            str: The agent's response
        """
        try:
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            return f"Error running agent: {str(e)}"

    async def arun(self, query: str) -> str:
        """
        Run the agent asynchronously
        
        Args:
            query (str): The user's question or request
            
        Returns:
            str: The agent's response
        """
        try:
            response = await self.agent_executor.ainvoke({"input": query})
            return response["output"]
        except Exception as e:
            return f"Error running agent: {str(e)}"


def main():
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize the agent
    # agent = ReactAgent()
    conversational_agent = ConversationalAgent()

    # Test cases
    test_queries = [
        "What is the current weather in San Francisco?",
        "Who won the most recent Super Bowl?", 
        "What are the top news headlines today?",
        "When was the last SpaceX launch?"
    ]

    print("Running test queries...")
    print("-" * 50)

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = conversational_agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    main()

