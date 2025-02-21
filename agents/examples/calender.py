import os
import requests

from agents.agent import Agent
from llm.llm import LLMBase
from pydantic import Field
from tools.tool import Tool
from typing import Any, Dict, Optional


class CalendlyTool(Tool):
    """Tool for interacting with Calendly API."""
    event_type: str = Field(description="The type of event to schedule")
    invitee_email: str = Field(description="Email of the person being invited")
    invitee_name: str = Field(description="Name of the person being invited")
    start_time: str = Field(description="Proposed start time in ISO format")

    def __init__(self):
        super().__init__()
        self.api_token = os.getenv("CALENDLY_API_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://api.calendly.com/v2"

    def run(self, event_type: str, invitee_email: str, invitee_name: str, start_time: str) -> Dict[str, Any]:
        """Schedule a meeting using Calendly.
        
        Args:
            event_type: Type of event to schedule
            invitee_email: Email of invitee
            invitee_name: Name of invitee
            start_time: Proposed start time
            
        Returns:
            Dict containing scheduling result
        """
        url = f"{self.base_url}/scheduled_events"
        payload = {
            "event_type": event_type,
            "invitee": {
                "email": invitee_email,
                "name": invitee_name
            },
            "start_time": start_time
        }

        response = requests.post(url, headers=self.headers, json=payload)
        return response.json()

    async def arun(self, event_type: str, invitee_email: str, invitee_name: str, start_time: str) -> Dict[str, Any]:
        """Async version of run method."""
        return self.run(
            event_type=event_type,
            invitee_email=invitee_email, 
            invitee_name=invitee_name,
            start_time=start_time
        )


class CalendarAgent(Agent):
    """Agent that handles calendar scheduling through Calendly."""

    def __init__(self, llm: LLMBase):
        system_prompt = """You are an AI assistant that helps schedule meetings using Calendly.
        When given scheduling requests, you will use the CalendlyTool to create calendar events.
        
        You understand how to:
        - Parse natural language date/time requests
        - Convert times to ISO format
        - Handle scheduling conflicts
        - Validate email addresses
        - Format names appropriately
        
        Help users schedule meetings efficiently while ensuring all required information is collected."""
        
        tools = [CalendlyTool()]
        super().__init__(llm, system_prompt, tools)


def main():
    from llm.openai.openai_llm import OpenAILLM
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = OpenAILLM.create_llm()
    agent = CalendarAgent(llm)
    
    request = """Schedule a 30 minute meeting with John Smith (john.smith@email.com) 
                 tomorrow at 2pm for a project review."""
    
    response = agent.process(request, model="gpt-3.5-turbo")
    print("Scheduling Result:")
    print(response)


if __name__ == "__main__":
    main()
