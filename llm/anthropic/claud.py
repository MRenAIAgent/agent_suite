# pylint: disable=too-few-public-methods
from openai import OpenAI
from dotenv import load_dotenv
import os

from llm.llm import LLMBase

class ClaudLLM(LLMBase):
    def __init__(self, api_key):
        self.base_url = "https://api.anthropic.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        
    @classmethod
    def create_llm(cls):
        """Create an instance of the Claude LLM.
        
        Returns:
            ClaudLLM: A new instance of the Claude LLM
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return cls(api_key=api_key)

    def chat_completion(self, model: str, messages: list, tools: list = None, stream: bool = False):
        """Create a chat completion using the LLM.
        
        Args:
            messages: List of message dictionaries
            stream: Boolean for streaming
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream
        )
        return response.choices[0].message.content

    def get_client(self):
        """Returns the OpenAI client instance."""
        return self.client

def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    llm = ClaudLLM.create_llm()
    response = llm.chat_completion(model="claude-3-opus-20240229", messages=messages)
    print(response)

if __name__ == "__main__":
    load_dotenv()
    main()
