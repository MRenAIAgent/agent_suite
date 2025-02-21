# pylint: disable=too-few-public-methods
from openai import OpenAI
from dotenv import load_dotenv
import os

from llm.llm import LLMBase

class OpenAILLM(LLMBase):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
    @classmethod
    def create_llm(cls):
        """Create an instance of the OpenAI LLM.
        
        Returns:
            OpenAILLM: A new instance of the OpenAI LLM
        """
        api_key = os.getenv("OPENAI_API_KEY")
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
    llm = OpenAILLM.create_llm()
    response = llm.chat_completion(model="gpt-3.5-turbo", messages=messages)
    print(response)

if __name__ == "__main__":
    load_dotenv()
    main()
