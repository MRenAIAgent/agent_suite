# pylint: disable=too-few-public-methods
from litellm import completion
from dotenv import load_dotenv

from llm.llm import LLMBase


class LiteLLM(LLMBase):
    def __init__(self):
        pass
        
    @classmethod
    def create_llm(cls):
        """Create an instance of the LiteLLM.
        
        Returns:
            LiteLLM: A new instance of the LiteLLM
        """
        return cls()

    def chat_completion(self, model: str, messages: list, tools: list = None, stream: bool = False):
        """Create a chat completion using the LLM.
        
        Args:
            messages: List of message dictionaries
            stream: Boolean for streaming
        """
        response = completion(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
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
    llm = LiteLLM.create_llm()
    response = llm.chat_completion(model="openai/gpt-4o-mini", messages=messages)
    print(response)

if __name__ == "__main__":
    load_dotenv()
    main()
