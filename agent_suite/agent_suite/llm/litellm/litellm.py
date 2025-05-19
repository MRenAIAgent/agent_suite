#!/usr/bin/env python3
"""
LiteLLM Wrapper

This module provides a wrapper for the LiteLLM library,
which enables using various LLM providers through a unified interface.
"""
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

from ...llm.base import LLM

logger = logging.getLogger(__name__)

# Try to import litellm, but provide mock if not available
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    logger.warning("LiteLLM package not installed. Using mock implementation.")
    HAS_LITELLM = False


class LiteLLM(LLM):
    """LiteLLM wrapper for unified LLM interactions."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "anthropic/claude-3-7-sonnet-20250219",
        **kwargs
    ):
        """Initialize the LiteLLM wrapper.
        
        Args:
            api_key: Optional API key (defaults to env)
            default_model: Default model to use
            **kwargs: Additional parameters
        """
        self.default_model = default_model
        self.api_key = api_key
        self.kwargs = kwargs
        
        # Set up the API key if provided
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key
    
    @classmethod
    def create_llm(cls, api_key: Optional[str] = None, **kwargs) -> "LiteLLM":
        """Create a new LiteLLM instance.
        
        Args:
            api_key: Optional API key for the LLM
            **kwargs: Additional parameters
            
        Returns:
            LiteLLM instance
        """
        return cls(api_key=api_key, **kwargs)
    
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: Optional model to use (overrides default)
            **kwargs: Additional model parameters
            
        Returns:
            Generated text
        """
        # Use provided model or default
        model_name = model or self.default_model
        
        # Handle mock case
        if not HAS_LITELLM:
            logger.warning(f"Using mock LiteLLM for model {model_name}")
            return f"This is a mock response from {model_name} because LiteLLM is not installed."
        
        # Merge provided kwargs with defaults
        params = {**self.kwargs, **kwargs}
        
        try:
            response = await litellm.acompletion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            
            # Extract the content from the response
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            
            # Fallback if structure is different
            return str(response)
            
        except Exception as e:
            logger.error(f"Error generating with LiteLLM: {e}")
            # Return error message in production, but consider raising in development
            return f"Error generating response: {str(e)}"
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Generate a response from a chat conversation.
        
        Args:
            messages: List of message dictionaries
            model: Optional model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools
            tool_choice: Optional tool choice
            **kwargs: Additional model parameters
            
        Returns:
            Response from the model
        """
        # Use provided model or default
        model_name = model or self.default_model
        
        # Handle mock case
        if not HAS_LITELLM:
            logger.warning(f"Using mock LiteLLM for model {model_name}")
            mock_response = {
                "id": "mock-completion-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"This is a mock response from {model_name} because LiteLLM is not installed."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            # If tools were provided, mock a tool call
            if tools and len(tools) > 0:
                tool_name = tools[0].get("function", {}).get("name", "mock_tool")
                mock_response["choices"][0]["message"] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "mock-tool-call-id",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": "{}"
                            }
                        }
                    ]
                }
            
            # Create a mock response object with structure similar to real response
            class MockResponse:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
                    self.choices = [MockChoice(c) for c in data.get("choices", [])]
                    
            class MockChoice:
                def __init__(self, data):
                    for key, value in data.items():
                        if key == "message":
                            self.message = MockMessage(value)
                        else:
                            setattr(self, key, value)
                            
            class MockMessage:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
                        
            return MockResponse(mock_response)
        
        # Set up parameters
        params = {
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        if tools:
            params["tools"] = tools
            
        if tool_choice:
            params["tool_choice"] = tool_choice
        
        try:
            # Call LiteLLM completion
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                **params
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            # In production, return mock response on error
            return self._create_error_response(str(e), model_name)
    
    def _create_error_response(self, error_message: str, model_name: str) -> Any:
        """Create an error response object with similar structure to real responses.
        
        Args:
            error_message: The error message
            model_name: The model name
            
        Returns:
            Error response object
        """
        error_data = {
            "id": "error-completion-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Error: {error_message}"
                    },
                    "finish_reason": "error"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        class ErrorResponse:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
                self.choices = [ErrorChoice(c) for c in data.get("choices", [])]
                    
        class ErrorChoice:
            def __init__(self, data):
                for key, value in data.items():
                    if key == "message":
                        self.message = ErrorMessage(value)
                    else:
                        setattr(self, key, value)
                            
        class ErrorMessage:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
                        
        return ErrorResponse(error_data)

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
