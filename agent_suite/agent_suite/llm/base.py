#!/usr/bin/env python3
"""
Base LLM Class

This module defines the base interfaces for LLM (Large Language Model) interactions.
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class LLM(ABC):
    """Base abstract class for LLM interactions."""
    
    @abstractmethod
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: Optional specific model to use
            **kwargs: Additional model parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Generate a response from a chat conversation.
        
        Args:
            messages: List of message dictionaries
            model: Optional specific model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model parameters
            
        Returns:
            Response from the model
        """
        pass
    
    @classmethod
    def create_llm(cls, provider: str = "openai", api_key: Optional[str] = None, **kwargs) -> "LLM":
        """Factory method to create an LLM instance.
        
        Args:
            provider: LLM provider name
            api_key: Optional API key for the provider
            **kwargs: Additional parameters for LLM initialization
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider.lower() == "openai":
            from agent_suite.llm.openai.client import OpenAILLM
            return OpenAILLM(api_key=api_key, **kwargs)
        elif provider.lower() == "anthropic":
            from agent_suite.llm.anthropic.client import AnthropicLLM
            return AnthropicLLM(api_key=api_key, **kwargs)
        elif provider.lower() == "litellm":
            from agent_suite.llm.litellm.litellm import LiteLLM
            return LiteLLM(api_key=api_key, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
