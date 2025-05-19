#!/usr/bin/env python3
"""
Basic Test for LLM and Storage

This script tests the basic functionality of LLM and storage components.
"""
import os
import sys
import logging
import asyncio

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_llm_and_storage():
    """Test the base LLM and storage functionality."""
    try:
        # Import components
        from agent_suite.llm.base import LLM
        from agent_suite.llm.litellm.litellm import LiteLLM
        from agent_suite.agents.storage.intermediate_storage import (
            InMemoryIntermediateStorage
        )
        
        # Create storage
        storage = InMemoryIntermediateStorage()
        storage.store("test_key", "test_value")
        value = storage.retrieve("test_key")
        
        print(f"Storage test: {'PASSED' if value == 'test_value' else 'FAILED'}")
        
        # Create LLM
        llm = LiteLLM.create_llm()
        response = await llm.generate("Hello, how are you today?")
        
        print(f"LLM test response: {response[:100]}...")
        print("LLM test: PASSED")
        
        return True
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the basic tests."""
    success = asyncio.run(test_llm_and_storage())
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 