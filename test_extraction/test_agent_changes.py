#!/usr/bin/env python3
"""
Test script for validating changes to the Agent class using FieldsExtractionService.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to sys.path to ensure imports work
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from dotenv import load_dotenv
from llm.openai.openai_llm import OpenAILLM
from agents.examples.fields_extraction.fields_extraction_agent import FieldsExtractionService
from log.logging import LogManager

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 50}")
    print(f" {text}")
    print(f"{'=' * 50}{RESET}")

def print_success(text):
    """Print a success message."""
    print(f"{GREEN}{text}{RESET}")

def print_error(text):
    """Print an error message."""
    print(f"{RED}{text}{RESET}")

def print_warning(text):
    """Print a warning message."""
    print(f"{YELLOW}{text}{RESET}")

def print_json(data):
    """Print formatted JSON."""
    print(json.dumps(data, indent=2))

class TestLogManager(LogManager):
    """Custom log manager that captures logs for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.logs = []
        self.debug_enabled = True
    
    def log_debug(self, message):
        """Log a debug message."""
        self.logs.append(f"DEBUG: {message}")
        # Uncomment to see debug logs in real time
        # print(f"{BLUE}DEBUG: {message}{RESET}")
    
    def log_interaction(self, user_input, agent_response, model, timestamp, metadata=None):
        """Log an interaction."""
        interaction = {
            "user_input": user_input,
            "agent_response": agent_response,
            "model": model,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        self.logs.append(f"INTERACTION: {json.dumps(interaction)}")
    
    def print_logs(self):
        """Print all logs."""
        for log in self.logs:
            if log.startswith("DEBUG:"):
                print(f"{BLUE}{log}{RESET}")
            else:
                print(f"{YELLOW}{log}{RESET}")
    
    def print_logs_in_pretty_format(self):
        """Print logs in a pretty format."""
        print(f"{BLUE}=== Logs in Pretty Format ==={RESET}")
        for log in self.logs:
            if log.startswith("DEBUG:"):
                print(f"{BLUE}{log[7:]}{RESET}")
            elif log.startswith("INTERACTION:"):
                interaction = json.loads(log[13:])
                print(f"{YELLOW}User: {interaction['user_input']}{RESET}")
                print(f"{GREEN}Agent: {interaction['agent_response']}{RESET}")

async def test_text_extraction(extraction_service, log_manager):
    """Test extraction from text."""
    print_header("Testing Extraction from Text")
    
    # Sample text
    text = """
    DRIVER LICENSE
    NAME: DOE, JOHN MICHAEL
    DOB: 01/01/1980
    EXP: 01/01/2025
    LICENSE #: D12345678
    GENDER: M
    STATE: CA
    """
    
    try:
        print("Processing text extraction...")
        result = await extraction_service.extract_from_text(text, model="gpt-4")
        print_success("Text extraction successful!")
        print("Result:")
        print_json(result)
        
        # Print logs to debug
        print_header("Agent Logs")
        log_manager.print_logs_in_pretty_format()
        
        # Clear logs for next test
        log_manager.logs = []
        
        return True
    except Exception as e:
        print_error(f"Text extraction failed: {str(e)}")
        return False

async def test_file_extraction(extraction_service, log_manager):
    """Test extraction from file."""
    print_header("Testing Extraction from File")
    
    file_path = os.path.join(os.path.dirname(__file__), "license.txt")
    if not os.path.exists(file_path):
        print_error(f"Test file not found: {file_path}")
        return False
    
    try:
        print(f"Processing file extraction from {file_path}...")
        result = await extraction_service.extract_from_file(file_path, model="gpt-4")
        print_success("File extraction successful!")
        print("Result:")
        print_json(result)
        
        # Print logs to debug
        print_header("Agent Logs")
        log_manager.print_logs_in_pretty_format()
        
        # Clear logs for next test
        log_manager.logs = []
        
        return True
    except Exception as e:
        print_error(f"File extraction failed: {str(e)}")
        return False

async def test_specific_fields(extraction_service, log_manager):
    """Test extraction of specific fields."""
    print_header("Testing Extraction of Specific Fields")
    
    text = """
    DRIVER LICENSE
    NAME: DOE, JOHN MICHAEL
    DOB: 01/01/1980
    EXP: 01/01/2025
    LICENSE #: D12345678
    GENDER: M
    STATE: CA
    """
    
    fields = ["license_number", "first_name", "last_name"]
    
    try:
        print(f"Extracting specific fields: {fields}")
        result = await extraction_service.extract_from_text(text, fields=fields, model="gpt-4")
        print_success("Specific fields extraction successful!")
        print("Result:")
        print_json(result)
        
        # Print logs to debug
        print_header("Agent Logs")
        log_manager.print_logs_in_pretty_format()
        
        return True
    except Exception as e:
        print_error(f"Specific fields extraction failed: {str(e)}")
        return False

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    print_header("Agent Changes Validation Test")
    
    # Create the log manager
    log_manager = TestLogManager()
    
    # Create the extraction service
    try:
        llm = OpenAILLM.create_llm()
        extraction_service = FieldsExtractionService(llm)
        
        # Set the log manager for the agent
        extraction_service.agent.log_manager = log_manager
        
        print_success("FieldsExtractionService created successfully!")
    except Exception as e:
        print_error(f"Failed to create FieldsExtractionService: {str(e)}")
        return 1
    
    # Run the tests
    results = []
    results.append(await test_text_extraction(extraction_service, log_manager))
    results.append(await test_file_extraction(extraction_service, log_manager))
    results.append(await test_specific_fields(extraction_service, log_manager))
    
    # Print summary
    print_header("Test Summary")
    successes = sum(results)
    total = len(results)
    
    if successes == total:
        print_success(f"All tests passed! ({successes}/{total})")
        print_success("Agent class changes validated successfully!")
    else:
        print_error(f"Some tests failed. ({successes}/{total} passed)")
        print_warning("Agent class changes may have issues that need to be addressed.")
    
    return 0 if successes == total else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 