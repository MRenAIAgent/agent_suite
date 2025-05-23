#!/usr/bin/env python3
"""
Script to run all tests for the memory management system.

Usage:
    python run_tests.py                # Run all tests
    python run_tests.py unit           # Run only unit tests
    python run_tests.py integration    # Run only integration tests
"""

import sys
import os
import subprocess
import pytest

def setup():
    """Ensure we're in the right directory and dependencies are installed."""
    # Change to the tests directory if we're not already there
    if not os.path.basename(os.getcwd()) == 'tests':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
    
    # Add agent_suite directory to Python path
    agent_suite_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../agent_suite"))
    if agent_suite_dir not in sys.path:
        sys.path.insert(0, agent_suite_dir)
    
    # Install dependencies if needed
    try:
        import pytest
    except ImportError:
        print("Installing test dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_tests(test_type=None):
    """Run the specified tests."""
    setup()
    
    if test_type == 'unit':
        print("Running unit tests...")
        return pytest.main(["-xvs", "unit/"])
    elif test_type == 'integration':
        print("Running integration tests...")
        return pytest.main(["-xvs", "integration/"])
    else:
        print("Running all tests...")
        return pytest.main(["-xvs", "unit/", "integration/"])

if __name__ == "__main__":
    test_type = sys.argv[1] if len(sys.argv) > 1 else None
    
    if test_type not in (None, 'unit', 'integration'):
        print(f"Unknown test type: {test_type}")
        print("Usage: python run_tests.py [unit|integration]")
        sys.exit(1)
    
    sys.exit(run_tests(test_type)) 