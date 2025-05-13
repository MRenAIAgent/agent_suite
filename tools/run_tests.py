#!/usr/bin/env python3
"""
Test runner for the enhanced tool system.

This script runs all the tests for the enhanced tool system and reports the results.
"""
import os
import sys
import unittest
import importlib
import asyncio

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def import_test_modules():
    """Import all test modules in the tools/tests directory."""
    tests_dir = os.path.join(os.path.dirname(__file__), "tests")
    test_modules = []

    # Check if tests directory exists
    if not os.path.exists(tests_dir):
        print(f"Warning: Tests directory not found: {tests_dir}")
        return test_modules

    # Find all test_*.py files
    for filename in os.listdir(tests_dir):
        if filename.startswith("test_") and filename.endswith(".py"):
            module_name = f"tools.tests.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                test_modules.append(module)
                print(f"Imported {module_name}")
            except ImportError as e:
                print(f"Error importing {module_name}: {e}")

    # Include the original test file
    try:
        test_enhanced_tools = importlib.import_module("tools.test_enhanced_tools")
        test_modules.append(test_enhanced_tools)
        print(f"Imported tools.test_enhanced_tools")
    except ImportError as e:
        print(f"Error importing tools.test_enhanced_tools: {e}")

    return test_modules


def run_tests():
    """Run all tests for the enhanced tool module."""
    print("=== Running Enhanced Tool System Tests ===\n")
    
    # Import test modules
    import_test_modules()
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases from modules in tools/tests
    tests_dir = os.path.join(os.path.dirname(__file__), "tests")
    if os.path.exists(tests_dir):
        test_suite.addTests(test_loader.discover(tests_dir, pattern="test_*.py"))
    
    # Add test cases from test_enhanced_tools.py
    test_enhanced_tools_file = os.path.join(os.path.dirname(__file__), "test_enhanced_tools.py")
    if os.path.exists(test_enhanced_tools_file):
        # Get the module name
        module_name = "tools.test_enhanced_tools"
        
        # Import it if not already imported
        try:
            module = importlib.import_module(module_name)
            
            # Find all test functions in the module
            for name in dir(module):
                if name.startswith("test_"):
                    test_function = getattr(module, name)
                    if callable(test_function):
                        test_class = type("DynamicTestCase", (unittest.TestCase,), {name: test_function})
                        test_suite.addTest(test_loader.loadTestsFromTestCase(test_class))
                        print(f"Added test function: {name}")
        except ImportError as e:
            print(f"Error importing {module_name}: {e}")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests()) 