#!/usr/bin/env python3
"""
Test Runner for Math Recognition System

This script runs all tests for the math recognition system with proper
configuration and reporting.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --fast            # Run only fast tests
    python run_tests.py --integration     # Run only integration tests
    python run_tests.py --verbose         # Verbose output
    python run_tests.py --coverage        # Run with coverage report
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def run_pytest_command(args, test_files=None):
    """Run pytest with specified arguments."""
    cmd = ["python", "-m", "pytest"]
    
    # Add test files or default to current directory
    if test_files:
        cmd.extend(test_files)
    else:
        cmd.append(str(Path(__file__).parent))
    
    # Add pytest arguments
    cmd.extend(args)
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the command
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run Math Recognition System Tests")
    
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Run only fast tests (skip slow integration tests)"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true", 
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests (individual components)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    
    parser.add_argument(
        "--html-coverage",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        help="Run tests in parallel with N workers"
    )
    
    parser.add_argument(
        "--component",
        choices=["answer", "error", "root_cause", "gap", "hybrid", "integration"],
        help="Run tests for specific component only"
    )
    
    parser.add_argument(
        "--markers", "-m",
        help="Run tests matching given mark expression"
    )
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    # Verbosity
    if args.verbose:
        pytest_args.extend(["-v", "-s"])
    
    # Coverage
    if args.coverage or args.html_coverage:
        pytest_args.extend([
            "--cov=math_learning.math_recognization",
            "--cov-report=term-missing"
        ])
        
        if args.html_coverage:
            pytest_args.extend(["--cov-report=html:htmlcov"])
    
    # Parallel execution
    if args.parallel:
        pytest_args.extend(["-n", str(args.parallel)])
    
    # Test markers
    if args.markers:
        pytest_args.extend(["-m", args.markers])
    
    # Test selection
    test_files = []
    
    if args.component:
        component_map = {
            "answer": "test_answer_correctness_engine.py",
            "error": "test_error_analysis_engine.py", 
            "root_cause": "test_root_cause_analyzer.py",
            "gap": "test_knowledge_gap_identifier.py",
            "hybrid": "test_hybrid_error_analyzer.py",
            "integration": "test_integration.py"
        }
        test_files = [component_map[args.component]]
    
    elif args.fast:
        # Fast tests - exclude integration and performance tests
        pytest_args.extend(["-m", "not slow"])
        test_files = [
            "test_answer_correctness_engine.py",
            "test_error_analysis_engine.py"
        ]
    
    elif args.integration:
        test_files = ["test_integration.py"]
    
    elif args.unit:
        test_files = [
            "test_answer_correctness_engine.py",
            "test_error_analysis_engine.py",
            "test_hybrid_error_analyzer.py"
        ]
    
    # Add common pytest arguments
    pytest_args.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Reduce noise
    ])
    
    # Print test configuration
    print("🧮 Math Recognition System - Test Runner")
    print("=" * 60)
    print(f"Test Selection: {', '.join(test_files) if test_files else 'All tests'}")
    print(f"Pytest Args: {' '.join(pytest_args)}")
    print()
    
    # Run the tests
    exit_code = run_pytest_command(pytest_args, test_files)
    
    # Print summary
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Exit code: {exit_code}")
    
    # Additional reporting
    if args.coverage and exit_code == 0:
        print("\n📊 Coverage report generated")
    
    if args.html_coverage and exit_code == 0:
        coverage_path = Path(__file__).parent / "htmlcov" / "index.html"
        print(f"📊 HTML coverage report: {coverage_path}")
    
    return exit_code

def run_specific_test_suites():
    """Run specific predefined test suites."""
    
    print("🧮 Math Recognition System - Test Suites")
    print("=" * 60)
    
    suites = {
        "smoke": {
            "description": "Quick smoke tests",
            "files": ["test_answer_correctness_engine.py::TestAnswerCorrectnessEngine::test_numerical_answers"],
            "args": ["-v"]
        },
        "core": {
            "description": "Core functionality tests", 
            "files": [
                "test_answer_correctness_engine.py",
                "test_error_analysis_engine.py"
            ],
            "args": ["-v"]
        },
        "full": {
            "description": "Complete test suite",
            "files": None,  # All files
            "args": ["-v", "--tb=short"]
        }
    }
    
    for suite_name, suite_config in suites.items():
        print(f"\n🔍 Running {suite_name} suite: {suite_config['description']}")
        print("-" * 40)
        
        exit_code = run_pytest_command(
            suite_config["args"], 
            suite_config["files"]
        )
        
        if exit_code != 0:
            print(f"❌ {suite_name} suite failed!")
            return exit_code
        else:
            print(f"✅ {suite_name} suite passed!")
    
    print("\n🎉 All test suites completed successfully!")
    return 0

if __name__ == "__main__":
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Please install it:")
        print("   pip install pytest pytest-asyncio pytest-cov pytest-xdist")
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path(__file__).parent.name == "tests":
        print("❌ Please run this script from the tests directory")
        sys.exit(1)
    
    # Run main test runner
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1) 