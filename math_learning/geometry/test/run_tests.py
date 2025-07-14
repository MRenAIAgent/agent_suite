#!/usr/bin/env python3
"""
Test runner for the Geometry Learning System test suite.

This script provides various options for running the geometry learning system tests,
including unit tests, integration tests, and performance tests.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Geometry Learning System tests")
    
    # Test selection arguments
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run only integration tests"
    )
    parser.add_argument(
        "--slow", 
        action="store_true", 
        help="Include slow running tests"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests including slow ones"
    )
    
    # Test file selection
    parser.add_argument(
        "--file", 
        type=str, 
        help="Run tests from specific file (e.g., test_geometry_knowledge_graph.py)"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        help="Run specific test function"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        help="Quiet output"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--html-report", 
        action="store_true", 
        help="Generate HTML coverage report"
    )
    
    # Performance options
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--profile", 
        action="store_true", 
        help="Profile test execution"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))
    
    # Test selection
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.all:
        cmd.extend(["-m", "unit or integration or slow"])
    elif not args.slow:
        cmd.extend(["-m", "not slow"])
    
    # Specific file or test
    if args.file:
        cmd.append(str(test_dir / args.file))
    
    if args.test:
        cmd.extend(["-k", args.test])
    
    # Output options
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    
    # Coverage
    if args.coverage:
        cmd.extend(["--cov=../src", "--cov-report=term-missing"])
        if args.html_report:
            cmd.extend(["--cov-report=html"])
    
    # Performance options
    if args.benchmark:
        cmd.extend(["--benchmark-only"])
    
    if args.profile:
        cmd.extend(["--profile"])
    
    # Add common options
    cmd.extend([
        "--tb=short",
        "--durations=10"
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run tests
    try:
        result = subprocess.run(cmd, cwd=test_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_quick_tests():
    """Run a quick subset of tests for development."""
    print("Running quick test suite...")
    cmd = [
        "python", "-m", "pytest",
        "-m", "unit and not slow",
        "-x",  # Stop on first failure
        "--tb=short"
    ]
    
    test_dir = Path(__file__).parent
    subprocess.run(cmd, cwd=test_dir)


def run_comprehensive_tests():
    """Run comprehensive test suite including integration and slow tests."""
    print("Running comprehensive test suite...")
    cmd = [
        "python", "-m", "pytest",
        "-m", "unit or integration or slow",
        "--tb=short",
        "--durations=20"
    ]
    
    test_dir = Path(__file__).parent
    subprocess.run(cmd, cwd=test_dir)


def generate_test_report():
    """Generate a comprehensive test report."""
    print("Generating test report...")
    test_dir = Path(__file__).parent
    
    # Run tests with coverage and HTML report
    cmd = [
        "python", "-m", "pytest",
        "--cov=../src",
        "--cov-report=html",
        "--cov-report=term",
        "--junit-xml=test_results.xml",
        "--tb=short"
    ]
    
    subprocess.run(cmd, cwd=test_dir)
    
    print("\nTest report generated:")
    print(f"- HTML coverage report: {test_dir}/htmlcov/index.html")
    print(f"- JUnit XML report: {test_dir}/test_results.xml")


if __name__ == "__main__":
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("Error: pytest is not installed. Please install it with:")
        print("pip install pytest pytest-cov")
        sys.exit(1)
    
    # Add the geometry src directory to Python path
    src_dir = Path(__file__).parent.parent / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
    
    # Add the math_learning directory to Python path
    math_learning_dir = Path(__file__).parent.parent.parent
    if math_learning_dir.exists():
        sys.path.insert(0, str(math_learning_dir))
    
    # If no arguments provided, show help and run quick tests
    if len(sys.argv) == 1:
        print("Geometry Learning System Test Runner")
        print("=" * 40)
        print()
        print("Usage examples:")
        print("  python run_tests.py --unit          # Run unit tests only")
        print("  python run_tests.py --integration   # Run integration tests only")
        print("  python run_tests.py --all           # Run all tests")
        print("  python run_tests.py --coverage      # Run with coverage")
        print("  python run_tests.py --file test_geometry_knowledge_graph.py")
        print()
        print("Running quick test suite by default...")
        print()
        run_quick_tests()
    else:
        sys.exit(main()) 