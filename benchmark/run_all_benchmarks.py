#!/usr/bin/env python3
"""
Run All Benchmarks

This script runs all available benchmarks in the benchmark directory
to verify that they are working correctly.
"""

import os
import sys
import argparse
import logging
import importlib.util
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run_test_script(script_path: str, args: Optional[List[str]] = None) -> bool:
    """
    Run a test script and check if it succeeds.
    
    Args:
        script_path: Path to the script to run
        args: List of arguments to pass to the script
        
    Returns:
        True if the script ran successfully, False otherwise
    """
    if args is None:
        args = []  # Default to no arguments
        
    logger.info(f"Running test script: {script_path} {' '.join(args)}")
    
    try:
        # Save the original sys.argv
        original_argv = sys.argv.copy()
        
        # Set up new sys.argv for the script
        sys.argv = [script_path] + args
        
        # Load the script as a module
        spec = importlib.util.spec_from_file_location("test_module", script_path)
        if spec is None or spec.loader is None:
            logger.error(f"Failed to load script: {script_path}")
            return False
            
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_module"] = module
        spec.loader.exec_module(module)
        
        # If the script has a main function, call it
        if hasattr(module, "main"):
            main_func = module.main
            
            # Check if it's a coroutine function
            if asyncio.iscoroutinefunction(main_func):
                # If it's a coroutine, await it
                await main_func()
            else:
                # If it's a regular function, just call it
                main_func()
        
        # Restore the original sys.argv
        sys.argv = original_argv
        
        logger.info(f"Successfully ran: {script_path}")
        return True
    except Exception as e:
        logger.error(f"Error running {script_path}: {e}")
        return False
    finally:
        # Ensure we restore sys.argv even if there's an exception
        sys.argv = original_argv
        
        # Remove the module from sys.modules to avoid conflicts
        if "test_module" in sys.modules:
            del sys.modules["test_module"]


def find_benchmark_scripts() -> Dict[str, List[str]]:
    """
    Find all benchmark scripts in the benchmark directory.
    
    Returns:
        Dictionary mapping benchmark types to lists of script paths
    """
    benchmark_dir = Path(__file__).parent
    benchmarks = {}
    
    # Look for subdirectories
    for subdir in benchmark_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("__"):
            scripts = []
            
            # Look for Python files in the code subdirectory
            code_dir = subdir / "code"
            if code_dir.exists():
                for script in code_dir.glob("*.py"):
                    if script.name != "__init__.py" and "benchmark" in script.name.lower():
                        scripts.append(str(script))
            
            if scripts:
                benchmarks[subdir.name] = scripts
    
    return benchmarks


def get_script_args(script_path: str, dry_run: bool) -> List[str]:
    """
    Get appropriate arguments for a script based on its content.
    
    Args:
        script_path: Path to the script
        dry_run: Whether to run in dry run mode
        
    Returns:
        List of arguments for the script
    """
    if not dry_run:
        return []
        
    # Read the script to see if it supports --dry-run
    try:
        with open(script_path, 'r') as f:
            content = f.read()
            
        if "--dry-run" in content or "dry_run" in content:
            return ["--dry-run"]
        elif "--help" in content:
            return ["--help"]
        else:
            # No dry run option, use other safe options
            if "rag_top_benchmark.py" in script_path:
                return ["--output-dir", "/tmp/rag_results", "--limit", "1"]
            else:
                return ["--help"]
    except Exception as e:
        logger.warning(f"Error checking script arguments for {script_path}: {e}")
        return ["--help"]


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--include", nargs="+", help="Benchmarks to include (e.g., rag agent)")
    parser.add_argument("--exclude", nargs="+", help="Benchmarks to exclude")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually run the benchmarks")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Adjust logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find all benchmark scripts
    benchmarks = find_benchmark_scripts()
    logger.info(f"Found {len(benchmarks)} benchmark types:")
    for benchmark_type, scripts in benchmarks.items():
        logger.info(f"  {benchmark_type}: {len(scripts)} scripts")
        for script in scripts:
            logger.info(f"    - {os.path.basename(script)}")
    
    # Filter benchmarks based on include/exclude args
    if args.include:
        include_set = set(args.include)
        benchmarks = {k: v for k, v in benchmarks.items() if k in include_set}
    
    if args.exclude:
        exclude_set = set(args.exclude)
        benchmarks = {k: v for k, v in benchmarks.items() if k not in exclude_set}
    
    # Run all benchmark scripts
    total_scripts = sum(len(scripts) for scripts in benchmarks.values())
    successful_scripts = 0
    failed_scripts = 0
    
    for benchmark_type, scripts in benchmarks.items():
        logger.info(f"Running {benchmark_type} benchmarks...")
        
        for script in scripts:
            script_args = get_script_args(script, args.dry_run)
            success = await run_test_script(script, script_args)
            
            if success:
                successful_scripts += 1
            else:
                failed_scripts += 1
    
    # Print summary
    logger.info(f"Benchmark Run Summary:")
    logger.info(f"  Total scripts: {total_scripts}")
    logger.info(f"  Successful: {successful_scripts}")
    logger.info(f"  Failed: {failed_scripts}")
    
    if failed_scripts > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 