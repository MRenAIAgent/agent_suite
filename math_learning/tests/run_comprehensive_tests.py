#!/usr/bin/env python3
"""
Comprehensive Test Runner for Math Learning System

This script runs all comprehensive test suites and provides detailed reporting
on test results, performance metrics, and system coverage.

Test Suites:
- Storage Backend Tests (memory, real databases, integration)
- Graph Operations Tests (construction, queries, optimization)
- Memory Management Tests (usage, leaks, cleanup)
- Performance Tests (load, stress, scalability)
- Existing Core Tests (learning, gap analysis, scenarios)
"""

import sys
import os
import time
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


class ComprehensiveTestRunner:
    """Comprehensive test runner with detailed reporting."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.test_suites = {
            "storage": {
                "name": "Storage Backend Tests",
                "path": "math_learning/tests/storage/test_simple_storage.py",
                "description": "Memory storage, data structures, integration scenarios",
                "timeout": 300  # 5 minutes
            },
            "graph": {
                "name": "Graph Operations Tests", 
                "path": "math_learning/tests/graph/test_simple_graph_operations.py",
                "description": "Knowledge graph construction, queries, optimization",
                "timeout": 180  # 3 minutes
            },
            "memory": {
                "name": "Memory Management Tests",
                "path": "math_learning/tests/memory/test_simple_memory_management.py", 
                "description": "Memory usage, leak detection, cleanup strategies",
                "timeout": 240  # 4 minutes
            },
            "performance": {
                "name": "Performance & Scalability Tests",
                "path": "math_learning/tests/performance/test_simple_performance.py",
                "description": "Load testing, stress testing, scalability analysis",
                "timeout": 360  # 6 minutes
            },
            "core": {
                "name": "Core Functionality Tests",
                "path": "math_learning/tests/test_*.py",
                "description": "Learning graphs, gap analysis, user scenarios",
                "timeout": 120  # 2 minutes
            },
            "integration": {
                "name": "Integration Tests",
                "path": "math_learning/tests/integration/",
                "description": "AI agent integration, multi-user scenarios",
                "timeout": 180  # 3 minutes
            },
            "rag": {
                "name": "RAG Backend Tests",
                "path": "math_learning/tests/rag/",
                "description": "RAG integration and backend functionality",
                "timeout": 120  # 2 minutes
            },
            "real_db": {
                "name": "Real Database Tests",
                "path": "math_learning/tests/real_db/",
                "description": "Neo4j, Redis, Qdrant integration tests",
                "timeout": 240  # 4 minutes
            }
        }
    
    def run_test_suite(self, suite_name: str, suite_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test suite and return results."""
        print(f"\n{'='*80}")
        print(f"🧪 RUNNING: {suite_config['name']}")
        print(f"📁 Path: {suite_config['path']}")
        print(f"📝 Description: {suite_config['description']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Construct pytest command
            cmd = [
                sys.executable, "-m", "pytest",
                suite_config["path"],
                "-v",
                "--tb=short",
                "--maxfail=10",  # Stop after 10 failures
                "--capture=no"  # Show print statements
            ]
            
            # Add coverage if available
            try:
                import pytest_cov
                cmd.extend(["--cov=math_learning", "--cov-report=term-missing"])
            except ImportError:
                pass
            
            # Run the tests
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=suite_config["timeout"] + 30  # Extra buffer
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse results
            output_lines = result.stdout.split('\n')
            error_lines = result.stderr.split('\n')
            
            # Extract test statistics
            stats = self._parse_pytest_output(output_lines)
            
            suite_result = {
                "name": suite_config["name"],
                "status": "PASSED" if result.returncode == 0 else "FAILED",
                "duration": duration,
                "return_code": result.returncode,
                "stats": stats,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "summary": self._generate_suite_summary(stats, duration)
            }
            
            # Print summary
            self._print_suite_summary(suite_name, suite_result)
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            suite_result = {
                "name": suite_config["name"],
                "status": "TIMEOUT",
                "duration": duration,
                "return_code": -1,
                "stats": {"timeout": True},
                "stdout": "",
                "stderr": f"Test suite timed out after {suite_config['timeout']} seconds",
                "summary": f"❌ TIMEOUT after {duration:.1f}s"
            }
            
            print(f"❌ **{suite_config['name'].upper()} TIMED OUT**")
            print(f"   Duration: {duration:.1f} seconds")
            
            return suite_result
            
        except Exception as e:
            duration = time.time() - start_time
            suite_result = {
                "name": suite_config["name"],
                "status": "ERROR",
                "duration": duration,
                "return_code": -2,
                "stats": {"error": str(e)},
                "stdout": "",
                "stderr": str(e),
                "summary": f"❌ ERROR: {str(e)}"
            }
            
            print(f"❌ **{suite_config['name'].upper()} ERROR**")
            print(f"   Error: {str(e)}")
            
            return suite_result
    
    def _parse_pytest_output(self, output_lines: List[str]) -> Dict[str, Any]:
        """Parse pytest output to extract statistics."""
        stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "warnings": 0
        }
        
        for line in output_lines:
            # Look for summary line like "5 failed, 10 passed, 2 skipped in 30.2s"
            if " passed" in line or " failed" in line or " skipped" in line:
                if "=" in line and "in " in line:
                    # This is likely the summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i + 1 < len(parts):
                            count = int(part)
                            next_part = parts[i + 1]
                            
                            if "passed" in next_part:
                                stats["passed"] = count
                            elif "failed" in next_part:
                                stats["failed"] = count
                            elif "skipped" in next_part:
                                stats["skipped"] = count
                            elif "error" in next_part:
                                stats["errors"] = count
                    break
        
        stats["total"] = stats["passed"] + stats["failed"] + stats["skipped"] + stats["errors"]
        return stats
    
    def _generate_suite_summary(self, stats: Dict[str, Any], duration: float) -> str:
        """Generate a summary string for a test suite."""
        if "timeout" in stats:
            return f"❌ TIMEOUT after {duration:.1f}s"
        elif "error" in stats:
            return f"❌ ERROR: {stats['error']}"
        elif stats.get("failed", 0) > 0 or stats.get("errors", 0) > 0:
            return f"❌ FAILED ({stats['failed']} failed, {stats['errors']} errors) in {duration:.1f}s"
        elif stats.get("total", 0) == 0:
            return f"⚠️ NO TESTS FOUND in {duration:.1f}s"
        else:
            return f"✅ PASSED ({stats['passed']} passed, {stats['skipped']} skipped) in {duration:.1f}s"
    
    def _print_suite_summary(self, suite_name: str, result: Dict[str, Any]):
        """Print summary for a test suite."""
        print(f"\n{result['summary']}")
        
        if result["stats"].get("total", 0) > 0:
            stats = result["stats"]
            print(f"   📊 Tests: {stats['total']} total, {stats['passed']} passed, {stats['failed']} failed, {stats['skipped']} skipped")
        
        if result["status"] == "FAILED" and result["stderr"]:
            print(f"   ❌ Errors: {result['stderr'][:200]}...")
    
    def run_all_tests(self, suites: List[str] = None) -> Dict[str, Any]:
        """Run all test suites or specified suites."""
        self.start_time = time.time()
        
        if suites is None:
            suites = list(self.test_suites.keys())
        
        print("🚀 STARTING COMPREHENSIVE TEST SUITE")
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🧪 Test Suites: {', '.join(suites)}")
        print(f"🔧 Python Version: {sys.version}")
        print(f"📁 Working Directory: {os.getcwd()}")
        
        # Run each test suite
        for suite_name in suites:
            if suite_name not in self.test_suites:
                print(f"⚠️ Unknown test suite: {suite_name}")
                continue
            
            suite_config = self.test_suites[suite_name]
            self.test_results[suite_name] = self.run_test_suite(suite_name, suite_config)
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        # Print final summary
        self._print_final_summary()
        
        return report
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = self.end_time - self.start_time
        
        # Calculate overall statistics
        overall_stats = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "error_tests": 0,
            "passed_suites": 0,
            "failed_suites": 0,
            "timeout_suites": 0,
            "error_suites": 0
        }
        
        for suite_name, result in self.test_results.items():
            stats = result["stats"]
            
            # Aggregate test counts
            overall_stats["total_tests"] += stats.get("total", 0)
            overall_stats["passed_tests"] += stats.get("passed", 0)
            overall_stats["failed_tests"] += stats.get("failed", 0)
            overall_stats["skipped_tests"] += stats.get("skipped", 0)
            overall_stats["error_tests"] += stats.get("errors", 0)
            
            # Count suite statuses
            if result["status"] == "PASSED":
                overall_stats["passed_suites"] += 1
            elif result["status"] == "FAILED":
                overall_stats["failed_suites"] += 1
            elif result["status"] == "TIMEOUT":
                overall_stats["timeout_suites"] += 1
            elif result["status"] == "ERROR":
                overall_stats["error_suites"] += 1
        
        # Calculate success rates
        total_suites = len(self.test_results)
        suite_success_rate = overall_stats["passed_suites"] / total_suites * 100 if total_suites > 0 else 0
        test_success_rate = overall_stats["passed_tests"] / overall_stats["total_tests"] * 100 if overall_stats["total_tests"] > 0 else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "overall_stats": overall_stats,
            "suite_success_rate": suite_success_rate,
            "test_success_rate": test_success_rate,
            "suite_results": self.test_results,
            "summary": {
                "status": "PASSED" if overall_stats["failed_suites"] == 0 and overall_stats["timeout_suites"] == 0 and overall_stats["error_suites"] == 0 else "FAILED",
                "total_suites": total_suites,
                "total_tests": overall_stats["total_tests"],
                "duration_minutes": total_duration / 60
            }
        }
        
        return report
    
    def _print_final_summary(self):
        """Print final comprehensive summary."""
        total_duration = self.end_time - self.start_time
        
        print(f"\n{'='*100}")
        print("📊 COMPREHENSIVE TEST SUITE SUMMARY")
        print(f"{'='*100}")
        
        # Overall statistics
        overall_stats = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "passed_suites": 0,
            "failed_suites": 0,
            "timeout_suites": 0,
            "error_suites": 0
        }
        
        # Calculate stats and print suite results
        print("\n🧪 TEST SUITE RESULTS:")
        for suite_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"  {status_icon} {result['name']}: {result['summary']}")
            
            stats = result["stats"]
            overall_stats["total_tests"] += stats.get("total", 0)
            overall_stats["passed_tests"] += stats.get("passed", 0)
            overall_stats["failed_tests"] += stats.get("failed", 0)
            overall_stats["skipped_tests"] += stats.get("skipped", 0)
            
            if result["status"] == "PASSED":
                overall_stats["passed_suites"] += 1
            elif result["status"] == "FAILED":
                overall_stats["failed_suites"] += 1
            elif result["status"] == "TIMEOUT":
                overall_stats["timeout_suites"] += 1
            elif result["status"] == "ERROR":
                overall_stats["error_suites"] += 1
        
        # Print overall statistics
        total_suites = len(self.test_results)
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"  🏃 Test Suites: {overall_stats['passed_suites']}/{total_suites} passed ({overall_stats['passed_suites']/total_suites*100:.1f}%)")
        if overall_stats['total_tests'] > 0:
            print(f"  🧪 Individual Tests: {overall_stats['passed_tests']}/{overall_stats['total_tests']} passed ({overall_stats['passed_tests']/overall_stats['total_tests']*100:.1f}%)")
        else:
            print(f"  🧪 Individual Tests: 0/0 passed (no tests found)")
        print(f"  ⏱️ Total Duration: {total_duration/60:.1f} minutes")
        
        if overall_stats["failed_suites"] > 0:
            print(f"  ❌ Failed Suites: {overall_stats['failed_suites']}")
        if overall_stats["timeout_suites"] > 0:
            print(f"  ⏰ Timeout Suites: {overall_stats['timeout_suites']}")
        if overall_stats["error_suites"] > 0:
            print(f"  💥 Error Suites: {overall_stats['error_suites']}")
        if overall_stats["skipped_tests"] > 0:
            print(f"  ⏭️ Skipped Tests: {overall_stats['skipped_tests']}")
        
        # Final status
        all_passed = (overall_stats["failed_suites"] == 0 and 
                     overall_stats["timeout_suites"] == 0 and 
                     overall_stats["error_suites"] == 0)
        
        print(f"\n🎯 FINAL STATUS: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        if not all_passed:
            print("\n🔍 RECOMMENDED ACTIONS:")
            if overall_stats["failed_suites"] > 0:
                print("  • Review failed test cases and fix underlying issues")
            if overall_stats["timeout_suites"] > 0:
                print("  • Investigate performance issues causing timeouts")
            if overall_stats["error_suites"] > 0:
                print("  • Check test environment and dependencies")
        
        print(f"{'='*100}")
    
    def save_report(self, filename: str = None) -> str:
        """Save comprehensive report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_test_report_{timestamp}.json"
        
        report = self._generate_comprehensive_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Test report saved to: {filename}")
        return filename


def main():
    """Main function to run comprehensive tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive math learning system tests")
    parser.add_argument(
        "--suites",
        nargs="*",
        help="Specific test suites to run (default: all)",
        choices=["storage", "graph", "memory", "performance", "core", "integration", "rag", "real_db"]
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save detailed report to JSON file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only core tests for quick validation"
    )
    
    args = parser.parse_args()
    
    # Determine which suites to run
    if args.quick:
        suites = ["core"]
    elif args.suites:
        suites = args.suites
    else:
        suites = None  # Run all
    
    # Create and run test runner
    runner = ComprehensiveTestRunner()
    report = runner.run_all_tests(suites)
    
    # Save report if requested
    if args.save_report:
        runner.save_report()
    
    # Exit with appropriate code
    sys.exit(0 if report["summary"]["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main() 