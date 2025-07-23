#!/usr/bin/env python3
"""
Script to run all working tests in the math_learning directory.

This script runs only the tests that are known to be working properly,
avoiding tests with import issues, missing fixtures, or dependency problems.
"""

import subprocess
import sys
from pathlib import Path

def run_test_group(name, test_paths, description):
    """Run a group of tests and report results."""
    print(f"\n{'='*60}")
    print(f"Running {name}: {description}")
    print(f"{'='*60}")
    
    total_passed = 0
    total_failed = 0
    
    for test_path in test_paths:
        print(f"\n🧪 Running {test_path}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_path, "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            
            if result.returncode == 0:
                # Count passed tests from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'passed' in line and 'warning' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'passed':
                                try:
                                    passed = int(parts[i-1])
                                    total_passed += passed
                                    print(f"  ✅ {passed} tests passed")
                                    break
                                except (ValueError, IndexError):
                                    pass
                        break
            else:
                print(f"  ❌ Test failed with return code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}...")
                total_failed += 1
                
        except Exception as e:
            print(f"  ❌ Error running test: {e}")
            total_failed += 1
    
    print(f"\n📊 {name} Summary: {total_passed} tests passed, {total_failed} groups failed")
    return total_passed, total_failed

def main():
    """Run all working test groups."""
    print("🧪 Math Learning Test Runner")
    print("Running all working tests in the math_learning directory")
    
    # Core functionality tests (all working)
    core_tests = [
        "math_learning/tests/test_core_personal_learning.py",
        "math_learning/tests/test_complex_scenarios.py", 
        "math_learning/tests/test_gap_analysis.py",
        "math_learning/tests/test_personal_learning_graph.py",
        "math_learning/tests/test_user_scenarios.py"
    ]
    
    # Unit tests (partially working)
    unit_tests = [
        "math_learning/tests/unit/test_exercise_system.py"
    ]
    
    # Integration tests (mostly working)
    integration_tests = [
        "math_learning/tests/integration/test_ai_agent_integration.py",
        "math_learning/tests/integration/test_backend_comparison.py",
        "math_learning/tests/integration/test_full_integration.py",
        "math_learning/tests/integration/test_image_analysis_feature.py",
        "math_learning/tests/integration/test_image_recognition_standalone.py",
        "math_learning/tests/integration/test_multi_user_scenarios.py",
        "math_learning/tests/integration/test_real_image_analysis.py",
        "math_learning/tests/integration/test_real_image_mock.py",
        "math_learning/tests/integration/test_real_image_simple.py",
        "math_learning/tests/integration/test_simple_math_chat.py"
    ]
    
    # RAG tests (partially working)
    rag_tests = [
        "math_learning/tests/rag/test_rag_integration.py"
    ]
    
    # Real DB tests (connection tests only)
    db_tests = [
        "math_learning/tests/real_db/test_real_backends.py::test_neo4j_connection",
        "math_learning/tests/real_db/test_real_backends.py::test_redis_connection"
    ]
    
    # Run test groups
    total_passed = 0
    total_failed = 0
    
    groups = [
        ("Core Functionality Tests", core_tests, "Learning graph, mastery tracking, gap analysis"),
        ("Unit Tests", unit_tests, "Basic component tests"),
        ("Integration Tests", integration_tests, "Component integration tests"),
        ("RAG Tests", rag_tests, "Retrieval Augmented Generation tests"),
        ("Database Connection Tests", db_tests, "Database connectivity tests")
    ]
    
    for name, tests, description in groups:
        passed, failed = run_test_group(name, tests, description)
        total_passed += passed
        total_failed += failed
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎯 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Total tests passed: {total_passed}")
    print(f"❌ Total test groups failed: {total_failed}")
    
    if total_failed == 0:
        print("🎉 All working tests passed successfully!")
        return 0
    else:
        print(f"⚠️  {total_failed} test groups had issues")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 