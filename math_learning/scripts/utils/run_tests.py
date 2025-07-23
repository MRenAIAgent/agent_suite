#!/usr/bin/env python3
"""
Test runner for Personal Learning Graph tests.

This script runs the comprehensive test suite for the personal learning system.
"""

import sys
import os
import subprocess

def run_tests():
    """Run the personal learning graph tests."""
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "tests", "test_personal_learning_graph.py")
    
    print("🧪 Running Personal Learning Graph Tests")
    print("=" * 50)
    
    try:
        # Try to run with pytest first
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
        ], cwd=script_dir, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")
            
    except FileNotFoundError:
        print("⚠️  pytest not found, trying to run tests directly...")
        
        # Fallback: run the test file directly
        try:
            result = subprocess.run([
                sys.executable, test_file
            ], cwd=script_dir, capture_output=False)
            
            if result.returncode == 0:
                print("\n✅ Tests completed!")
            else:
                print(f"\n❌ Tests failed (exit code: {result.returncode})")
                
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    return result.returncode == 0

def install_pytest():
    """Install pytest if it's not available."""
    print("📦 Installing pytest...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"], check=True)
        print("✅ pytest installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install pytest")
        return False

if __name__ == "__main__":
    print("🚀 Personal Learning Graph Test Suite")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                      capture_output=True, check=True)
        print("✅ pytest is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  pytest not found")
        if input("Install pytest? (y/n): ").lower().startswith('y'):
            if not install_pytest():
                print("Continuing without pytest...")
    
    # Run the tests
    success = run_tests()
    
    if success:
        print("\n🎉 Test suite completed successfully!")
    else:
        print("\n💥 Test suite encountered issues")
        sys.exit(1) 