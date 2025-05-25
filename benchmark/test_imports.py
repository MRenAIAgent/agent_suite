#!/usr/bin/env python3
"""
Test imports for the benchmark package.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from benchmark.agent.code.agent_benchmark import create_and_run_benchmark
    print("✅ Successfully imported agent benchmark")
except Exception as e:
    print(f"❌ Failed to import agent benchmark: {e}")
    
try:
    from benchmark.rag.code.rag_top_benchmark import main as rag_main
    print("✅ Successfully imported RAG benchmark")
except Exception as e:
    print(f"❌ Failed to import RAG benchmark: {e}")
    
print("Import test complete!") 