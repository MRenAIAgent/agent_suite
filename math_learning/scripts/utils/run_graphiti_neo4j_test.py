#!/usr/bin/env python3
"""
Script to run the Graphiti + Neo4j math learning test with database usage monitoring.
"""

import asyncio
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_usage_checker import DatabaseUsageChecker
from test_graphiti_neo4j_math_learning import test_graphiti_neo4j_math_learning


async def main():
    """Run the Graphiti + Neo4j test with usage monitoring."""
    print("🚀 Starting Graphiti + Neo4j Math Learning Test")
    print("="*80)
    
    # Initialize usage checker
    usage_checker = DatabaseUsageChecker()
    
    # Check usage before test
    print("\n📊 CHECKING DATABASE USAGE BEFORE TEST")
    print("-" * 50)
    usage_before = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_before, "Before Graphiti + Neo4j Test")
    
    # Run the test
    print("\n🧪 RUNNING GRAPHITI + NEO4J TEST")
    print("-" * 50)
    try:
        await test_graphiti_neo4j_math_learning()
        print("✅ Test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return
    
    # Check usage after test
    print("\n📊 CHECKING DATABASE USAGE AFTER TEST")
    print("-" * 50)
    usage_after = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_after, "After Graphiti + Neo4j Test")
    
    # Show the difference
    print("\n📈 DATABASE USAGE CHANGES")
    print("-" * 50)
    usage_checker.print_usage_comparison(usage_before, usage_after)
    
    print("\n🎉 Test and usage monitoring completed!")


if __name__ == "__main__":
    asyncio.run(main()) 