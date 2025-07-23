#!/usr/bin/env python3
"""
Simple test to check database connections.
"""

import asyncio
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_real_databases import get_real_databases_config, check_database_connections

async def main():
    config = get_real_databases_config()
    print(f"Neo4j URI: {config.neo4j_uri}")
    print(f"Qdrant: {config.qdrant_host}:{config.qdrant_port}")
    
    connections = await check_database_connections(config)
    print('Connections:', connections)
    
    if all(connections.values()):
        print("✅ All databases connected successfully!")
        return True
    else:
        print("❌ Some database connections failed!")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 