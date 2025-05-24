#!/bin/bash
# Script to run Neo4j tests with connection details

# Check if the connection details are provided
if [ -z "$NEO4J_URI" ] || [ -z "$NEO4J_PASSWORD" ]; then
    echo "Neo4j connection details missing. Please provide them as follows:"
    echo ""
    echo "Usage:"
    echo "  NEO4J_URI=neo4j+s://xxx.databases.neo4j.io NEO4J_PASSWORD=yourpassword ./examples/run_neo4j_tests.sh"
    echo ""
    echo "Or set them as environment variables before running this script:"
    echo "  export NEO4J_URI=neo4j+s://xxx.databases.neo4j.io"
    echo "  export NEO4J_PASSWORD=yourpassword"
    echo "  ./examples/run_neo4j_tests.sh"
    echo ""
    echo "You can sign up for a free Neo4j AuraDB account at https://console.neo4j.io/"
    exit 1
fi

# Default username if not provided
if [ -z "$NEO4J_USERNAME" ]; then
    export NEO4J_USERNAME="neo4j"
fi

# Default database name if not provided
if [ -z "$NEO4J_DATABASE" ]; then
    export NEO4J_DATABASE="neo4j"
fi

# Handle special characters in password (remove any trailing % sign)
NEO4J_PASSWORD=$(echo "$NEO4J_PASSWORD" | sed 's/%$//')
export NEO4J_PASSWORD

echo "===================================================================="
echo "Running Neo4j tests with the following connection details:"
echo "URI: $NEO4J_URI"
echo "Username: $NEO4J_USERNAME"
echo "Database: $NEO4J_DATABASE"
echo "===================================================================="
echo ""

# Step 1: Run the basic connection test
echo "Step 1: Testing basic Neo4j connection..."
python examples/neo4j_test.py
if [ $? -ne 0 ]; then
    echo "❌ Basic connection test failed. Please check your connection details."
    exit 1
fi
echo ""

# Step 2: Run the basic integration test
echo "Step 2: Running basic Neo4j integration test..."
python -m tests.integration.test_neo4j_integration
if [ $? -ne 0 ]; then
    echo "❌ Basic integration test failed."
    exit 1
fi
echo ""

# Step 3: Run the limits test
echo "Step 3: Running Neo4j limits test..."
python -m tests.integration.test_neo4j_limits
if [ $? -ne 0 ]; then
    echo "❌ Limits test failed."
    exit 1
fi
echo ""

# Step 4: Run the LongtermMemory integration test
echo "Step 4: Running Neo4j with LongtermMemory test..."
python -m tests.integration.test_neo4j_with_longterm_memory
if [ $? -ne 0 ]; then
    echo "❌ LongtermMemory integration test failed."
    exit 1
fi
echo ""

# Step 5: Run all Neo4j store unit tests with pytest
echo "Step 5: Running Neo4j store unit tests..."
python -m pytest tests/unit/agents/memory/stores/test_neo4j_store.py -v
if [ $? -ne 0 ]; then
    echo "❌ Store unit tests failed."
    exit 1
fi
echo ""

echo "✅ All Neo4j tests completed successfully!" 