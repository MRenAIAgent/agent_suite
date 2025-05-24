# Neo4j AuraDB Free Setup Guide

This guide helps you set up a free Neo4j AuraDB instance for testing the Neo4jStore implementation.

## Step 1: Create a Neo4j AuraDB Free Account

1. Go to [Neo4j AuraDB Free](https://console.neo4j.io/) and sign up for a free account
2. No credit card is required for the free tier
3. You get one free database instance with 50MB storage

## Step 2: Create a Database

1. After logging in, click "Create Database"
2. Select "AuraDB Free" (the free option)
3. Name your database (e.g., "agent-suite-test")
4. Click "Create Database"
5. Wait for your database to be created (this takes a few minutes)

## Step 3: Get Connection Details

Once your database is ready:

1. Click "Connect" on your database card
2. You'll see the connection URI, username, and password
3. Save these details as you'll need them for testing

## Step 4: Run Tests

Export the connection details as environment variables:

```bash
# Replace with your actual connection details
export NEO4J_URI="neo4j+s://xxxxxxxx.databases.neo4j.io"
export NEO4J_USERNAME="neo4j"  # Usually "neo4j" by default
export NEO4J_PASSWORD="your-password"
```

Then run the tests:

```bash
# Test basic connection
python examples/neo4j_test.py

# Run integration tests
python -m tests.integration.test_neo4j_integration
```

## AuraDB Free Limitations

- 50MB storage
- No multi-database support (only the default `neo4j` database)
- 1 million node limit
- Databases that are inactive for 90 days may be deleted

These limitations are more than sufficient for testing the Neo4jStore implementation.

## Cleanup

When you're done testing, you can delete the database to free up resources:

1. Go to your [Neo4j AuraDB console](https://console.neo4j.io/)
2. Click the three dots on your database card
3. Select "Terminate"
4. Confirm termination 