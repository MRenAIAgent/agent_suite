#!/bin/bash
# Helper script to run tests with .env file

# Load environment variables from .env file
set -a
source "$(dirname "$(dirname "$0")")/.env"
set +a

# Run the tests
"$(dirname "$0")/run_neo4j_tests.sh" 