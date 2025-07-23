#!/bin/bash

# Setup and Test Real Databases Script
# This script sets up Qdrant and Neo4j using Docker and runs the integration tests

set -e  # Exit on any error

echo "🚀 Setting up Real Database Integration Test Environment"
echo "======================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Start databases
start_databases() {
    print_status "Starting Qdrant and Neo4j databases..."
    
    # Stop any existing containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Start the databases
    docker-compose up -d
    
    print_status "Waiting for databases to be ready..."
    
    # Wait for Qdrant
    print_status "Waiting for Qdrant to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:6333/health > /dev/null 2>&1; then
            print_success "Qdrant is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Qdrant failed to start within 30 seconds"
            docker-compose logs qdrant
            exit 1
        fi
        sleep 1
    done
    
    # Wait for Neo4j
    print_status "Waiting for Neo4j to be ready..."
    for i in {1..60}; do
        if docker exec neo4j_math_learning cypher-shell -u neo4j -p password "RETURN 1" > /dev/null 2>&1; then
            print_success "Neo4j is ready!"
            break
        fi
        if [ $i -eq 60 ]; then
            print_error "Neo4j failed to start within 60 seconds"
            docker-compose logs neo4j
            exit 1
        fi
        sleep 1
    done
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Not in a virtual environment. Consider using one."
    fi
    
    # Install required packages
    pip install qdrant-client neo4j sentence-transformers
    
    print_success "Dependencies installed"
}

# Run the tests
run_tests() {
    print_status "Running real database integration tests..."
    
    # Make sure we're in the right directory
    cd "$(dirname "$0")"
    
    # Run the test
    python test_real_databases.py
    
    test_exit_code=$?
    
    if [ $test_exit_code -eq 0 ]; then
        print_success "All tests passed! 🎉"
    else
        print_error "Some tests failed. Check the output above for details."
    fi
    
    return $test_exit_code
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker-compose down --remove-orphans
    print_success "Cleanup complete"
}

# Show database status
show_status() {
    print_status "Database Status:"
    echo "=================="
    
    # Qdrant status
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        print_success "Qdrant: Running (http://localhost:6333)"
    else
        print_error "Qdrant: Not running"
    fi
    
    # Neo4j status
    if docker exec neo4j_math_learning cypher-shell -u neo4j -p password "RETURN 1" > /dev/null 2>&1; then
        print_success "Neo4j: Running (http://localhost:7474, bolt://localhost:7687)"
    else
        print_error "Neo4j: Not running"
    fi
    
    echo ""
    print_status "Docker containers:"
    docker-compose ps
}

# Show help
show_help() {
    echo "Real Database Integration Test Setup"
    echo "===================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start databases and run tests (default)"
    echo "  stop      Stop databases"
    echo "  restart   Restart databases"
    echo "  test      Run tests only (assumes databases are running)"
    echo "  status    Show database status"
    echo "  logs      Show database logs"
    echo "  clean     Stop databases and remove volumes"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Start databases and run tests"
    echo "  $0 start          # Same as above"
    echo "  $0 test           # Run tests only"
    echo "  $0 status         # Check if databases are running"
    echo "  $0 stop           # Stop databases"
    echo ""
}

# Main script logic
main() {
    case "${1:-start}" in
        "start")
            check_docker
            start_databases
            install_dependencies
            run_tests
            ;;
        "stop")
            cleanup
            ;;
        "restart")
            cleanup
            start_databases
            ;;
        "test")
            install_dependencies
            run_tests
            ;;
        "status")
            show_status
            ;;
        "logs")
            docker-compose logs
            ;;
        "clean")
            print_status "Stopping databases and removing volumes..."
            docker-compose down --volumes --remove-orphans
            print_success "Clean complete"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Trap to cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@" 