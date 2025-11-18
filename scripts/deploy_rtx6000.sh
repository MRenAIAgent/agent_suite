#!/bin/bash

###############################################################################
# RTX 6000 NVIDIA Container Deployment Script
#
# This script automates the deployment of the Math Learning application
# on a VM with NVIDIA RTX 6000 GPU support.
#
# Usage:
#   ./deploy_rtx6000.sh [command]
#
# Commands:
#   check       - Check prerequisites and system requirements
#   build       - Build the GPU-enabled Docker image
#   deploy      - Deploy the application stack
#   start       - Start all services
#   stop        - Stop all services
#   restart     - Restart all services
#   status      - Show status of all services
#   logs        - Show logs from all services
#   gpu-test    - Test GPU availability in container
#   cleanup     - Stop and remove all containers and volumes
#   help        - Show this help message
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MATH_LEARNING_DIR="$PROJECT_ROOT/math_learning"
DOCKER_COMPOSE_FILE="$MATH_LEARNING_DIR/docker-compose.gpu.yml"
ENV_FILE="$MATH_LEARNING_DIR/.env"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running with sufficient privileges
check_privileges() {
    if [ "$EUID" -eq 0 ]; then
        log_warning "Running as root. This is not recommended."
        log_warning "Consider adding your user to the docker group instead."
    fi
}

# Check if NVIDIA GPU is available
check_gpu() {
    log_info "Checking for NVIDIA GPU..."

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. NVIDIA drivers may not be installed."
        log_error "Please install NVIDIA drivers before proceeding."
        return 1
    fi

    if ! nvidia-smi &> /dev/null; then
        log_error "nvidia-smi failed. GPU may not be accessible."
        return 1
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -n 1)

    log_success "GPU detected: $GPU_NAME"
    log_info "GPU Memory: $GPU_MEMORY"
    log_info "Driver Version: $DRIVER_VERSION"
    log_info "CUDA Version: $CUDA_VERSION"

    return 0
}

# Check if Docker is installed and running
check_docker() {
    log_info "Checking Docker installation..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker before proceeding."
        return 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or you don't have permissions."
        log_error "Try: sudo systemctl start docker"
        log_error "Or add your user to docker group: sudo usermod -aG docker \$USER"
        return 1
    fi

    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    log_success "Docker installed: $DOCKER_VERSION"

    return 0
}

# Check if Docker Compose is installed
check_docker_compose() {
    log_info "Checking Docker Compose installation..."

    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose plugin."
        return 1
    fi

    COMPOSE_VERSION=$(docker compose version --short)
    log_success "Docker Compose installed: $COMPOSE_VERSION"

    return 0
}

# Check if NVIDIA Container Toolkit is installed
check_nvidia_docker() {
    log_info "Checking NVIDIA Container Toolkit..."

    if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_error "NVIDIA Container Toolkit not properly configured."
        log_error "Please install and configure NVIDIA Container Toolkit."
        return 1
    fi

    log_success "NVIDIA Container Toolkit is properly configured"

    return 0
}

# Check disk space
check_disk_space() {
    log_info "Checking disk space..."

    AVAILABLE_SPACE=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    REQUIRED_SPACE=20

    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        log_warning "Low disk space: ${AVAILABLE_SPACE}GB available"
        log_warning "At least ${REQUIRED_SPACE}GB recommended"
    else
        log_success "Disk space sufficient: ${AVAILABLE_SPACE}GB available"
    fi
}

# Check memory
check_memory() {
    log_info "Checking system memory..."

    TOTAL_MEM=$(free -g | awk 'NR==2 {print $2}')
    REQUIRED_MEM=8

    if [ "$TOTAL_MEM" -lt "$REQUIRED_MEM" ]; then
        log_warning "Low system memory: ${TOTAL_MEM}GB"
        log_warning "At least ${REQUIRED_MEM}GB recommended"
    else
        log_success "System memory sufficient: ${TOTAL_MEM}GB"
    fi
}

# Create .env file if it doesn't exist
create_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating .env file..."
        cat > "$ENV_FILE" << 'EOF'
# API Keys (replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Application Settings
LOG_LEVEL=INFO
WORKERS=4
THREADS=2
EOF
        log_success "Created .env file at $ENV_FILE"
        log_warning "Please update API keys in $ENV_FILE"
    else
        log_info ".env file already exists"
    fi
}

# Run all prerequisite checks
check_prerequisites() {
    log_info "Running prerequisite checks..."
    echo ""

    local failed=0

    check_gpu || failed=1
    echo ""

    check_docker || failed=1
    echo ""

    check_docker_compose || failed=1
    echo ""

    check_nvidia_docker || failed=1
    echo ""

    check_disk_space
    echo ""

    check_memory
    echo ""

    if [ $failed -eq 1 ]; then
        log_error "Prerequisite checks failed. Please fix the issues above."
        exit 1
    fi

    log_success "All prerequisite checks passed!"
    return 0
}

# Build Docker image
build_image() {
    log_info "Building GPU-enabled Docker image..."

    cd "$PROJECT_ROOT"

    if docker build -f Dockerfile.gpu -t math-learning-gpu:latest .; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Deploy the application
deploy_application() {
    log_info "Deploying Math Learning application..."

    create_env_file

    cd "$MATH_LEARNING_DIR"

    # Pull required images
    log_info "Pulling required Docker images..."
    docker compose -f "$DOCKER_COMPOSE_FILE" pull qdrant neo4j

    # Start services
    log_info "Starting services..."
    if docker compose -f "$DOCKER_COMPOSE_FILE" up -d; then
        log_success "Application deployed successfully"
        echo ""
        log_info "Services starting up. This may take a minute..."
        sleep 10
        show_status
    else
        log_error "Failed to deploy application"
        exit 1
    fi
}

# Start services
start_services() {
    log_info "Starting services..."

    cd "$MATH_LEARNING_DIR"

    if docker compose -f "$DOCKER_COMPOSE_FILE" up -d; then
        log_success "Services started"
        show_status
    else
        log_error "Failed to start services"
        exit 1
    fi
}

# Stop services
stop_services() {
    log_info "Stopping services..."

    cd "$MATH_LEARNING_DIR"

    if docker compose -f "$DOCKER_COMPOSE_FILE" down; then
        log_success "Services stopped"
    else
        log_error "Failed to stop services"
        exit 1
    fi
}

# Restart services
restart_services() {
    log_info "Restarting services..."
    stop_services
    sleep 2
    start_services
}

# Show service status
show_status() {
    log_info "Service status:"
    echo ""

    cd "$MATH_LEARNING_DIR"
    docker compose -f "$DOCKER_COMPOSE_FILE" ps

    echo ""
    log_info "GPU status in container:"
    docker compose -f "$DOCKER_COMPOSE_FILE" exec -T math-learning nvidia-smi 2>/dev/null || log_warning "Math learning container not running"
}

# Show logs
show_logs() {
    cd "$MATH_LEARNING_DIR"
    docker compose -f "$DOCKER_COMPOSE_FILE" logs -f
}

# Test GPU in container
test_gpu() {
    log_info "Testing GPU availability in container..."

    cd "$MATH_LEARNING_DIR"

    log_info "Running nvidia-smi in container..."
    docker compose -f "$DOCKER_COMPOSE_FILE" exec math-learning nvidia-smi

    echo ""
    log_info "Testing PyTorch CUDA availability..."
    docker compose -f "$DOCKER_COMPOSE_FILE" exec math-learning python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
EOF

    if [ $? -eq 0 ]; then
        log_success "GPU test passed!"
    else
        log_error "GPU test failed"
        exit 1
    fi
}

# Cleanup everything
cleanup() {
    log_warning "This will stop and remove all containers and volumes."
    read -p "Are you sure? (yes/no): " -r
    echo

    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Cleaning up..."
        cd "$MATH_LEARNING_DIR"
        docker compose -f "$DOCKER_COMPOSE_FILE" down -v
        log_success "Cleanup complete"
    else
        log_info "Cleanup cancelled"
    fi
}

# Show help
show_help() {
    cat << EOF
RTX 6000 NVIDIA Container Deployment Script

Usage:
    $0 [command]

Commands:
    check       Check prerequisites and system requirements
    build       Build the GPU-enabled Docker image
    deploy      Deploy the application stack (build + start)
    start       Start all services
    stop        Stop all services
    restart     Restart all services
    status      Show status of all services
    logs        Show logs from all services (follow mode)
    gpu-test    Test GPU availability in container
    cleanup     Stop and remove all containers and volumes
    help        Show this help message

Examples:
    $0 check                # Check if system meets requirements
    $0 deploy               # Deploy the entire stack
    $0 status               # Check service status
    $0 logs                 # View logs
    $0 gpu-test             # Verify GPU is working in container

For more information, see: $PROJECT_ROOT/docs/RTX6000_DEPLOYMENT_GUIDE.md
EOF
}

# Main script logic
main() {
    check_privileges

    case "${1:-help}" in
        check)
            check_prerequisites
            ;;
        build)
            build_image
            ;;
        deploy)
            check_prerequisites
            build_image
            deploy_application
            ;;
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        gpu-test)
            test_gpu
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
