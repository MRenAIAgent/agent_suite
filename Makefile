# Agent Suite - Root Makefile
# Provides convenient commands for testing the entire project

.PHONY: help install install-dev clean test test-agents test-math test-all coverage lint format check docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# Python and pip commands
PYTHON := python3
PIP := pip3
PYTEST := python -m pytest

help: ## Show this help message
	@echo "$(BLUE)Agent Suite - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Testing Commands:$(NC)"
	@echo "  make test              - Run all tests (agents + math_learning)"
	@echo "  make test-agents       - Run agent framework tests only"
	@echo "  make test-math         - Run math learning tests only"
	@echo "  make test-comprehensive - Run comprehensive test suite (storage, graph, memory, performance)"
	@echo "  make test-storage      - Run storage backend tests"
	@echo "  make test-graph        - Run graph operations tests"
	@echo "  make test-memory       - Run memory management tests"
	@echo "  make test-performance  - Run performance and scalability tests"
	@echo "  make test-quick        - Run quick validation tests"
	@echo "  make test-all          - Run ALL tests including external dependencies"
	@echo "  make coverage          - Run tests with coverage report"
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make clean         - Clean up temporary files"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code"
	@echo "  make check         - Run all code quality checks"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make install-dev && make test"

# Installation commands
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -e .
	@if [ -f "requirements.txt" ]; then $(PIP) install -r requirements.txt; fi

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -e .
	@if [ -f "requirements.txt" ]; then $(PIP) install -r requirements.txt; fi
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-html pytest-xdist
	$(PIP) install black flake8 isort mypy ruff
	@echo "$(GREEN)Development dependencies installed successfully!$(NC)"
	@echo "$(BLUE)Installing math_learning dependencies...$(NC)"
	@cd math_learning && $(MAKE) install-dev
	@echo "$(GREEN)All dependencies installed!$(NC)"

# Testing commands
test: test-agents test-math ## Run all standard tests
	@echo "$(GREEN)All tests completed!$(NC)"

test-agents: ## Run agent framework tests
	@echo "$(BLUE)Running agent framework tests...$(NC)"
	PYTHONPATH=$$PWD $(PYTEST) tests/ -v --tb=short --asyncio-mode=auto -x
	@echo "$(GREEN)Agent tests completed!$(NC)"

test-math: ## Run math learning tests
	@echo "$(BLUE)Running math learning tests...$(NC)"
	@cd math_learning && $(MAKE) test
	@echo "$(GREEN)Math learning tests completed!$(NC)"

test-math-working: ## Run only the working math learning tests
	@echo "$(BLUE)Running working math learning tests...$(NC)"
	@cd math_learning && $(PYTHON) tests/run_working_tests.py
	@echo "$(GREEN)Working math learning tests completed!$(NC)"

test-comprehensive: ## Run comprehensive test suite with storage, graph, memory, and performance tests
	@echo "$(BLUE)Running comprehensive test suite with all new tests...$(NC)"
	@cd math_learning && $(PYTHON) tests/run_comprehensive_tests.py --save-report
	@echo "$(GREEN)Comprehensive test suite completed!$(NC)"

test-storage: ## Run storage backend tests
	@echo "$(BLUE)Running storage backend tests...$(NC)"
	@cd math_learning && $(PYTHON) tests/run_comprehensive_tests.py --suites storage
	@echo "$(GREEN)Storage tests completed!$(NC)"

test-graph: ## Run graph operations tests
	@echo "$(BLUE)Running graph operations tests...$(NC)"
	@cd math_learning && $(PYTHON) tests/run_comprehensive_tests.py --suites graph
	@echo "$(GREEN)Graph tests completed!$(NC)"

test-memory: ## Run memory management tests
	@echo "$(BLUE)Running memory management tests...$(NC)"
	@cd math_learning && $(PYTHON) tests/run_comprehensive_tests.py --suites memory
	@echo "$(GREEN)Memory tests completed!$(NC)"

test-performance: ## Run performance and scalability tests
	@echo "$(BLUE)Running performance and scalability tests...$(NC)"
	@cd math_learning && $(PYTHON) tests/run_comprehensive_tests.py --suites performance
	@echo "$(GREEN)Performance tests completed!$(NC)"

test-quick: ## Run quick validation tests
	@echo "$(BLUE)Running quick validation tests...$(NC)"
	@cd math_learning && $(PYTHON) tests/run_comprehensive_tests.py --quick
	@echo "$(GREEN)Quick tests completed!$(NC)"

test-all: test-agents test-math-all ## Run ALL tests including external dependencies
	@echo "$(GREEN)All tests completed successfully!$(NC)"

test-math-all: ## Run all math learning tests including database tests
	@echo "$(BLUE)Running all math learning tests...$(NC)"
	@cd math_learning && $(MAKE) test-all
	@echo "$(GREEN)All math learning tests completed!$(NC)"

coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	PYTHONPATH=$$PWD $(PYTEST) tests/ --cov=agents --cov=tools --cov-report=html --cov-report=term-missing --cov-report=xml --asyncio-mode=auto
	@cd math_learning && $(MAKE) coverage
	@echo "$(GREEN)Coverage reports generated!$(NC)"
	@echo "$(BLUE)Agent coverage: htmlcov/index.html$(NC)"
	@echo "$(BLUE)Math learning coverage: math_learning/htmlcov/index.html$(NC)"

# Development commands
clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@cd math_learning && $(MAKE) clean
	@echo "$(GREEN)Cleanup completed!$(NC)"

lint: ## Run code linting
	@echo "$(BLUE)Running code linting...$(NC)"
	ruff check . --fix || true
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
	@cd math_learning && $(MAKE) lint
	@echo "$(GREEN)Linting completed!$(NC)"

format: ## Format code using black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black . --line-length=88
	isort . --profile black
	@cd math_learning && $(MAKE) format
	@echo "$(GREEN)Code formatting completed!$(NC)"

check: lint ## Run all code quality checks
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy . --ignore-missing-imports || true
	@cd math_learning && $(MAKE) check
	@echo "$(GREEN)Code quality checks completed!$(NC)"

# Quick testing workflows
test-quick: ## Run quick tests (unit tests only)
	@echo "$(BLUE)Running quick tests...$(NC)"
	PYTHONPATH=$$PWD $(PYTEST) tests/unit/ -x --tb=short --asyncio-mode=auto
	@cd math_learning && $(MAKE) test-quick
	@echo "$(GREEN)Quick tests completed!$(NC)"

# Development workflow commands
dev-setup: install-dev ## Complete development setup
	@echo "$(GREEN)Development environment setup completed!$(NC)"
	@echo "$(BLUE)You can now run: make test$(NC)"

ci-test: ## Run tests suitable for CI environment
	@echo "$(BLUE)Running CI tests...$(NC)"
	PYTHONPATH=$$PWD $(PYTEST) tests/ -v --tb=short --cov=agents --cov=tools --cov-report=xml --asyncio-mode=auto
	@cd math_learning && $(MAKE) ci-test
	@echo "$(GREEN)CI tests completed!$(NC)"

# Utility commands
show-structure: ## Show current directory structure
	@echo "$(BLUE)Agent Suite Directory Structure:$(NC)"
	@tree -I '__pycache__|*.pyc|.pytest_cache|htmlcov|node_modules|.git' -a -L 2 || \
	find . -type d -name "__pycache__" -prune -o -type f -print | head -30

env-check: ## Check environment and dependencies
	@echo "$(BLUE)Environment Check:$(NC)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "$(BLUE)Checking key dependencies:$(NC)"
	@$(PYTHON) -c "import pytest; print('✓ pytest available')" 2>/dev/null || echo "✗ pytest not available"
	@$(PYTHON) -c "import agents; print('✓ agents module available')" 2>/dev/null || echo "✗ agents module not available"
	@$(PYTHON) -c "import tools; print('✓ tools module available')" 2>/dev/null || echo "✗ tools module not available"

# Math learning specific commands (delegated)
test-math-unit: ## Run math learning unit tests
	@cd math_learning && $(MAKE) test-unit

test-math-integration: ## Run math learning integration tests
	@cd math_learning && $(MAKE) test-integration

test-math-system: ## Run math learning system tests
	@cd math_learning && $(MAKE) test-system

test-math-db: ## Run math learning database tests
	@cd math_learning && $(MAKE) test-real-db

setup-math-db: ## Setup math learning databases
	@cd math_learning && $(MAKE) setup-db

# Example usage targets
examples: ## Show example usage commands
	@echo "$(BLUE)Example Usage:$(NC)"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make install-dev    # Install all dependencies"
	@echo "  make test          # Run standard tests"
	@echo ""
	@echo "$(GREEN)Full Testing:$(NC)"
	@echo "  make test-all      # Run all tests"
	@echo "  make coverage      # Generate coverage reports"
	@echo ""
	@echo "$(GREEN)Development Workflow:$(NC)"
	@echo "  make dev-setup     # Complete dev environment setup"
	@echo "  make test-quick    # Quick tests during development"
	@echo ""
	@echo "$(GREEN)Math Learning Specific:$(NC)"
	@echo "  make test-math-working  # Run only working math tests"
	@echo "  make setup-math-db      # Setup math learning databases"
	@echo "  make test-math-db       # Run math database tests"

# Integration with external systems
benchmark: ## Run benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	@if [ -d "benchmark" ]; then \
		cd benchmark && $(PYTHON) run_all_benchmarks.py; \
	else \
		echo "$(YELLOW)No benchmark directory found$(NC)"; \
	fi 