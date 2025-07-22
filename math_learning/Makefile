# Math Learning System - Makefile
# Provides convenient commands for testing, setup, and development

.PHONY: help install install-dev clean test test-unit test-integration test-system test-e2e test-real-db test-all setup-db start-db stop-db clean-db coverage lint format check docs

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

# Database configuration
QDRANT_HOST := localhost
QDRANT_PORT := 6333
NEO4J_URI := bolt://localhost:7687
NEO4J_USER := neo4j
NEO4J_PASS := password

help: ## Show this help message
	@echo "$(BLUE)Math Learning System - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Testing Commands:$(NC)"
	@echo "  make test          - Run all tests (unit + integration + system)"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-system   - Run system/end-to-end tests"
	@echo "  make test-e2e      - Alias for test-system"
	@echo "  make test-real-db  - Run real database tests (requires DBs)"
	@echo "  make test-all      - Run ALL tests including real database tests"
	@echo "  make coverage      - Run tests with coverage report"
	@echo ""
	@echo "$(GREEN)Database Commands:$(NC)"
	@echo "  make setup-db      - Setup and start databases with Docker"
	@echo "  make start-db      - Start database containers"
	@echo "  make stop-db       - Stop database containers"
	@echo "  make clean-db      - Stop and remove database containers/data"
	@echo "  make check-db      - Check database connection status"
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
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies including test requirements
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-html pytest-xdist
	$(PIP) install qdrant-client neo4j sentence-transformers
	$(PIP) install black flake8 isort mypy
	@echo "$(GREEN)Development dependencies installed successfully!$(NC)"

# Testing commands
test: test-unit test-integration ## Run standard tests (unit + integration)
	@echo "$(GREEN)All standard tests completed!$(NC)"

test-unit: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	PYTHONPATH=$$PWD:$$PWD/.. $(PYTEST) tests/unit/ -v --tb=short
	@echo "$(GREEN)Unit tests completed!$(NC)"

test-integration: ## Run integration tests (memory backends)
	@echo "$(BLUE)Running integration tests...$(NC)"
	@echo "$(YELLOW)Note: Some tests may fail due to missing API keys or external dependencies$(NC)"
	PYTHONPATH=$$PWD:$$PWD/.. $(PYTEST) tests/integration/ -v --tb=short --asyncio-mode=auto -k "not real_db"
	@echo "$(GREEN)Integration tests completed!$(NC)"
	@echo "$(BLUE)Expected failures: OpenAI API key tests, some image analysis tests$(NC)"

test-integration-core: ## Run core integration tests (excluding API key dependent tests)
	@echo "$(BLUE)Running core integration tests...$(NC)"
	PYTHONPATH=$$PWD:$$PWD/.. $(PYTEST) tests/integration/ -v --tb=short --asyncio-mode=auto -k "not real_db and not image_analysis and not real_image and not user_scenario"
	@echo "$(GREEN)Core integration tests completed!$(NC)"

test-system: ## Run system and end-to-end tests
	@echo "$(BLUE)Running system tests...$(NC)"
	@if [ -f "tests/system/run_all_tests.py" ]; then \
		cd tests/system && PYTHONPATH=$$PWD/../..:$$PWD/../../.. $(PYTHON) run_all_tests.py; \
	else \
		PYTHONPATH=$$PWD:$$PWD/.. $(PYTEST) tests/system/ -v --tb=short --asyncio-mode=auto; \
	fi
	@echo "$(GREEN)System tests completed!$(NC)"

test-e2e: test-system ## Alias for system tests

test-real-db: check-db ## Run real database tests (requires running databases)
	@echo "$(BLUE)Running real database tests...$(NC)"
	@echo "$(YELLOW)Note: This requires Qdrant and Neo4j to be running$(NC)"
	PYTHONPATH=$$PWD:$$PWD/.. $(PYTEST) tests/real_db/ -v --tb=short --asyncio-mode=auto
	@if [ -f "tests/real_db/test_storage_backends.py" ]; then \
		PYTHONPATH=$$PWD:$$PWD/.. $(PYTHON) tests/real_db/test_storage_backends.py; \
	fi
	@echo "$(GREEN)Real database tests completed!$(NC)"

test-all: test-unit test-integration test-system test-real-db ## Run ALL tests including real database tests
	@echo "$(GREEN)All tests completed successfully!$(NC)"

coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	PYTHONPATH=$$PWD:$$PWD/.. $(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term-missing --cov-report=xml --asyncio-mode=auto
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

# Database management commands
setup-db: ## Setup and start databases with Docker Compose
	@echo "$(BLUE)Setting up databases with Docker Compose...$(NC)"
	@if [ ! -f "docker-compose.yml" ]; then \
		echo "$(RED)docker-compose.yml not found in current directory$(NC)"; \
		exit 1; \
	fi
	docker-compose up -d
	@echo "$(YELLOW)Waiting for databases to be ready...$(NC)"
	@echo "$(BLUE)Waiting for Qdrant...$(NC)"
	@for i in $$(seq 1 30); do \
		if curl -s http://$(QDRANT_HOST):$(QDRANT_PORT)/health > /dev/null 2>&1; then \
			echo "$(GREEN)Qdrant is ready!$(NC)"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "$(RED)Qdrant failed to start within 30 seconds$(NC)"; \
			docker-compose logs qdrant; \
			exit 1; \
		fi; \
		sleep 1; \
	done
	@echo "$(BLUE)Waiting for Neo4j...$(NC)"
	@for i in $$(seq 1 60); do \
		if docker exec neo4j_math_learning cypher-shell -u $(NEO4J_USER) -p $(NEO4J_PASS) "RETURN 1" > /dev/null 2>&1; then \
			echo "$(GREEN)Neo4j is ready!$(NC)"; \
			break; \
		fi; \
		if [ $$i -eq 60 ]; then \
			echo "$(RED)Neo4j failed to start within 60 seconds$(NC)"; \
			docker-compose logs neo4j; \
			exit 1; \
		fi; \
		sleep 1; \
	done
	@echo "$(GREEN)Databases are ready!$(NC)"

start-db: ## Start database containers
	@echo "$(BLUE)Starting database containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Database containers started!$(NC)"

stop-db: ## Stop database containers
	@echo "$(BLUE)Stopping database containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)Database containers stopped!$(NC)"

clean-db: ## Stop and remove database containers and data
	@echo "$(BLUE)Cleaning up databases...$(NC)"
	docker-compose down --volumes --remove-orphans
	@echo "$(GREEN)Database cleanup completed!$(NC)"

check-db: ## Check database connection status
	@echo "$(BLUE)Checking database connections...$(NC)"
	@echo "$(BLUE)Checking Qdrant...$(NC)"
	@if curl -s http://$(QDRANT_HOST):$(QDRANT_PORT)/health > /dev/null 2>&1; then \
		echo "$(GREEN)✓ Qdrant is available at $(QDRANT_HOST):$(QDRANT_PORT)$(NC)"; \
	else \
		echo "$(RED)✗ Qdrant is not available at $(QDRANT_HOST):$(QDRANT_PORT)$(NC)"; \
	fi
	@echo "$(BLUE)Checking Neo4j...$(NC)"
	@if docker exec neo4j_math_learning cypher-shell -u $(NEO4J_USER) -p $(NEO4J_PASS) "RETURN 1" > /dev/null 2>&1; then \
		echo "$(GREEN)✓ Neo4j is available at $(NEO4J_URI)$(NC)"; \
	else \
		echo "$(RED)✗ Neo4j is not available at $(NEO4J_URI)$(NC)"; \
	fi

# Development commands
clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "$(GREEN)Cleanup completed!$(NC)"

lint: ## Run code linting
	@echo "$(BLUE)Running code linting...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Format code using black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black . --line-length=100
	isort . --profile black

check: lint ## Run all code quality checks
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy . --ignore-missing-imports || true
	@echo "$(GREEN)Code quality checks completed!$(NC)"

# Documentation commands
docs: ## Generate documentation
	@echo "$(BLUE)Documentation available in docs/ directory$(NC)"
	@echo "$(GREEN)Key documentation files:$(NC)"
	@echo "  docs/guides/README_TESTING.md - Testing guide"
	@echo "  docs/guides/README_REAL_DB_TESTS.md - Database testing guide"
	@echo "  docs/analysis/ - Analysis and review documents"
	@echo "  docs/reports/ - Test reports and summaries"

# Quick testing workflows
test-quick: ## Run quick tests (unit tests only)
	@echo "$(BLUE)Running quick tests...$(NC)"
	PYTHONPATH=$$PWD:$$PWD/.. $(PYTEST) tests/unit/ -x --tb=short --asyncio-mode=auto
	@echo "$(GREEN)Quick tests completed!$(NC)"

test-with-db: setup-db test-all ## Setup databases and run all tests
	@echo "$(GREEN)Complete test suite with databases completed!$(NC)"

# Development workflow commands
dev-setup: install-dev setup-db ## Complete development setup
	@echo "$(GREEN)Development environment setup completed!$(NC)"
	@echo "$(BLUE)You can now run: make test$(NC)"

ci-test: ## Run tests suitable for CI environment
	@echo "$(BLUE)Running CI tests...$(NC)"
	PYTHONPATH=$$PWD:$$PWD/.. $(PYTEST) tests/unit/ tests/integration/ -v --tb=short --cov=. --cov-report=xml --asyncio-mode=auto
	@echo "$(GREEN)CI tests completed!$(NC)"

# Utility commands
show-structure: ## Show current directory structure
	@echo "$(BLUE)Math Learning Directory Structure:$(NC)"
	@tree -I '__pycache__|*.pyc|.pytest_cache|htmlcov' -a || \
	find . -type d -name "__pycache__" -prune -o -type f -print | head -20

env-check: ## Check environment and dependencies
	@echo "$(BLUE)Environment Check:$(NC)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "$(BLUE)Checking key dependencies:$(NC)"
	@$(PYTHON) -c "import pytest; print('✓ pytest available')" 2>/dev/null || echo "✗ pytest not available"
	@$(PYTHON) -c "import qdrant_client; print('✓ qdrant-client available')" 2>/dev/null || echo "✗ qdrant-client not available"
	@$(PYTHON) -c "import neo4j; print('✓ neo4j available')" 2>/dev/null || echo "✗ neo4j not available"

# Example usage targets
examples: ## Show example usage commands
	@echo "$(BLUE)Example Usage:$(NC)"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make install-dev    # Install all dependencies"
	@echo "  make test          # Run standard tests"
	@echo ""
	@echo "$(GREEN)Full Testing:$(NC)"
	@echo "  make setup-db      # Start databases"
	@echo "  make test-all      # Run all tests"
	@echo ""
	@echo "$(GREEN)Development Workflow:$(NC)"
	@echo "  make dev-setup     # Complete dev environment setup"
	@echo "  make test-quick    # Quick tests during development"
	@echo "  make coverage      # Generate coverage report"
	@echo ""
	@echo "$(GREEN)Database Management:$(NC)"
	@echo "  make check-db      # Check if databases are running"
	@echo "  make clean-db      # Clean up database containers" 