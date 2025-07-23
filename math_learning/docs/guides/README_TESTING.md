# Math Learning System - Testing Guide

## 🎯 Overview

The Math Learning System includes a comprehensive testing infrastructure with multiple layers of testing, from unit tests to full integration tests with real database backends. This guide provides complete instructions for running and understanding all test suites.

## 📊 Test Coverage Summary

- **Total Test Files**: 50+ test files across multiple categories
- **Test Lines of Code**: 15,000+ lines of comprehensive testing
- **Success Rates**: 90-100% across all test suites
- **Database Integration**: Real Neo4j and Qdrant testing
- **Test Categories**: Unit, Integration, Performance, End-to-End, Real Backend

## 🏗️ Testing Architecture

### Test Categories

1. **Unit Tests** ⚡ - Fast, isolated component testing
2. **Integration Tests** 🔗 - Component interaction validation  
3. **System Tests** 🎯 - End-to-end workflow verification
4. **Real Backend Tests** 🗄️ - Actual database integration
5. **Performance Tests** 📈 - Scalability and stress testing

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install pytest pytest-asyncio pytest-cov

# For real database tests, install additional dependencies
pip install qdrant-client neo4j sentence-transformers

# Or install all test dependencies at once
pip install -r requirements_test_real_dbs.txt
```

### Run All Tests (Recommended)

```bash
# Navigate to math_learning directory
cd math_learning

# Run comprehensive test suite
python run_all_tests.py

# Expected output:
# ✅ Full Integration Test PASSED (1.49s, 20 scenarios)
# ✅ Multi-User Scenarios Test PASSED (1.26s, 10 user types) 
# ✅ Edge Cases Test PASSED (1.22s, 10 edge cases)
# 🎯 Overall Success Rate: 100.0% (3/3)
```

---

## 📋 Detailed Test Instructions

### 1. Unit Tests

#### Core Learning Graph Tests
```bash
# Run core functionality tests (no external dependencies)
cd math_learning/tests
python test_core_personal_learning.py

# Expected output:
# Testing LearningGraph basic functionality...
# ✅ Initialization test passed
# ✅ Mastery setting test passed
# ✅ Exercise recording test passed
# 🎉 All LearningGraph basic tests passed!
```

#### Geometry System Tests
```bash
# Run geometry-specific unit tests
cd math_learning/geometry/test
python run_tests.py --unit

# Run specific geometry component tests
python run_tests.py --file test_geometry_knowledge_graph.py
python run_tests.py --file test_geometry_learning_graph.py

# Expected output:
# 📊 TEST RESULTS:
# Total Test Functions: 150+
# Passed: 150+ ✅
# Failed: 0 ❌
# Success Rate: 100.0%
```

#### Run Unit Tests with Coverage
```bash
cd math_learning/geometry/test
python run_tests.py --unit --coverage --html-report

# Generates HTML coverage report in htmlcov/index.html
```

### 2. Integration Tests

#### RAG Backend Integration Tests
```bash
# Run integration tests with memory backends
cd math_learning/tests
python -m pytest test_rag_backend_integration_sync.py -v

# Run async integration tests
python -m pytest test_rag_backend_integration.py -v

# Expected output:
# test_entity_storage_and_retrieval PASSED
# test_relationship_storage_and_retrieval PASSED  
# test_complete_learning_scenario PASSED
# ✅ 13 tests passing, 0 failing
```

#### Multi-User Integration Tests
```bash
cd math_learning
python test_multi_user_scenarios.py

# Tests multiple student profiles:
# • Alex - Foundation Strong, Fraction Struggles
# • Blake - Weak Foundation, Jumping Ahead  
# • Casey - Inconsistent Performance
# • Dana - Advanced with Specific Gaps
# • And 6 more realistic student profiles
```

### 3. Real Database Backend Tests

#### Prerequisites: Start Real Databases

**Option 1: Automated Setup (Recommended)**
```bash
cd math_learning

# Make script executable
chmod +x setup_and_test_real_dbs.sh

# Start databases and run all tests
./setup_and_test_real_dbs.sh start

# Other commands:
./setup_and_test_real_dbs.sh test     # Run tests only
./setup_and_test_real_dbs.sh status   # Check database status
./setup_and_test_real_dbs.sh stop     # Stop databases
./setup_and_test_real_dbs.sh clean    # Stop and remove all data
```

**Option 2: Manual Setup**
```bash
cd math_learning

# Start databases with Docker Compose
docker-compose up -d

# Wait for databases to be ready (30-60 seconds for Neo4j)
curl http://localhost:6333/health  # Check Qdrant
docker exec neo4j_math_learning cypher-shell -u neo4j -p password "RETURN 1"  # Check Neo4j
```

#### Run Real Database Tests

```bash
# Test storage backends directly
python test_storage_backends.py

# Expected output:
# 🔍 Testing Qdrant Vector Storage Backend...
#   ✅ Stored: addition (ID: 1)
#   ✅ Stored: multiplication (ID: 2)
#   🔍 Testing vector search...
#     Query: 'How do I combine numbers?'
#       1. addition (score: 0.842)
# 🕸️  Testing Neo4j Graph Storage Backend...
#   ✅ Stored: Addition (ID: 123)
#   🔗 Testing relationship storage...
#   ✅ Created relationship (ID: 456)
# 🎯 Overall Success Rate: 100.0% (3/3)

# Test full RAG service integration
python test_real_databases.py

# Test specific real backend scenarios
python test_real_backend_integration.py
```

#### Database Configuration

**Qdrant Configuration:**
- Host: localhost:6333 (HTTP), localhost:6334 (gRPC)
- Collections: Auto-created during tests
- Vector Size: 384 (sentence-transformers/all-MiniLM-L6-v2)

**Neo4j Configuration:**
- Host: localhost:7687 (Bolt), localhost:7474 (HTTP)
- Username: neo4j
- Password: password
- Database: neo4j (default)

### 4. System-Level and Performance Tests

#### Comprehensive System Tests
```bash
cd math_learning/testing

# Run all system-level tests
python run_all_tests.py

# Run specific test categories
python real_math_problems_test.py      # Real educational content
python advanced_test_scenarios.py      # Complex scenarios
python performance_test.py             # Performance benchmarks
python edge_case_test.py              # Error handling
python integration_test.py            # End-to-end workflows

# Expected output:
# 🧮 K-12 ALGEBRA DIAGNOSTIC SYSTEM - COMPREHENSIVE VALIDATION
# ✅ Real Educational Math Problems Test COMPLETED (2.1s)
# ✅ Advanced Scenarios & Edge Cases COMPLETED (1.8s)
# ✅ Performance & Scalability Testing COMPLETED (3.2s)
# 🏆 Overall Success Rate: 100% (5/5)
```

#### Specific Performance Tests
```bash
# Test with large datasets
python testing/performance_test.py

# Test classroom scenarios (25+ concurrent students)
python testing/integration_test.py

# Stress test with 500+ exercises
python testing/comprehensive_system_level_test.py
```

### 5. Specialized Tests

#### Graphiti + Neo4j Integration
```bash
# Test advanced graph intelligence features
python test_graphiti_neo4j_math_learning.py

# Note: Requires graphiti package
pip install graphiti
```

#### Database Usage Verification
```bash
# Verify actual database usage (not just mocks)
python test_connections.py

# Compare memory vs real backends
python test_backend_comparison.py
```

---

## 🔧 Test Configuration and Options

### Environment Variables

```bash
# Neo4j Configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"

# Qdrant Configuration  
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"

# Test Configuration
export TEST_TIMEOUT="30"
export TEST_CLEANUP="true"
```

### Pytest Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only slow/performance tests
pytest -m slow

# Run async tests
pytest -m asyncio

# Skip real database tests
pytest -m "not real_db"
```

### Test Output Options

```bash
# Verbose output
python run_tests.py --verbose

# Generate coverage report
python run_tests.py --coverage

# Generate HTML coverage report
python run_tests.py --coverage --html-report

# Run specific test file
python run_tests.py --file test_geometry_knowledge_graph.py

# Run specific test function
python run_tests.py --test "test_mastery_calculation"
```

---

## 🐛 Troubleshooting

### Common Issues

#### Database Connection Failures

**Qdrant not available:**
```bash
# Check if container is running
docker ps | grep qdrant

# Check logs
docker-compose logs qdrant

# Restart Qdrant
docker-compose restart qdrant

# Manual health check
curl http://localhost:6333/health
```

**Neo4j not available:**
```bash
# Check if container is running
docker ps | grep neo4j

# Check logs (Neo4j takes 30-60 seconds to start)
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j

# Manual connection test
docker exec neo4j_math_learning cypher-shell -u neo4j -p password "RETURN 1"
```

#### Import Errors

**Missing dependencies:**
```bash
# Install all test dependencies
pip install -r requirements_test_real_dbs.txt

# Install specific packages
pip install pytest pytest-asyncio pytest-cov
pip install qdrant-client neo4j sentence-transformers
```

**Module not found:**
```bash
# Ensure you're in the correct directory
cd math_learning

# Check Python path
python -c "import sys; print(sys.path)"

# Add current directory to path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Test Failures

**Vector size mismatch:**
- Ensure you're using the correct embedding model
- Default tests use 384-dimensional vectors (all-MiniLM-L6-v2)
- Check vector_size configuration in test files

**Authentication errors:**
- Neo4j default credentials: neo4j/password
- Check docker-compose.yml for any custom settings
- Verify environment variables are set correctly

**Port conflicts:**
- Qdrant: 6333, 6334
- Neo4j: 7474, 7687
- Use `lsof -i :6333` to check if ports are in use

### Performance Issues

**Slow test execution:**
```bash
# Run only fast unit tests
python run_tests.py --unit

# Skip slow performance tests
pytest -m "not slow"

# Run tests in parallel (if supported)
pytest -n auto
```

**Memory issues:**
```bash
# Monitor memory usage during tests
python testing/performance_test.py --monitor-memory

# Reduce test dataset sizes
export TEST_DATASET_SIZE=small
```

---

## 📊 Test Results and Metrics

### Expected Success Rates

- **Unit Tests**: 100% success rate
- **Integration Tests**: 95-100% success rate
- **Real Database Tests**: 90-100% success rate (depends on DB availability)
- **System Tests**: 95-100% success rate
- **Performance Tests**: 90-100% success rate

### Performance Benchmarks

- **Unit Test Suite**: < 5 seconds total execution
- **Integration Tests**: < 10 seconds total execution
- **Real Database Tests**: < 30 seconds total execution
- **Full System Tests**: < 60 seconds total execution

### Coverage Targets

- **Code Coverage**: > 90% for core components
- **Feature Coverage**: 100% of documented features
- **Scenario Coverage**: Major educational use cases
- **Error Coverage**: Edge cases and error conditions

---

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Math Learning Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
          - 6334:6334
          
      neo4j:
        image: neo4j:5.15-community
        ports:
          - 7474:7474
          - 7687:7687
        env:
          NEO4J_AUTH: neo4j/password
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements_test_real_dbs.txt
    
    - name: Wait for databases
      run: |
        timeout 60 bash -c 'until curl -f http://localhost:6333/health; do sleep 1; done'
        timeout 120 bash -c 'until docker exec neo4j cypher-shell -u neo4j -p password "RETURN 1"; do sleep 1; done'
    
    - name: Run unit tests
      run: |
        cd math_learning
        python run_all_tests.py --unit
    
    - name: Run integration tests
      run: |
        cd math_learning
        python run_all_tests.py --integration
    
    - name: Run real database tests
      run: |
        cd math_learning
        python test_storage_backends.py
    
    - name: Generate coverage report
      run: |
        cd math_learning
        python run_all_tests.py --coverage --html-report
    
    - name: Upload coverage reports
      uses: actions/upload-artifact@v3
      with:
        name: coverage-report
        path: math_learning/htmlcov/
```

---

## 🤝 Contributing to Tests

### Adding New Tests

1. **Follow the existing test structure and naming conventions**
2. **Use appropriate pytest markers** (`@pytest.mark.unit`, `@pytest.mark.integration`)
3. **Include docstrings** explaining what the test verifies
4. **Use shared fixtures** from `conftest.py` when possible
5. **Add error handling and edge case tests**

### Test Guidelines

- **Unit tests** should test single functions/methods in isolation
- **Integration tests** should test component interactions
- **Use mocking** for external dependencies in unit tests
- **Include assertions** that verify expected behavior
- **Clean up** test data and resources properly

### Example Test Template

```python
import pytest
import asyncio
from unittest.mock import Mock, patch

class TestNewFeature:
    """Test cases for new feature."""
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic functionality of new feature."""
        # Arrange
        feature = NewFeature()
        
        # Act
        result = feature.do_something()
        
        # Assert
        assert result is not None
        assert result.status == "success"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_scenario(self, rag_service):
        """Test integration with RAG service."""
        # Test integration logic here
        pass
    
    @pytest.mark.slow
    def test_performance_scenario(self):
        """Test performance with large datasets."""
        # Performance test logic here
        pass
```

---

## 📚 Additional Resources

### Documentation Files

- `COMPREHENSIVE_TEST_SUMMARY.md` - Detailed test documentation
- `INTEGRATION_SUMMARY.md` - RAG integration overview  
- `README_REAL_DB_TESTS.md` - Real database testing guide
- `FINAL_STATUS_SUMMARY.md` - Complete system status

### Test Result Files

- `evaluation_summary_*.json` - Test execution results
- `htmlcov/index.html` - Coverage reports
- `logs/` - Test execution logs

### Configuration Files

- `docker-compose.yml` - Database setup
- `requirements_test_real_dbs.txt` - Test dependencies
- `pytest.ini` - Pytest configuration
- `conftest.py` - Shared test fixtures

---

## 🎯 Summary

The Math Learning System provides comprehensive testing infrastructure that ensures:

✅ **Complete Test Coverage** - Unit, Integration, System, Performance  
✅ **Real Database Testing** - Actual Neo4j and Qdrant integration  
✅ **Production Scenarios** - Real educational content and workflows  
✅ **Automated Infrastructure** - Docker-based database management  
✅ **Performance Validation** - Stress testing and scalability verification  
✅ **CI/CD Ready** - Automated testing pipelines  

This testing approach ensures the system works reliably in production environments with real databases, real student data, and real educational workflows.

For questions or issues, refer to the troubleshooting section or check the detailed test documentation files. 