# Real Database Integration Tests

This directory contains comprehensive tests for verifying that the math learning system actually stores data in real Qdrant (vector database) and Neo4j (graph database) instances, rather than just using in-memory storage.

## 🎯 What These Tests Verify

### Qdrant Vector Storage
- ✅ Vector embedding storage and retrieval
- ✅ Semantic similarity search
- ✅ Metadata filtering and queries
- ✅ Batch operations
- ✅ Collection management

### Neo4j Graph Storage  
- ✅ Entity storage and retrieval
- ✅ Relationship creation and queries
- ✅ Custom Cypher query execution
- ✅ Graph traversal operations
- ✅ Entity updates and deletions

### Integration Scenarios
- ✅ Cross-system data linking
- ✅ Coordinated storage operations
- ✅ Real-world usage patterns

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Make the setup script executable (if not already)
chmod +x setup_and_test_real_dbs.sh

# Start databases and run all tests
./setup_and_test_real_dbs.sh

# Or just run specific commands:
./setup_and_test_real_dbs.sh start    # Start databases and run tests
./setup_and_test_real_dbs.sh test     # Run tests only (assumes DBs running)
./setup_and_test_real_dbs.sh status   # Check database status
./setup_and_test_real_dbs.sh stop     # Stop databases
./setup_and_test_real_dbs.sh clean    # Stop and remove all data
```

### Option 2: Manual Setup

1. **Start the databases:**
   ```bash
   docker-compose up -d
   ```

2. **Wait for databases to be ready:**
   ```bash
   # Check Qdrant
   curl http://localhost:6333/health
   
   # Check Neo4j (wait ~30-60 seconds for first startup)
   docker exec neo4j_math_learning cypher-shell -u neo4j -p password "RETURN 1"
   ```

3. **Install Python dependencies:**
   ```bash
   pip install qdrant-client neo4j sentence-transformers
   ```

4. **Run the tests:**
   ```bash
   # Direct storage backend tests (recommended)
   python test_storage_backends.py
   
   # Full RAG service integration tests
   python test_real_databases.py
   ```

## 📁 Test Files

### `test_storage_backends.py`
**Direct storage backend testing** - Tests the storage implementations directly without going through the RAG service layer.

**What it tests:**
- Direct Qdrant vector operations
- Direct Neo4j graph operations  
- Cross-system integration scenarios
- Error handling and cleanup

**Run with:** `python test_storage_backends.py`

### `test_real_databases.py`
**Full RAG service integration** - Tests the complete math learning system using real databases through the RAG service layer.

**What it tests:**
- Complete RAG service with real backends
- Math concept storage and retrieval
- Learning graph construction
- End-to-end workflows

**Run with:** `python test_real_databases.py`

### `docker-compose.yml`
Docker Compose configuration for running Qdrant and Neo4j locally.

**Services:**
- **Qdrant**: Vector database on ports 6333 (HTTP) and 6334 (gRPC)
- **Neo4j**: Graph database on ports 7474 (HTTP) and 7687 (Bolt)

### `setup_and_test_real_dbs.sh`
Automated setup script that handles database startup, dependency installation, and test execution.

## 🔧 Database Configuration

### Qdrant Configuration
- **Host:** localhost
- **Port:** 6333 (HTTP), 6334 (gRPC)
- **Collections:** Auto-created during tests
- **Vector Size:** 384 (for sentence-transformers/all-MiniLM-L6-v2)

### Neo4j Configuration
- **Host:** localhost
- **Ports:** 7474 (HTTP), 7687 (Bolt)
- **Username:** neo4j
- **Password:** password
- **Database:** neo4j (default)

## 🧪 Test Scenarios

### Basic Storage Tests
1. **Vector Storage**: Store math concept embeddings in Qdrant
2. **Graph Storage**: Store concept entities and relationships in Neo4j
3. **Retrieval**: Query both systems for stored data
4. **Updates**: Modify existing data
5. **Cleanup**: Proper data cleanup after tests

### Advanced Integration Tests
1. **Cross-System Linking**: Store related data in both systems with references
2. **Semantic Search**: Find relevant content using vector similarity
3. **Graph Traversal**: Navigate concept prerequisites and relationships
4. **Metadata Filtering**: Filter results by difficulty, grade level, etc.
5. **Batch Operations**: Handle multiple operations efficiently

### Real-World Scenarios
1. **Math Concept Learning**: Store and retrieve algebra concepts
2. **Prerequisite Mapping**: Build and query prerequisite relationships
3. **Content Recommendation**: Find related learning materials
4. **Progress Tracking**: Update and query learning progress

## 📊 Expected Output

### Successful Test Run
```
🧪 STORAGE BACKEND DIRECT TESTS
   Testing Qdrant and Neo4j storage backends directly
======================================================================

🔍 Checking database availability...
  ✅ Qdrant is available
  ✅ Neo4j is available

✅ All databases are available!

🔍 Testing Qdrant Vector Storage Backend...
  ✅ Qdrant storage initialized
  
  📝 Testing vector storage...
    ✅ Stored: addition (ID: 1)
    ✅ Stored: multiplication (ID: 2)
    ✅ Stored: fractions (ID: 3)
  
  🔍 Testing vector search...
    Query: 'How do I combine numbers?'
      1. addition (score: 0.842)
      2. multiplication (score: 0.651)
  
  🎯 Testing metadata filtering...
    Easy concepts (difficulty < 0.5): 2 found
      - addition (difficulty: 0.2)
      - multiplication (difficulty: 0.4)
  
  ✅ Qdrant test completed successfully

🕸️  Testing Neo4j Graph Storage Backend...
  ✅ Neo4j storage initialized
  
  📝 Testing entity storage...
    ✅ Stored: Addition (ID: 123)
    ✅ Stored: Multiplication (ID: 124)
    ✅ Stored: Fractions (ID: 125)
  
  🔗 Testing relationship storage...
    ✅ Created relationship (ID: 456)
    ✅ Created relationship (ID: 457)
  
  ✅ Neo4j test completed successfully

🔄 Testing Qdrant + Neo4j Integration...
  ✅ Both storages initialized
  
  📚 Storing algebra concept in both systems...
    ✅ Stored in Neo4j (ID: 789)
    ✅ Stored in Qdrant: definition (ID: 4)
    ✅ Stored in Qdrant: method (ID: 5)
    ✅ Stored in Qdrant: example (ID: 6)
  
  ✅ Integration test completed successfully

======================================================================
📊 TEST RESULTS SUMMARY
======================================================================
  ✅ PASSED - Qdrant
  ✅ PASSED - Neo4j  
  ✅ PASSED - Integration

🎯 Overall Success Rate: 100.0% (3/3)
🎉 EXCELLENT! Storage backends are working perfectly!

💡 What was tested:
  • Direct Qdrant vector storage and search
  • Direct Neo4j graph storage and queries
  • Cross-system integration scenarios
  • Metadata filtering and custom queries
  • Proper cleanup and connection management
```

## 🐛 Troubleshooting

### Database Connection Issues

**Qdrant not available:**
```bash
# Check if container is running
docker ps | grep qdrant

# Check logs
docker-compose logs qdrant

# Restart Qdrant
docker-compose restart qdrant
```

**Neo4j not available:**
```bash
# Check if container is running  
docker ps | grep neo4j

# Check logs
docker-compose logs neo4j

# Neo4j takes longer to start (30-60 seconds)
# Wait and try again, or restart:
docker-compose restart neo4j
```

### Import Errors

**Missing dependencies:**
```bash
pip install qdrant-client neo4j sentence-transformers
```

**Module not found:**
```bash
# Make sure you're in the math_learning directory
cd math_learning

# Check Python path
python -c "import sys; print(sys.path)"
```

### Test Failures

**Vector size mismatch:**
- Ensure you're using the correct embedding model
- Default tests use 384-dimensional vectors (all-MiniLM-L6-v2)

**Authentication errors:**
- Neo4j default credentials: neo4j/password
- Check docker-compose.yml for any custom settings

**Port conflicts:**
- Qdrant: 6333, 6334
- Neo4j: 7474, 7687
- Make sure these ports aren't used by other services

## 🔄 Continuous Integration

To run these tests in CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Start databases
  run: docker-compose up -d
  
- name: Wait for databases
  run: |
    timeout 60 bash -c 'until curl -f http://localhost:6333/health; do sleep 1; done'
    timeout 120 bash -c 'until docker exec neo4j_math_learning cypher-shell -u neo4j -p password "RETURN 1"; do sleep 1; done'
    
- name: Install dependencies
  run: pip install qdrant-client neo4j sentence-transformers
  
- name: Run storage tests
  run: python test_storage_backends.py
  
- name: Cleanup
  run: docker-compose down --volumes
```

## 📈 Performance Notes

- **First run**: Slower due to Docker image downloads and Neo4j initialization
- **Subsequent runs**: Much faster with cached images and data
- **Test duration**: ~30-60 seconds for complete test suite
- **Resource usage**: ~2GB RAM for both databases

## 🔒 Security Notes

- Default credentials are for testing only
- In production, use strong passwords and proper authentication
- Consider network isolation for database containers
- Regular security updates for database images

## 🤝 Contributing

When adding new tests:

1. Follow the existing test structure
2. Include proper cleanup in test teardown
3. Add descriptive output messages
4. Test both success and failure scenarios
5. Update this README with new test descriptions 