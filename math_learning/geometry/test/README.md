# Geometry Learning System Test Suite

This directory contains a comprehensive test suite for the Geometry Learning System, designed to verify all basic features and ensure system reliability.

## Test Structure

### Test Files

- **`conftest.py`** - Shared fixtures and test configuration
- **`test_geometry_knowledge_graph.py`** - Unit tests for GeometryKnowledgeGraph
- **`test_geometry_learning_graph.py`** - Unit tests for GeometryLearningGraph  
- **`test_geometry_gap_analyzer.py`** - Unit tests for GeometryGapAnalyzer
- **`test_geometry_recommender.py`** - Unit tests for GeometryRecommender
- **`test_geometry_learning_system.py`** - Unit and integration tests for GeometryLearningSystem
- **`test_integration.py`** - End-to-end integration tests
- **`run_tests.py`** - Test runner script with various options
- **`pytest.ini`** - Pytest configuration

### Test Categories

#### Unit Tests (`@pytest.mark.unit`)
- Test individual components in isolation
- Verify inheritance from base classes
- Test geometry-specific functionality
- Fast execution, no external dependencies

#### Integration Tests (`@pytest.mark.integration`)
- Test component interactions
- Verify data flow between components
- Test complete workflows
- May require setup/teardown

#### Slow Tests (`@pytest.mark.slow`)
- Performance tests with large datasets
- Long-running scenarios
- Stress testing
- Optional for quick development cycles

## Test Coverage

### GeometryKnowledgeGraph Tests
- ✅ Inheritance from base KnowledgeGraph
- ✅ Exercise loading and parsing
- ✅ Geometry concept initialization
- ✅ Prerequisite relationship creation
- ✅ Grade level categorization
- ✅ Difficulty estimation
- ✅ Visual/construction/spatial reasoning concept identification
- ✅ Error handling for invalid data
- ✅ Performance with large datasets

### GeometryLearningGraph Tests
- ✅ Inheritance from base LearningGraph
- ✅ Visual learning preference tracking (0.0-1.0)
- ✅ Spatial reasoning score assessment
- ✅ Construction familiarity tracking
- ✅ Geometry exercise recording with metadata
- ✅ Visual learning effectiveness analysis
- ✅ Construction performance analysis
- ✅ Learning style identification
- ✅ Performance analysis by exercise type
- ✅ Spatial reasoning progression tracking
- ✅ Learning velocity calculation

### GeometryGapAnalyzer Tests
- ✅ Inheritance from base GapAnalyzer
- ✅ Enhanced gap detection with geometry attributes
- ✅ Visual component analysis
- ✅ Spatial reasoning gap identification
- ✅ Construction skills assessment
- ✅ Geometry category classification
- ✅ Learning strategy recommendations
- ✅ Prerequisite gap analysis
- ✅ Gap priority scoring
- ✅ Adaptive threshold adjustment

### GeometryRecommender Tests
- ✅ Inheritance from base Recommender
- ✅ Geometry-specific exercise recommendations
- ✅ Learning style matching (visual, kinesthetic, analytical)
- ✅ Visual aids recommendations
- ✅ Construction tool recommendations
- ✅ Spatial reasoning level assessment
- ✅ Personalized learning path generation
- ✅ Adaptive recommendations based on performance
- ✅ Multi-modal learning approaches
- ✅ Real-world application recommendations

### GeometryLearningSystem Tests
- ✅ Complete system initialization
- ✅ Student session creation and management
- ✅ Gap analysis integration
- ✅ Exercise recommendation integration
- ✅ Learning path generation
- ✅ Exercise recording and progress tracking
- ✅ Visual learning support
- ✅ Construction learning support
- ✅ Spatial reasoning assessment
- ✅ System analytics and reporting
- ✅ Session persistence (save/load)

### Integration Tests
- ✅ Component initialization integration
- ✅ Knowledge graph ↔ Learning graph integration
- ✅ Gap analyzer integration with all components
- ✅ Recommender integration with all components
- ✅ Complete system workflow testing
- ✅ Data flow verification
- ✅ Adaptive learning behavior
- ✅ Visual learning features integration
- ✅ Spatial reasoning features integration
- ✅ Construction learning features integration
- ✅ Error propagation and handling
- ✅ Performance with realistic data loads

## Running Tests

### Quick Start

```bash
# Run the test runner (shows usage and runs quick tests)
python run_tests.py

# Run unit tests only
python run_tests.py --unit

# Run integration tests only
python run_tests.py --integration

# Run all tests including slow ones
python run_tests.py --all
```

### Advanced Usage

```bash
# Run with coverage reporting
python run_tests.py --coverage --html-report

# Run specific test file
python run_tests.py --file test_geometry_knowledge_graph.py

# Run specific test function
python run_tests.py --test test_inheritance

# Run tests matching pattern
python run_tests.py --test "visual_learning"

# Verbose output
python run_tests.py --unit --verbose

# Generate comprehensive test report
python -c "from run_tests import generate_test_report; generate_test_report()"
```

### Direct Pytest Usage

```bash
# Basic test run
pytest

# Run only unit tests
pytest -m unit

# Run with coverage
pytest --cov=../src --cov-report=html

# Run specific test file
pytest test_geometry_knowledge_graph.py

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Show test durations
pytest --durations=10
```

## Test Data and Fixtures

### Shared Fixtures (conftest.py)

- **`sample_geometry_exercises`** - Sample exercise data for testing
- **`sample_student_data`** - Sample student profile data
- **`mock_knowledge_graph`** - Mock knowledge graph for unit tests
- **`mock_learning_graph`** - Mock learning graph for unit tests
- **`temp_data_dir`** - Temporary directory for test files
- **`geometry_knowledge_graph`** - Real geometry knowledge graph instance
- **`geometry_learning_graph`** - Real geometry learning graph instance
- **`geometry_gap_analyzer`** - Real geometry gap analyzer instance
- **`geometry_recommender`** - Real geometry recommender instance
- **`geometry_learning_system`** - Complete system instance

### Test Data Features

- **4 sample exercises** covering basic shapes, angles, area calculation, Pythagorean theorem
- **Varied difficulty levels** (1-4) and grade levels (K-5, 6-8, 9-12)
- **Mixed visual aids and construction requirements**
- **Different spatial reasoning levels** (1-3)
- **Realistic student performance data**

## Design Verification

### Basic Features Tested

1. **Init Setup (Knowledge Graph Building)** ✅
   - Exercise loading from JSON files
   - Automatic concept extraction
   - Prerequisite relationship creation
   - Grade level categorization

2. **Personal Learning Graph** ✅
   - Student-specific learning tracking
   - Visual learning preference (0.0-1.0 scale)
   - Spatial reasoning score assessment
   - Construction familiarity tracking
   - Exercise history with geometry metadata

3. **Gap Analysis and Exercise Recommendation** ✅
   - Enhanced gap detection with geometry attributes
   - Visual component analysis
   - Spatial reasoning requirements
   - Construction skills assessment
   - Learning style matching
   - Adaptive recommendations

### Geometry-Specific Features

- **Visual Learning Support** - Visual aids recommendations, effectiveness analysis
- **Spatial Reasoning Assessment** - Progressive difficulty, skill tracking
- **Construction Learning** - Tool familiarity, construction exercise support
- **Learning Style Adaptation** - Visual, kinesthetic, analytical, balanced
- **Real-World Applications** - Practical geometry applications

## Test Quality Assurance

### Error Handling
- Invalid JSON files
- Missing exercise data
- Invalid student operations
- Boundary value testing
- Exception propagation

### Performance Testing
- Large exercise datasets (500+ exercises)
- Multiple concurrent students (20+ students)
- Memory usage optimization
- Response time verification

### Data Integrity
- Exercise metadata preservation
- Student progress consistency
- Recommendation accuracy
- Gap analysis reliability

## Development Workflow

### Before Committing
```bash
# Run quick tests
python run_tests.py --unit

# Check specific component
python run_tests.py --file test_geometry_knowledge_graph.py
```

### Before Releases
```bash
# Run comprehensive test suite
python run_tests.py --all --coverage

# Generate test report
python -c "from run_tests import generate_test_report; generate_test_report()"
```

### Debugging Tests
```bash
# Run single test with verbose output
pytest test_geometry_knowledge_graph.py::TestGeometryKnowledgeGraph::test_inheritance -v

# Run with pdb debugging
pytest --pdb test_file.py::test_function

# Show test output
pytest -s test_file.py
```

## Requirements

### Python Packages
```bash
pip install pytest pytest-cov
```

### Optional Packages
```bash
pip install pytest-benchmark  # For performance testing
pip install pytest-html      # For HTML test reports
pip install pytest-xdist     # For parallel test execution
```

## Test Metrics

- **Total Test Files**: 7
- **Total Test Functions**: 150+
- **Test Coverage Target**: >90%
- **Test Categories**: Unit (60%), Integration (30%), Performance (10%)
- **Average Test Runtime**: <30 seconds (excluding slow tests)

## Contributing

### Adding New Tests

1. Follow the existing test structure and naming conventions
2. Use appropriate pytest markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
3. Include docstrings explaining what the test verifies
4. Use shared fixtures from `conftest.py` when possible
5. Add error handling and edge case tests

### Test Guidelines

- **Unit tests** should test single functions/methods in isolation
- **Integration tests** should test component interactions
- **Use mocking** for external dependencies in unit tests
- **Include assertions** that verify expected behavior
- **Test both success and failure cases**
- **Keep tests independent** - no test should depend on another

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the geometry src directory is in Python path
2. **Missing Fixtures**: Check that conftest.py is properly loaded
3. **Test Failures**: Run with `-v` flag for detailed output
4. **Performance Issues**: Use `--durations=10` to identify slow tests

### Getting Help

- Check test output for specific error messages
- Use `pytest --collect-only` to verify test discovery
- Run `python run_tests.py --help` for all available options
- Review individual test files for specific component testing 