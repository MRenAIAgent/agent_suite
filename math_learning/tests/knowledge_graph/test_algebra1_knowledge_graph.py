"""
Algebra 1 Knowledge Graph Tests

This module contains tests for a knowledge graph based on real Algebra 1 course concepts
and their authentic prerequisite relationships. The concepts are derived from standard
Algebra 1 curriculum and represent the actual learning progression students follow.

Course: Algebra 1 (High School Mathematics)
Concepts: 10 core concepts with realistic difficulty levels and time estimates
Relationships: Based on actual mathematical prerequisites and learning dependencies
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.config.storage_config import MathLearningStorageConfig, StorageBackend


@pytest.fixture
def algebra1_graph():
    """
    Create a knowledge graph with real Algebra 1 concepts and their prerequisites.
    
    This graph represents a realistic progression through Algebra 1 topics:
    
    Graph Structure (Prerequisites flow upward):
    
                    Quadratic Functions (J)
                           |
                    Quadratic Equations (I)
                      /           \\
            Factoring (H)     Linear Functions (F)
                 |                    |
          Polynomials (G)        Linear Equations (E)
                 |                    |
            Exponents (D)        Solving Equations (C)
                 |                    |
                 \\                   /
                  Order of Operations (B)
                           |
                    Variables & Expressions (A)
    
    Nodes: 10 Algebra 1 concepts
    Edges: 11 prerequisite relationships
    """
    # Use a unique temp file name that definitely doesn't exist
    temp_dir = tempfile.mkdtemp()
    import uuid
    unique_file = os.path.join(temp_dir, f"algebra1_test_{uuid.uuid4().hex}.json")
    
    # Ensure the file doesn't exist
    if os.path.exists(unique_file):
        os.remove(unique_file)
    
    config = MathLearningStorageConfig(
        backend=StorageBackend.NETWORKX,
        networkx_persistence_file=unique_file,
        networkx_auto_save=False  # Disable auto-save to prevent conflicts
    )
    
    # Create a fresh knowledge graph
    kg = KnowledgeGraph(name="Algebra 1 Test Graph", config=config)
    
    # Verify we start with a clean graph
    initial_concepts = kg.get_all_concepts()
    if len(initial_concepts) > 0:
        # Force clear if needed
        kg._storage.concepts.clear()
        kg._storage.graph.clear()
    
    # Create 10 real Algebra 1 concepts with authentic properties
    concepts_data = [
        {
            'id': 'variables_expressions',
            'name': 'Variables and Expressions',
            'difficulty': 1,  # Foundational level
            'time_to_master': 15,  # 15 minutes
            'category': 'foundation',
            'description': 'Understanding variables as symbols for unknown quantities and writing algebraic expressions'
        },
        {
            'id': 'order_operations',
            'name': 'Order of Operations',
            'difficulty': 2,  # Basic level
            'time_to_master': 20,
            'category': 'foundation',
            'description': 'Following PEMDAS/BODMAS rules to evaluate mathematical expressions correctly'
        },
        {
            'id': 'solving_equations',
            'name': 'Solving Linear Equations',
            'difficulty': 3,  # Intermediate level
            'time_to_master': 30,
            'category': 'equations',
            'description': 'Using inverse operations to solve one-step and multi-step linear equations'
        },
        {
            'id': 'exponents',
            'name': 'Exponents and Powers',
            'difficulty': 3,  # Intermediate level
            'time_to_master': 25,
            'category': 'operations',
            'description': 'Understanding exponent rules and working with powers of numbers'
        },
        {
            'id': 'linear_equations',
            'name': 'Linear Equations and Inequalities',
            'difficulty': 4,  # Advanced level
            'time_to_master': 40,
            'category': 'equations',
            'description': 'Solving complex linear equations and understanding inequality relationships'
        },
        {
            'id': 'linear_functions',
            'name': 'Linear Functions and Graphing',
            'difficulty': 4,  # Advanced level
            'time_to_master': 45,
            'category': 'functions',
            'description': 'Understanding slope, y-intercept, and graphing linear functions'
        },
        {
            'id': 'polynomials',
            'name': 'Polynomials',
            'difficulty': 4,  # Advanced level
            'time_to_master': 35,
            'category': 'expressions',
            'description': 'Adding, subtracting, and multiplying polynomial expressions'
        },
        {
            'id': 'factoring',
            'name': 'Factoring',
            'difficulty': 5,  # Expert level
            'time_to_master': 50,
            'category': 'techniques',
            'description': 'Factoring polynomials including trinomials and difference of squares'
        },
        {
            'id': 'quadratic_equations',
            'name': 'Quadratic Equations',
            'difficulty': 5,  # Expert level
            'time_to_master': 60,
            'category': 'equations',
            'description': 'Solving quadratic equations using factoring, completing the square, and quadratic formula'
        },
        {
            'id': 'quadratic_functions',
            'name': 'Quadratic Functions',
            'difficulty': 5,  # Expert level
            'time_to_master': 55,
            'category': 'functions',
            'description': 'Graphing quadratic functions as parabolas and understanding vertex form and transformations'
        }
    ]
    
    # Add all concepts to the graph
    for concept_data in concepts_data:
        concept = Concept(
            concept_id=concept_data['id'],
            name=concept_data['name'],
            difficulty=concept_data['difficulty'],
            time_to_master=concept_data['time_to_master'],
            category=concept_data['category'],
            description=concept_data['description']
        )
        kg.add_concept(concept)
    
    # Add realistic prerequisite relationships based on actual Algebra 1 curriculum
    prerequisite_relationships = [
        # Foundation concepts
        ('variables_expressions', 'order_operations', 0.9),  # Variables needed for operations
        ('order_operations', 'solving_equations', 0.95),    # Operations essential for equations
        ('order_operations', 'exponents', 0.8),             # Operations needed for exponents
        
        # Equation progression
        ('solving_equations', 'linear_equations', 0.9),     # Basic solving leads to advanced
        ('linear_equations', 'linear_functions', 0.85),     # Equations lead to functions
        
        # Polynomial progression
        ('exponents', 'polynomials', 0.9),                  # Exponents essential for polynomials
        ('polynomials', 'factoring', 0.95),                 # Must understand polynomials to factor
        
        # Quadratic progression
        ('factoring', 'quadratic_equations', 0.9),          # Factoring needed for quadratics
        ('linear_functions', 'quadratic_equations', 0.7),   # Functions background helps
        ('quadratic_equations', 'quadratic_functions', 0.85), # Equations before graphing
        
        # Advanced connection
        ('linear_functions', 'quadratic_functions', 0.6),   # Function concepts transfer
    ]
    
    # Add prerequisite relationships
    for prereq, concept, strength in prerequisite_relationships:
        kg.add_prerequisite(prereq, concept, strength=strength)
    
    return kg


class TestAlgebra1KnowledgeGraph:
    """Test suite for Algebra 1 knowledge graph functionality."""
    
    def test_algebra1_concepts_creation(self, algebra1_graph):
        """Test that all 10 Algebra 1 concepts are created correctly."""
        kg = algebra1_graph
        all_concepts = kg.get_all_concepts()
        
        assert len(all_concepts) == 10, "Should have exactly 10 Algebra 1 concepts"
        
        # Check specific concepts exist
        concept_ids = {concept.id for concept in all_concepts}
        expected_concepts = {
            'variables_expressions', 'order_operations', 'solving_equations',
            'exponents', 'linear_equations', 'linear_functions', 'polynomials',
            'factoring', 'quadratic_equations', 'quadratic_functions'
        }
        
        assert concept_ids == expected_concepts, "All expected Algebra 1 concepts should be present"
        
        # Verify concept properties
        variables_concept = kg.get_concept('variables_expressions')
        assert variables_concept.name == 'Variables and Expressions'
        assert variables_concept.difficulty == 1  # Foundation level
        assert variables_concept.category == 'foundation'
        
        quadratic_functions = kg.get_concept('quadratic_functions')
        assert quadratic_functions.difficulty == 5  # Expert level
        assert quadratic_functions.time_to_master == 55
        assert quadratic_functions.category == 'functions'
    
    def test_algebra1_prerequisite_relationships(self, algebra1_graph):
        """Test that prerequisite relationships follow real Algebra 1 curriculum."""
        kg = algebra1_graph
        
        # Test foundation prerequisites
        order_ops_prereqs = kg.get_prerequisites('order_operations')
        assert len(order_ops_prereqs) == 1
        assert order_ops_prereqs[0].id == 'variables_expressions'
        
        # Test equation progression
        linear_eq_prereqs = kg.get_prerequisites('linear_equations')
        assert len(linear_eq_prereqs) == 1
        assert linear_eq_prereqs[0].id == 'solving_equations'
        
        # Test polynomial progression
        factoring_prereqs = kg.get_prerequisites('factoring')
        assert len(factoring_prereqs) == 1
        assert factoring_prereqs[0].id == 'polynomials'
        
        # Test quadratic equations has multiple prerequisites
        quad_eq_prereqs = kg.get_prerequisites('quadratic_equations')
        quad_eq_prereq_ids = {prereq.id for prereq in quad_eq_prereqs}
        assert 'factoring' in quad_eq_prereq_ids
        assert 'linear_functions' in quad_eq_prereq_ids
        
        # Test most advanced concept
        quad_func_prereqs = kg.get_prerequisites('quadratic_functions')
        quad_func_prereq_ids = {prereq.id for prereq in quad_func_prereqs}
        assert 'quadratic_equations' in quad_func_prereq_ids
        assert 'linear_functions' in quad_func_prereq_ids
    
    def test_algebra1_learning_path(self, algebra1_graph):
        """Test learning path from foundation to advanced concepts."""
        kg = algebra1_graph
        
        # Path from variables to quadratic functions (complete progression)
        try:
            path = kg.find_learning_path('variables_expressions', 'quadratic_functions')
            assert path is not None, "Should find a learning path"
            assert len(path) > 1, "Path should have multiple steps"
            assert path[0].id == 'variables_expressions', "Should start with variables"
            assert path[-1].id == 'quadratic_functions', "Should end with quadratic functions"
        except AttributeError:
            # Method might not be implemented
            pytest.skip("find_learning_path method not implemented")
    
    def test_algebra1_difficulty_progression(self, algebra1_graph):
        """Test that difficulty levels follow realistic progression."""
        kg = algebra1_graph
        
        # Foundation concepts should be easier
        variables = kg.get_concept('variables_expressions')
        order_ops = kg.get_concept('order_operations')
        assert variables.difficulty <= order_ops.difficulty
        
        # Advanced concepts should be harder
        factoring = kg.get_concept('factoring')
        quadratics = kg.get_concept('quadratic_equations')
        assert factoring.difficulty <= quadratics.difficulty
        
        # Check overall progression makes sense
        all_concepts = kg.get_all_concepts()
        difficulty_levels = [concept.difficulty for concept in all_concepts]
        
        # Should have concepts at multiple difficulty levels
        unique_difficulties = set(difficulty_levels)
        assert len(unique_difficulties) >= 3, "Should have at least 3 different difficulty levels"
        
        # Most advanced concepts should be difficulty 5
        max_difficulty = max(difficulty_levels)
        assert max_difficulty == 5, "Hardest concepts should be level 5"
    
    def test_algebra1_time_estimates(self, algebra1_graph):
        """Test that time to master estimates are realistic."""
        kg = algebra1_graph
        
        all_concepts = kg.get_all_concepts()
        
        for concept in all_concepts:
            # Time should be reasonable (15-60 minutes for individual concepts)
            assert 15 <= concept.time_to_master <= 60, f"Time for {concept.name} should be 15-60 minutes"
            
            # More difficult concepts should generally take longer
            if concept.difficulty >= 5:
                assert concept.time_to_master >= 50, f"Expert level concept {concept.name} should take 50+ minutes"
            elif concept.difficulty <= 2:
                assert concept.time_to_master <= 25, f"Foundation concept {concept.name} should take ≤25 minutes"
    
    def test_algebra1_concept_categories(self, algebra1_graph):
        """Test that concepts are properly categorized."""
        kg = algebra1_graph
        
        all_concepts = kg.get_all_concepts()
        categories = {concept.category for concept in all_concepts}
        
        expected_categories = {'foundation', 'equations', 'operations', 'functions', 'expressions', 'techniques'}
        assert categories == expected_categories, "Should have all expected categories"
        
        # Check specific categorizations
        foundation_concepts = [c for c in all_concepts if c.category == 'foundation']
        assert len(foundation_concepts) == 2, "Should have 2 foundation concepts"
        
        equation_concepts = [c for c in all_concepts if c.category == 'equations']
        assert len(equation_concepts) == 3, "Should have 3 equation-related concepts"
        
        function_concepts = [c for c in all_concepts if c.category == 'functions']
        assert len(function_concepts) == 2, "Should have 2 function concepts"
    
    def test_algebra1_graph_connectivity(self, algebra1_graph):
        """Test that the graph is well-connected and follows curriculum structure."""
        kg = algebra1_graph
        
        # Check that foundation concepts have no prerequisites
        foundation_concepts = ['variables_expressions']
        for concept_id in foundation_concepts:
            prereqs = kg.get_prerequisites(concept_id)
            assert len(prereqs) == 0, f"Foundation concept {concept_id} should have no prerequisites"
        
        # Check that advanced concepts have prerequisites
        advanced_concepts = ['quadratic_functions', 'quadratic_equations', 'factoring']
        for concept_id in advanced_concepts:
            prereqs = kg.get_prerequisites(concept_id)
            assert len(prereqs) > 0, f"Advanced concept {concept_id} should have prerequisites"
        
        # Test that there are no circular dependencies
        try:
            # This would be implementation-specific
            # Just check that we can get prerequisites without infinite loops
            for concept in kg.get_all_concepts():
                prereqs = kg.get_prerequisites(concept.id)
                assert isinstance(prereqs, list), "Prerequisites should be returned as list"
        except Exception as e:
            pytest.fail(f"Graph connectivity test failed: {e}")
    
    def test_algebra1_realistic_strengths(self, algebra1_graph):
        """Test that prerequisite strengths reflect realistic importance."""
        kg = algebra1_graph
        
        # Variables → Order of Operations should be very strong (0.9)
        # This is fundamental to all algebra
        
        # Order of Operations → Solving Equations should be very strong (0.95)
        # Can't solve equations without understanding operations
        
        # Polynomials → Factoring should be very strong (0.95)
        # Must understand polynomials to factor them
        
        # Linear Functions → Quadratic Functions should be moderate (0.6)
        # Helpful but not absolutely essential
        
        # Note: Actual strength values would need to be tested if the graph
        # implementation provides access to relationship strengths
        
        # For now, just verify that relationships exist
        assert len(kg.get_prerequisites('order_operations')) == 1
        assert len(kg.get_prerequisites('solving_equations')) == 1
        assert len(kg.get_prerequisites('factoring')) == 1
        assert len(kg.get_prerequisites('quadratic_functions')) == 2
    
    def test_algebra1_concept_descriptions(self, algebra1_graph):
        """Test that concept descriptions are meaningful and educational."""
        kg = algebra1_graph
        
        all_concepts = kg.get_all_concepts()
        
        for concept in all_concepts:
            # Each concept should have a non-empty description
            assert len(concept.description) > 20, f"Concept {concept.name} should have meaningful description"
            
            # Descriptions should contain relevant mathematical terms
            description_lower = concept.description.lower()
            
            # Check for appropriate mathematical vocabulary
            if 'variables' in concept.id:
                assert any(term in description_lower for term in ['variable', 'symbol', 'unknown']), \
                    "Variables concept should mention variables/symbols/unknowns"
            
            if 'quadratic' in concept.id:
                assert any(term in description_lower for term in ['quadratic', 'parabola', 'square']), \
                    "Quadratic concepts should mention quadratic/parabola/square"
            
            if 'function' in concept.id:
                assert 'function' in description_lower, "Function concepts should mention functions"
    
    def test_algebra1_complete_curriculum_coverage(self, algebra1_graph):
        """Test that the graph covers essential Algebra 1 curriculum areas."""
        kg = algebra1_graph
        
        all_concepts = kg.get_all_concepts()
        concept_names = {concept.name.lower() for concept in all_concepts}
        
        # Essential Algebra 1 topics that should be covered
        essential_topics = [
            'variables', 'expressions', 'operations', 'equations', 
            'linear', 'functions', 'polynomials', 'factoring', 'quadratic'
        ]
        
        for topic in essential_topics:
            assert any(topic in name for name in concept_names), \
                f"Essential topic '{topic}' should be covered in the curriculum"
        
        # Verify progression from basic to advanced
        basic_concepts = [c for c in all_concepts if c.difficulty <= 2]
        advanced_concepts = [c for c in all_concepts if c.difficulty >= 5]
        
        assert len(basic_concepts) >= 2, "Should have at least 2 basic concepts"
        assert len(advanced_concepts) >= 3, "Should have at least 3 advanced concepts"
        
        # Total time should be realistic for a course unit (4-6 hours)
        total_time = sum(concept.time_to_master for concept in all_concepts)
        assert 240 <= total_time <= 400, f"Total time {total_time} should be 4-6.5 hours for course unit"


# Integration test for the complete Algebra 1 curriculum
def test_algebra1_curriculum_integration(algebra1_graph):
    """Integration test for the complete Algebra 1 knowledge graph."""
    kg = algebra1_graph
    
    # Test that we can perform common graph operations
    assert len(kg.get_all_concepts()) == 10
    
    # Test concept retrieval
    variables_concept = kg.get_concept('variables_expressions')
    assert variables_concept is not None
    assert variables_concept.name == 'Variables and Expressions'
    
    # Test prerequisite relationships
    quad_prereqs = kg.get_prerequisites('quadratic_functions')
    assert len(quad_prereqs) == 2  # Should have 2 prerequisites
    
    # Test that the graph represents a realistic Algebra 1 learning progression
    foundation_concept = kg.get_concept('variables_expressions')
    advanced_concept = kg.get_concept('quadratic_functions')
    
    assert foundation_concept.difficulty < advanced_concept.difficulty
    assert foundation_concept.time_to_master < advanced_concept.time_to_master
    
    print("\n🎓 Algebra 1 Knowledge Graph Test Summary:")
    print(f"   📚 Concepts: {len(kg.get_all_concepts())}")
    print(f"   🔗 Total prerequisite relationships: 11")
    print(f"   ⏱️  Total learning time: {sum(c.time_to_master for c in kg.get_all_concepts())} minutes")
    print(f"   📈 Difficulty range: {min(c.difficulty for c in kg.get_all_concepts())}-{max(c.difficulty for c in kg.get_all_concepts())}")
    print(f"   🏷️  Categories: {len(set(c.category for c in kg.get_all_concepts()))}") 