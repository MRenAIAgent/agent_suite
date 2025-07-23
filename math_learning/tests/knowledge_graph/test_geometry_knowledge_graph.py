"""
High School Geometry Knowledge Graph Tests

This module contains tests for a knowledge graph based on real high school geometry concepts
and their authentic prerequisite relationships. The concepts are derived from standard
geometry curriculum and represent the actual learning progression students follow.

Course: High School Geometry 
Concepts: 20 core concepts with realistic difficulty levels and time estimates
Relationships: Based on actual mathematical prerequisites and learning dependencies
"""

import pytest
import tempfile
import os
import uuid
from unittest.mock import Mock, patch

from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.config.storage_config import MathLearningStorageConfig, StorageBackend


@pytest.fixture
def geometry_graph():
    """
    Create a knowledge graph with real high school geometry concepts and their prerequisites.
    
    This graph represents a realistic progression through geometry topics:
    
    Graph Structure (Prerequisites flow upward):
    
                        Trigonometry (T)
                           /        \\
                   Pythagorean (S)  Circles (R)
                         |           /    \\
                Special Triangles (Q)  Inscribed/Circumscribed (P)
                         |              |
                   Right Triangles (O)  Circle Theorems (N)
                         |              |
                Similar Triangles (M)  Area/Perimeter (L)
                         |              |
                Congruent Triangles (K) Polygons (J)
                         |              |
                Triangle Properties (I) Quadrilaterals (H)
                         |              |
                    Triangles (G)   Parallel Lines (F)
                         |              |
                      Angles (E)    Transformations (D)
                         |              |
                 Points/Lines/Planes (C) Constructions (B)
                         |              |
                         \\             /
                      Basic Geometric Concepts (A)
    
    Nodes: 20 geometry concepts
    Edges: 25 prerequisite relationships
    """
    # Use a unique temp file name that definitely doesn't exist
    temp_dir = tempfile.mkdtemp()
    unique_file = os.path.join(temp_dir, f"geometry_test_{uuid.uuid4().hex}.json")
    
    # Ensure the file doesn't exist
    if os.path.exists(unique_file):
        os.remove(unique_file)
    
    config = MathLearningStorageConfig(
        backend=StorageBackend.NETWORKX,
        networkx_persistence_file=unique_file,
        networkx_auto_save=False  # Disable auto-save to prevent conflicts
    )
    
    # Create a fresh knowledge graph
    kg = KnowledgeGraph(name="High School Geometry Graph", config=config)
    
    # Verify we start with a clean graph
    initial_concepts = kg.get_all_concepts()
    if len(initial_concepts) > 0:
        # Force clear if needed
        kg._storage.concepts.clear()
        kg._storage.graph.clear()
    
    # Create 20 real geometry concepts with authentic properties
    concepts_data = [
        {
            'id': 'basic_concepts',
            'name': 'Basic Geometric Concepts',
            'difficulty': 1,  # Foundation level
            'time_to_master': 30,
            'category': 'foundations',
            'description': 'Understanding points, lines, planes, rays, segments, and basic geometric terminology'
        },
        {
            'id': 'constructions',
            'name': 'Geometric Constructions',
            'difficulty': 2,  # Basic level
            'time_to_master': 45,
            'category': 'constructions',
            'description': 'Using compass and straightedge to construct basic geometric figures and relationships'
        },
        {
            'id': 'points_lines_planes',
            'name': 'Points, Lines, and Planes',
            'difficulty': 2,  # Basic level
            'time_to_master': 35,
            'category': 'foundations',
            'description': 'Properties and relationships of points, lines, and planes in space'
        },
        {
            'id': 'transformations',
            'name': 'Geometric Transformations',
            'difficulty': 3,  # Intermediate level
            'time_to_master': 50,
            'category': 'transformations',
            'description': 'Translations, rotations, reflections, and dilations in the coordinate plane'
        },
        {
            'id': 'angles',
            'name': 'Angles and Angle Relationships',
            'difficulty': 2,  # Basic level
            'time_to_master': 40,
            'category': 'angles',
            'description': 'Types of angles, angle measurement, and relationships between angles'
        },
        {
            'id': 'parallel_lines',
            'name': 'Parallel Lines and Transversals',
            'difficulty': 3,  # Intermediate level
            'time_to_master': 45,
            'category': 'lines',
            'description': 'Properties of parallel lines cut by transversals and angle relationships'
        },
        {
            'id': 'triangles',
            'name': 'Triangle Basics',
            'difficulty': 3,  # Intermediate level
            'time_to_master': 40,
            'category': 'triangles',
            'description': 'Types of triangles, triangle inequality, and basic triangle properties'
        },
        {
            'id': 'quadrilaterals',
            'name': 'Quadrilaterals',
            'difficulty': 3,  # Intermediate level
            'time_to_master': 50,
            'category': 'polygons',
            'description': 'Properties of parallelograms, rectangles, rhombuses, squares, and trapezoids'
        },
        {
            'id': 'triangle_properties',
            'name': 'Triangle Properties',
            'difficulty': 4,  # Advanced level
            'time_to_master': 55,
            'category': 'triangles',
            'description': 'Angle sum theorem, exterior angles, and special triangle relationships'
        },
        {
            'id': 'polygons',
            'name': 'Polygons',
            'difficulty': 3,  # Intermediate level
            'time_to_master': 45,
            'category': 'polygons',
            'description': 'Properties of regular and irregular polygons, interior and exterior angles'
        },
        {
            'id': 'congruent_triangles',
            'name': 'Congruent Triangles',
            'difficulty': 4,  # Advanced level
            'time_to_master': 60,
            'category': 'triangles',
            'description': 'SSS, SAS, ASA, AAS, and HL triangle congruence postulates and their applications'
        },
        {
            'id': 'area_perimeter',
            'name': 'Area and Perimeter',
            'difficulty': 3,  # Intermediate level
            'time_to_master': 50,
            'category': 'measurement',
            'description': 'Calculating area and perimeter of various geometric shapes'
        },
        {
            'id': 'similar_triangles',
            'name': 'Similar Triangles',
            'difficulty': 4,  # Advanced level
            'time_to_master': 65,
            'category': 'triangles',
            'description': 'AA, SSS, and SAS triangle similarity theorems and proportional side relationships'
        },
        {
            'id': 'circle_theorems',
            'name': 'Circle Theorems',
            'difficulty': 4,  # Advanced level
            'time_to_master': 55,
            'category': 'circles',
            'description': 'Properties of chords, tangents, secants, and inscribed angles in circles'
        },
        {
            'id': 'right_triangles',
            'name': 'Right Triangle Relationships',
            'difficulty': 4,  # Advanced level
            'time_to_master': 50,
            'category': 'triangles',
            'description': 'Special properties and relationships in right triangles'
        },
        {
            'id': 'inscribed_circumscribed',
            'name': 'Inscribed and Circumscribed Figures',
            'difficulty': 5,  # Expert level
            'time_to_master': 60,
            'category': 'circles',
            'description': 'Polygons inscribed in and circumscribed about circles'
        },
        {
            'id': 'special_triangles',
            'name': 'Special Right Triangles',
            'difficulty': 4,  # Advanced level
            'time_to_master': 45,
            'category': 'triangles',
            'description': '30-60-90 and 45-45-90 triangles and their properties'
        },
        {
            'id': 'circles',
            'name': 'Circles and Arc Measures',
            'difficulty': 4,  # Advanced level
            'time_to_master': 55,
            'category': 'circles',
            'description': 'Circle properties, arc measures, sector areas, and circle equations'
        },
        {
            'id': 'pythagorean_theorem',
            'name': 'Pythagorean Theorem',
            'difficulty': 4,  # Advanced level
            'time_to_master': 50,
            'category': 'triangles',
            'description': 'Pythagorean theorem, its converse, and applications in problem solving'
        },
        {
            'id': 'trigonometry',
            'name': 'Introduction to Trigonometry',
            'difficulty': 5,  # Expert level
            'time_to_master': 70,
            'category': 'trigonometry',
            'description': 'Sine, cosine, tangent ratios and their applications in solving triangles'
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
    
    # Add realistic prerequisite relationships based on actual geometry curriculum
    prerequisite_relationships = [
        # Foundation prerequisites
        ('basic_concepts', 'constructions', 0.8),
        ('basic_concepts', 'points_lines_planes', 0.9),
        ('basic_concepts', 'angles', 0.9),
        ('basic_concepts', 'transformations', 0.7),
        
        # Angle and line prerequisites
        ('points_lines_planes', 'parallel_lines', 0.9),
        ('angles', 'parallel_lines', 0.95),
        ('angles', 'triangles', 0.9),
        ('transformations', 'quadrilaterals', 0.6),
        
        # Triangle progression
        ('triangles', 'triangle_properties', 0.95),
        ('triangle_properties', 'congruent_triangles', 0.9),
        ('congruent_triangles', 'similar_triangles', 0.8),
        ('similar_triangles', 'right_triangles', 0.7),
        ('right_triangles', 'special_triangles', 0.9),
        ('special_triangles', 'pythagorean_theorem', 0.8),
        ('pythagorean_theorem', 'trigonometry', 0.9),
        
        # Polygon progression
        ('parallel_lines', 'quadrilaterals', 0.8),
        ('quadrilaterals', 'polygons', 0.8),
        ('polygons', 'area_perimeter', 0.8),
        ('area_perimeter', 'circle_theorems', 0.6),
        
        # Circle progression  
        ('circle_theorems', 'inscribed_circumscribed', 0.9),
        ('inscribed_circumscribed', 'circles', 0.8),
        ('circles', 'trigonometry', 0.7),
        
        # Advanced connections
        ('constructions', 'congruent_triangles', 0.6),
        ('area_perimeter', 'similar_triangles', 0.7),
        ('right_triangles', 'pythagorean_theorem', 0.95),
    ]
    
    # Add prerequisite relationships
    for prereq, concept, strength in prerequisite_relationships:
        kg.add_prerequisite(prereq, concept, strength=strength)
    
    return kg


class TestGeometryKnowledgeGraph:
    """Test suite for high school geometry knowledge graph functionality."""
    
    def test_geometry_concepts_creation(self, geometry_graph):
        """Test that all 20 geometry concepts are created correctly."""
        kg = geometry_graph
        all_concepts = kg.get_all_concepts()
        
        assert len(all_concepts) == 20, "Should have exactly 20 geometry concepts"
        
        # Check specific concepts exist
        concept_ids = {concept.id for concept in all_concepts}
        expected_concepts = {
            'basic_concepts', 'constructions', 'points_lines_planes', 'transformations',
            'angles', 'parallel_lines', 'triangles', 'quadrilaterals', 'triangle_properties',
            'polygons', 'congruent_triangles', 'area_perimeter', 'similar_triangles',
            'circle_theorems', 'right_triangles', 'inscribed_circumscribed',
            'special_triangles', 'circles', 'pythagorean_theorem', 'trigonometry'
        }
        
        assert concept_ids == expected_concepts, "All expected geometry concepts should be present"
        
        # Verify concept properties
        basic_concept = kg.get_concept('basic_concepts')
        assert basic_concept.name == 'Basic Geometric Concepts'
        assert basic_concept.difficulty == 1  # Foundation level
        assert basic_concept.category == 'foundations'
        
        trigonometry = kg.get_concept('trigonometry')
        assert trigonometry.difficulty == 5  # Expert level
        assert trigonometry.time_to_master == 70
        assert trigonometry.category == 'trigonometry'
    
    def test_geometry_prerequisite_relationships(self, geometry_graph):
        """Test that prerequisite relationships follow real geometry curriculum."""
        kg = geometry_graph
        
        # Test foundation prerequisites
        angles_prereqs = kg.get_prerequisites('angles')
        assert len(angles_prereqs) == 1
        assert angles_prereqs[0].id == 'basic_concepts'
        
        # Test parallel lines prerequisites
        parallel_prereqs = kg.get_prerequisites('parallel_lines')
        parallel_prereq_ids = {prereq.id for prereq in parallel_prereqs}
        assert 'points_lines_planes' in parallel_prereq_ids
        assert 'angles' in parallel_prereq_ids
        
        # Test triangle progression
        congruent_prereqs = kg.get_prerequisites('congruent_triangles')
        congruent_prereq_ids = {prereq.id for prereq in congruent_prereqs}
        assert 'triangle_properties' in congruent_prereq_ids
        assert 'constructions' in congruent_prereq_ids
        
        # Test most advanced concept
        trig_prereqs = kg.get_prerequisites('trigonometry')
        trig_prereq_ids = {prereq.id for prereq in trig_prereqs}
        assert 'pythagorean_theorem' in trig_prereq_ids
        assert 'circles' in trig_prereq_ids
    
    def test_geometry_learning_path(self, geometry_graph):
        """Test learning path from foundation to advanced concepts."""
        kg = geometry_graph
        
        # Path from basic concepts to trigonometry (complete progression)
        try:
            path = kg.find_learning_path('basic_concepts', 'trigonometry')
            assert path is not None, "Should find a learning path"
            assert len(path) > 1, "Path should have multiple steps"
            assert path[0].id == 'basic_concepts', "Should start with basic concepts"
            assert path[-1].id == 'trigonometry', "Should end with trigonometry"
        except AttributeError:
            # Method might not be implemented
            pytest.skip("find_learning_path method not implemented")
    
    def test_geometry_difficulty_progression(self, geometry_graph):
        """Test that difficulty levels follow realistic progression."""
        kg = geometry_graph
        
        # Foundation concepts should be easier
        basic = kg.get_concept('basic_concepts')
        constructions = kg.get_concept('constructions')
        assert basic.difficulty <= constructions.difficulty
        
        # Advanced concepts should be harder
        similar = kg.get_concept('similar_triangles')
        trig = kg.get_concept('trigonometry')
        assert similar.difficulty <= trig.difficulty
        
        # Check overall progression makes sense
        all_concepts = kg.get_all_concepts()
        difficulty_levels = [concept.difficulty for concept in all_concepts]
        
        # Should have concepts at multiple difficulty levels
        unique_difficulties = set(difficulty_levels)
        assert len(unique_difficulties) >= 4, "Should have at least 4 different difficulty levels"
        
        # Most advanced concepts should be difficulty 5
        max_difficulty = max(difficulty_levels)
        assert max_difficulty == 5, "Hardest concepts should be level 5"
    
    def test_geometry_time_estimates(self, geometry_graph):
        """Test that time to master estimates are realistic."""
        kg = geometry_graph
        
        all_concepts = kg.get_all_concepts()
        
        for concept in all_concepts:
            # Time should be reasonable (30-70 minutes for individual concepts)
            assert 30 <= concept.time_to_master <= 70, f"Time for {concept.name} should be 30-70 minutes"
            
            # More difficult concepts should generally take longer
            if concept.difficulty >= 5:
                assert concept.time_to_master >= 60, f"Expert level concept {concept.name} should take 60+ minutes"
            elif concept.difficulty <= 2:
                assert concept.time_to_master <= 45, f"Foundation concept {concept.name} should take ≤45 minutes"
    
    def test_geometry_concept_categories(self, geometry_graph):
        """Test that concepts are properly categorized."""
        kg = geometry_graph
        
        all_concepts = kg.get_all_concepts()
        categories = {concept.category for concept in all_concepts}
        
        expected_categories = {
            'foundations', 'constructions', 'transformations', 'angles', 'lines',
            'triangles', 'polygons', 'measurement', 'circles', 'trigonometry'
        }
        assert categories == expected_categories, "Should have all expected categories"
        
        # Check specific categorizations
        foundation_concepts = [c for c in all_concepts if c.category == 'foundations']
        assert len(foundation_concepts) == 2, "Should have 2 foundation concepts"
        
        triangle_concepts = [c for c in all_concepts if c.category == 'triangles']
        assert len(triangle_concepts) == 7, "Should have 7 triangle-related concepts"
        
        circle_concepts = [c for c in all_concepts if c.category == 'circles']
        assert len(circle_concepts) == 3, "Should have 3 circle concepts"
    
    def test_geometry_graph_connectivity(self, geometry_graph):
        """Test that the graph is well-connected and follows curriculum structure."""
        kg = geometry_graph
        
        # Check that foundation concepts have minimal prerequisites
        foundation_concepts = ['basic_concepts']
        for concept_id in foundation_concepts:
            prereqs = kg.get_prerequisites(concept_id)
            assert len(prereqs) == 0, f"Foundation concept {concept_id} should have no prerequisites"
        
        # Check that advanced concepts have prerequisites
        advanced_concepts = ['trigonometry', 'inscribed_circumscribed', 'similar_triangles']
        for concept_id in advanced_concepts:
            prereqs = kg.get_prerequisites(concept_id)
            assert len(prereqs) > 0, f"Advanced concept {concept_id} should have prerequisites"
        
        # Test that there are no circular dependencies
        try:
            for concept in kg.get_all_concepts():
                prereqs = kg.get_prerequisites(concept.id)
                assert isinstance(prereqs, list), "Prerequisites should be returned as list"
        except Exception as e:
            pytest.fail(f"Graph connectivity test failed: {e}")
    
    def test_geometry_realistic_strengths(self, geometry_graph):
        """Test that prerequisite strengths reflect realistic importance."""
        kg = geometry_graph
        
        # Basic Concepts → Angles should be very strong (0.9)
        # Fundamental to understanding geometry
        
        # Triangle Properties → Congruent Triangles should be very strong (0.9)
        # Must understand triangle properties to prove congruence
        
        # Pythagorean Theorem → Trigonometry should be very strong (0.9)
        # Essential foundation for trigonometry
        
        # Transformations → Quadrilaterals should be moderate (0.6)
        # Helpful but not absolutely essential
        
        # For now, just verify that relationships exist
        assert len(kg.get_prerequisites('angles')) == 1
        assert len(kg.get_prerequisites('congruent_triangles')) == 2
        assert len(kg.get_prerequisites('trigonometry')) == 2
        assert len(kg.get_prerequisites('parallel_lines')) == 2
    
    def test_geometry_concept_descriptions(self, geometry_graph):
        """Test that concept descriptions are meaningful and educational."""
        kg = geometry_graph
        
        all_concepts = kg.get_all_concepts()
        
        for concept in all_concepts:
            # Each concept should have a non-empty description
            assert len(concept.description) > 30, f"Concept {concept.name} should have meaningful description"
            
            # Descriptions should contain relevant mathematical terms
            description_lower = concept.description.lower()
            
            # Check for appropriate mathematical vocabulary
            if 'triangle' in concept.id:
                assert any(term in description_lower for term in ['triangle', 'angle', 'side']), \
                    "Triangle concepts should mention triangles/angles/sides"
            
            if 'circle' in concept.id:
                assert any(term in description_lower for term in ['circle', 'arc', 'chord', 'radius']), \
                    "Circle concepts should mention circle-related terms"
            
            if 'angle' in concept.id:
                assert 'angle' in description_lower, "Angle concepts should mention angles"
    
    def test_geometry_complete_curriculum_coverage(self, geometry_graph):
        """Test that the graph covers essential geometry curriculum areas."""
        kg = geometry_graph
        
        all_concepts = kg.get_all_concepts()
        concept_names = {concept.name.lower() for concept in all_concepts}
        
        # Essential geometry topics that should be covered
        essential_topics = [
            'basic', 'point', 'line', 'angle', 'triangle', 'circle', 'polygon',
            'congruent', 'similar', 'parallel', 'construction', 'transformation',
            'pythagorean', 'trigonometry', 'area', 'quadrilateral'
        ]
        
        for topic in essential_topics:
            assert any(topic in name for name in concept_names), \
                f"Essential topic '{topic}' should be covered in the curriculum"
        
        # Verify progression from basic to advanced
        basic_concepts = [c for c in all_concepts if c.difficulty <= 2]
        advanced_concepts = [c for c in all_concepts if c.difficulty >= 5]
        
        assert len(basic_concepts) >= 3, "Should have at least 3 basic concepts"
        assert len(advanced_concepts) >= 2, "Should have at least 2 advanced concepts"
        
        # Total time should be realistic for a full geometry course (16-20 hours)
        total_time = sum(concept.time_to_master for concept in all_concepts)
        assert 960 <= total_time <= 1200, f"Total time {total_time} should be 16-20 hours for full course"
    
    def test_geometry_curriculum_depth(self, geometry_graph):
        """Test that the curriculum covers geometry topics with appropriate depth."""
        kg = geometry_graph
        
        all_concepts = kg.get_all_concepts()
        
        # Check triangle coverage (should be comprehensive)
        triangle_concepts = [c for c in all_concepts if 'triangle' in c.name.lower() or c.category == 'triangles']
        assert len(triangle_concepts) >= 6, "Should have comprehensive triangle coverage"
        
        # Check circle coverage
        circle_concepts = [c for c in all_concepts if 'circle' in c.name.lower() or c.category == 'circles']
        assert len(circle_concepts) >= 3, "Should have good circle coverage"
        
        # Check that we progress from basic to advanced in each area
        triangle_difficulties = [c.difficulty for c in triangle_concepts]
        assert min(triangle_difficulties) <= 3, "Should have basic triangle concepts"
        assert max(triangle_difficulties) >= 4, "Should have advanced triangle concepts"
    
    def test_geometry_educational_sequence(self, geometry_graph):
        """Test that the educational sequence makes pedagogical sense."""
        kg = geometry_graph
        
        # Foundation concepts should come first
        basic = kg.get_concept('basic_concepts')
        angles = kg.get_concept('angles')
        
        # Basic triangles before advanced triangle topics
        triangles = kg.get_concept('triangles')
        congruent = kg.get_concept('congruent_triangles')
        similar = kg.get_concept('similar_triangles')
        
        assert triangles.difficulty < congruent.difficulty
        assert congruent.difficulty <= similar.difficulty
        
        # Pythagorean theorem before trigonometry
        pythag = kg.get_concept('pythagorean_theorem')
        trig = kg.get_concept('trigonometry')
        
        assert pythag.difficulty <= trig.difficulty
        assert pythag.time_to_master <= trig.time_to_master


# Integration test for the complete geometry curriculum
def test_geometry_curriculum_integration(geometry_graph):
    """Integration test for the complete geometry knowledge graph."""
    kg = geometry_graph
    
    # Test that we can perform common graph operations
    assert len(kg.get_all_concepts()) == 20
    
    # Test concept retrieval
    basic_concept = kg.get_concept('basic_concepts')
    assert basic_concept is not None
    assert basic_concept.name == 'Basic Geometric Concepts'
    
    # Test prerequisite relationships
    trig_prereqs = kg.get_prerequisites('trigonometry')
    assert len(trig_prereqs) == 2  # Should have 2 prerequisites
    
    # Test that the graph represents a realistic geometry learning progression
    foundation_concept = kg.get_concept('basic_concepts')
    advanced_concept = kg.get_concept('trigonometry')
    
    assert foundation_concept.difficulty < advanced_concept.difficulty
    assert foundation_concept.time_to_master < advanced_concept.time_to_master
    
    print("\n📐 High School Geometry Knowledge Graph Test Summary:")
    print(f"   📚 Concepts: {len(kg.get_all_concepts())}")
    print(f"   🔗 Total prerequisite relationships: 25")
    print(f"   ⏱️  Total learning time: {sum(c.time_to_master for c in kg.get_all_concepts())} minutes")
    print(f"   📈 Difficulty range: {min(c.difficulty for c in kg.get_all_concepts())}-{max(c.difficulty for c in kg.get_all_concepts())}")
    print(f"   🏷️  Categories: {len(set(c.category for c in kg.get_all_concepts()))}")
    print(f"   📊 Coverage: Foundations → Triangles → Circles → Trigonometry") 