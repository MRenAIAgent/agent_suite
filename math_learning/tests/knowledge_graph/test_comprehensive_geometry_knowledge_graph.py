"""
Comprehensive High School Geometry Knowledge Graph Tests

This module contains tests for a complete knowledge graph based on the full high school geometry 
curriculum including all major concepts and their authentic prerequisite relationships. 
This addresses the missing concepts from the previous geometry test.

Course: Complete High School Geometry Curriculum
Concepts: 35 comprehensive concepts covering all geometry areas
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
def comprehensive_geometry_graph():
    """
    Create a comprehensive knowledge graph with ALL high school geometry concepts.
    
    This graph represents the complete geometry curriculum including:
    - Triangle centers and special segments
    - Complete quadrilateral properties  
    - Full circle vocabulary and relationships
    - Coordinate geometry
    - Transformations and congruence/similarity
    - Geometric proofs and reasoning
    - 3D geometry (surface area and volume)
    - Composite area calculations
    
    Nodes: 35 comprehensive geometry concepts
    Edges: 45+ prerequisite relationships
    """
    # Use a unique temp file name
    temp_dir = tempfile.mkdtemp()
    unique_file = os.path.join(temp_dir, f"comprehensive_geometry_{uuid.uuid4().hex}.json")
    
    if os.path.exists(unique_file):
        os.remove(unique_file)
    
    config = MathLearningStorageConfig(
        backend=StorageBackend.NETWORKX,
        networkx_persistence_file=unique_file,
        networkx_auto_save=False
    )
    
    kg = KnowledgeGraph(name="Comprehensive Geometry Graph", config=config)
    
    # Clear any existing data
    initial_concepts = kg.get_all_concepts()
    if len(initial_concepts) > 0:
        kg._storage.concepts.clear()
        kg._storage.graph.clear()
    
    # Create 35 comprehensive geometry concepts
    concepts_data = [
        # FOUNDATION LEVEL (1-2)
        {
            'id': 'basic_concepts',
            'name': 'Basic Geometric Concepts',
            'difficulty': 1,
            'time_to_master': 30,
            'category': 'foundations',
            'description': 'Points, lines, planes, rays, segments, and fundamental geometric terminology'
        },
        {
            'id': 'geometric_proofs',
            'name': 'Geometric Proofs and Reasoning',
            'difficulty': 2,
            'time_to_master': 50,
            'category': 'proofs',
            'description': 'Two-column proofs, postulates, theorems, and logical reasoning in geometry'
        },
        {
            'id': 'constructions',
            'name': 'Geometric Constructions',
            'difficulty': 2,
            'time_to_master': 45,
            'category': 'constructions',
            'description': 'Compass and straightedge constructions of basic geometric figures'
        },
        {
            'id': 'angles',
            'name': 'Angles and Angle Relationships',
            'difficulty': 2,
            'time_to_master': 40,
            'category': 'angles',
            'description': 'Types of angles, angle measurement, and relationships between angles'
        },
        
        # INTERMEDIATE LEVEL (3)
        {
            'id': 'parallel_lines',
            'name': 'Parallel Lines and Transversals',
            'difficulty': 3,
            'time_to_master': 45,
            'category': 'lines',
            'description': 'Properties of parallel lines cut by transversals and angle relationships'
        },
        {
            'id': 'transformations',
            'name': 'Geometric Transformations',
            'difficulty': 3,
            'time_to_master': 55,
            'category': 'transformations',
            'description': 'Translations, reflections, rotations, and dilations in the coordinate plane'
        },
        {
            'id': 'coordinate_geometry',
            'name': 'Coordinate Geometry',
            'difficulty': 3,
            'time_to_master': 50,
            'category': 'coordinates',
            'description': 'Distance formula, midpoint formula, and slope relationships in coordinate plane'
        },
        {
            'id': 'triangles_basic',
            'name': 'Triangle Basics',
            'difficulty': 3,
            'time_to_master': 40,
            'category': 'triangles',
            'description': 'Types of triangles, triangle inequality theorem, and basic triangle properties'
        },
        {
            'id': 'circle_vocabulary',
            'name': 'Circle Vocabulary and Basic Properties',
            'difficulty': 3,
            'time_to_master': 35,
            'category': 'circles',
            'description': 'Radius, diameter, chord, secant, tangent, and basic circle relationships'
        },
        {
            'id': 'area_perimeter_basic',
            'name': 'Basic Area and Perimeter',
            'difficulty': 3,
            'time_to_master': 45,
            'category': 'measurement',
            'description': 'Area and perimeter of basic shapes: rectangles, triangles, and simple polygons'
        },
        
        # ADVANCED LEVEL (4)
        {
            'id': 'triangle_properties',
            'name': 'Triangle Properties and Theorems',
            'difficulty': 4,
            'time_to_master': 55,
            'category': 'triangles',
            'description': 'Angle sum theorem, exterior angles, and side-angle relationships in triangles'
        },
        {
            'id': 'triangle_special_segments',
            'name': 'Triangle Special Segments',
            'difficulty': 4,
            'time_to_master': 60,
            'category': 'triangles',
            'description': 'Medians, altitudes, angle bisectors, and perpendicular bisectors in triangles'
        },
        {
            'id': 'triangle_centers',
            'name': 'Triangle Centers',
            'difficulty': 4,
            'time_to_master': 65,
            'category': 'triangles',
            'description': 'Centroid, orthocenter, incenter, and circumcenter of triangles'
        },
        {
            'id': 'congruent_triangles',
            'name': 'Congruent Triangles',
            'difficulty': 4,
            'time_to_master': 60,
            'category': 'triangles',
            'description': 'SSS, SAS, ASA, AAS, and HL triangle congruence postulates and proofs'
        },
        {
            'id': 'similar_triangles',
            'name': 'Similar Triangles',
            'difficulty': 4,
            'time_to_master': 65,
            'category': 'triangles',
            'description': 'AA, SSS, and SAS triangle similarity theorems and proportional relationships'
        },
        {
            'id': 'right_triangles',
            'name': 'Right Triangle Relationships',
            'difficulty': 4,
            'time_to_master': 50,
            'category': 'triangles',
            'description': 'Special properties and relationships in right triangles'
        },
        {
            'id': 'special_right_triangles',
            'name': 'Special Right Triangles',
            'difficulty': 4,
            'time_to_master': 45,
            'category': 'triangles',
            'description': '30-60-90 and 45-45-90 triangles and their side relationships'
        },
        {
            'id': 'pythagorean_theorem',
            'name': 'Pythagorean Theorem',
            'difficulty': 4,
            'time_to_master': 50,
            'category': 'triangles',
            'description': 'Pythagorean theorem, its converse, and applications in problem solving'
        },
        {
            'id': 'parallelograms',
            'name': 'Parallelograms',
            'difficulty': 4,
            'time_to_master': 50,
            'category': 'quadrilaterals',
            'description': 'Properties of parallelograms and methods to prove quadrilaterals are parallelograms'
        },
        {
            'id': 'rectangles_rhombi_squares',
            'name': 'Rectangles, Rhombi, and Squares',
            'difficulty': 4,
            'time_to_master': 55,
            'category': 'quadrilaterals',
            'description': 'Special parallelograms: rectangles, rhombi, squares, and their unique properties'
        },
        {
            'id': 'trapezoids',
            'name': 'Trapezoids',
            'difficulty': 4,
            'time_to_master': 45,
            'category': 'quadrilaterals',
            'description': 'Properties of trapezoids, including isosceles trapezoids and midsegments'
        },
        {
            'id': 'polygon_properties',
            'name': 'Polygon Properties',
            'difficulty': 4,
            'time_to_master': 50,
            'category': 'polygons',
            'description': 'Interior and exterior angles of polygons, regular polygons, and polygon classification'
        },
        {
            'id': 'circle_angles_arcs',
            'name': 'Circle Angles and Arcs',
            'difficulty': 4,
            'time_to_master': 60,
            'category': 'circles',
            'description': 'Central angles, inscribed angles, arc measures, and angle-arc relationships'
        },
        {
            'id': 'circle_segment_relationships',
            'name': 'Circle Segment Relationships',
            'difficulty': 4,
            'time_to_master': 55,
            'category': 'circles',
            'description': 'Chord-chord, secant-secant, and tangent-secant relationships in circles'
        },
        {
            'id': 'coordinate_proofs',
            'name': 'Coordinate Geometry Proofs',
            'difficulty': 4,
            'time_to_master': 60,
            'category': 'coordinates',
            'description': 'Using coordinate geometry to prove properties of triangles and quadrilaterals'
        },
        {
            'id': 'transformation_congruence',
            'name': 'Transformations and Congruence',
            'difficulty': 4,
            'time_to_master': 55,
            'category': 'transformations',
            'description': 'Using transformations to establish congruence and analyze geometric figures'
        },
        {
            'id': 'composite_area',
            'name': 'Composite Area Calculations',
            'difficulty': 4,
            'time_to_master': 50,
            'category': 'measurement',
            'description': 'Area of composite shapes, irregular polygons, and complex geometric figures'
        },
        {
            'id': 'surface_area',
            'name': 'Surface Area of 3D Shapes',
            'difficulty': 4,
            'time_to_master': 55,
            'category': '3d_geometry',
            'description': 'Surface area of prisms, pyramids, cylinders, cones, and spheres'
        },
        {
            'id': 'volume_3d',
            'name': 'Volume of 3D Shapes',
            'difficulty': 4,
            'time_to_master': 55,
            'category': '3d_geometry',
            'description': 'Volume formulas for prisms, pyramids, cylinders, cones, and spheres'
        },
        
        # EXPERT LEVEL (5)
        {
            'id': 'transformation_similarity',
            'name': 'Transformations and Similarity',
            'difficulty': 5,
            'time_to_master': 60,
            'category': 'transformations',
            'description': 'Using dilations and similarity transformations to analyze geometric relationships'
        },
        {
            'id': 'inscribed_circumscribed',
            'name': 'Inscribed and Circumscribed Figures',
            'difficulty': 5,
            'time_to_master': 65,
            'category': 'circles',
            'description': 'Polygons inscribed in and circumscribed about circles, including construction methods'
        },
        {
            'id': 'advanced_circle_theorems',
            'name': 'Advanced Circle Theorems',
            'difficulty': 5,
            'time_to_master': 60,
            'category': 'circles',
            'description': 'Power of a point, radical axis, and advanced circle theorem applications'
        },
        {
            'id': 'complex_proofs',
            'name': 'Complex Geometric Proofs',
            'difficulty': 5,
            'time_to_master': 70,
            'category': 'proofs',
            'description': 'Multi-step proofs involving multiple geometric concepts and advanced reasoning'
        },
        {
            'id': 'coordinate_transformations',
            'name': 'Coordinate Transformations',
            'difficulty': 5,
            'time_to_master': 65,
            'category': 'coordinates',
            'description': 'Algebraic representations of transformations and their effects on coordinates'
        },
        {
            'id': 'trigonometry_intro',
            'name': 'Introduction to Trigonometry',
            'difficulty': 5,
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
    
    # Add comprehensive prerequisite relationships (45+ relationships)
    prerequisite_relationships = [
        # Foundation prerequisites
        ('basic_concepts', 'geometric_proofs', 0.9),
        ('basic_concepts', 'constructions', 0.8),
        ('basic_concepts', 'angles', 0.95),
        ('basic_concepts', 'coordinate_geometry', 0.7),
        
        # Angle and line development
        ('angles', 'parallel_lines', 0.95),
        ('geometric_proofs', 'parallel_lines', 0.8),
        ('parallel_lines', 'triangles_basic', 0.8),
        ('angles', 'triangles_basic', 0.9),
        
        # Coordinate geometry progression
        ('coordinate_geometry', 'coordinate_proofs', 0.9),
        ('coordinate_geometry', 'transformation_congruence', 0.8),
        ('coordinate_geometry', 'coordinate_transformations', 0.8),
        
        # Triangle progression - basic to advanced
        ('triangles_basic', 'triangle_properties', 0.95),
        ('triangle_properties', 'triangle_special_segments', 0.9),
        ('triangle_special_segments', 'triangle_centers', 0.9),
        ('triangle_properties', 'congruent_triangles', 0.9),
        ('congruent_triangles', 'similar_triangles', 0.8),
        ('similar_triangles', 'right_triangles', 0.7),
        ('right_triangles', 'special_right_triangles', 0.9),
        ('right_triangles', 'pythagorean_theorem', 0.95),
        ('pythagorean_theorem', 'trigonometry_intro', 0.9),
        
        # Quadrilateral progression
        ('parallel_lines', 'parallelograms', 0.85),
        ('parallelograms', 'rectangles_rhombi_squares', 0.9),
        ('parallelograms', 'trapezoids', 0.7),
        ('rectangles_rhombi_squares', 'polygon_properties', 0.8),
        ('trapezoids', 'polygon_properties', 0.7),
        
        # Circle progression
        ('circle_vocabulary', 'circle_angles_arcs', 0.95),
        ('circle_angles_arcs', 'circle_segment_relationships', 0.8),
        ('circle_segment_relationships', 'inscribed_circumscribed', 0.8),
        ('circle_segment_relationships', 'advanced_circle_theorems', 0.9),
        ('inscribed_circumscribed', 'advanced_circle_theorems', 0.8),
        
        # Transformation progression
        ('transformations', 'transformation_congruence', 0.9),
        ('transformation_congruence', 'transformation_similarity', 0.8),
        ('transformations', 'coordinate_transformations', 0.8),
        
        # Area and measurement progression
        ('area_perimeter_basic', 'composite_area', 0.9),
        ('polygon_properties', 'composite_area', 0.8),
        ('area_perimeter_basic', 'surface_area', 0.8),
        ('surface_area', 'volume_3d', 0.9),
        
        # Proof progression
        ('geometric_proofs', 'congruent_triangles', 0.8),
        ('geometric_proofs', 'coordinate_proofs', 0.8),
        ('congruent_triangles', 'complex_proofs', 0.8),
        ('similar_triangles', 'complex_proofs', 0.7),
        ('coordinate_proofs', 'complex_proofs', 0.8),
        
        # Advanced connections
        ('constructions', 'triangle_centers', 0.7),
        ('constructions', 'inscribed_circumscribed', 0.8),
        ('triangle_centers', 'complex_proofs', 0.7),
        ('advanced_circle_theorems', 'complex_proofs', 0.8),
        ('transformation_similarity', 'trigonometry_intro', 0.6),
        ('coordinate_transformations', 'complex_proofs', 0.7),
    ]
    
    # Add prerequisite relationships
    for prereq, concept, strength in prerequisite_relationships:
        kg.add_prerequisite(prereq, concept, strength=strength)
    
    return kg


class TestComprehensiveGeometryKnowledgeGraph:
    """Test suite for comprehensive high school geometry knowledge graph."""
    
    def test_comprehensive_concepts_creation(self, comprehensive_geometry_graph):
        """Test that all 35 comprehensive geometry concepts are created correctly."""
        kg = comprehensive_geometry_graph
        all_concepts = kg.get_all_concepts()
        
        assert len(all_concepts) == 35, "Should have exactly 35 comprehensive geometry concepts"
        
        # Check specific missing concepts are now included
        concept_ids = {concept.id for concept in all_concepts}
        
        # Triangle special segments and centers
        assert 'triangle_special_segments' in concept_ids, "Should include triangle special segments"
        assert 'triangle_centers' in concept_ids, "Should include triangle centers"
        
        # Complete quadrilateral coverage
        assert 'parallelograms' in concept_ids, "Should include parallelograms"
        assert 'rectangles_rhombi_squares' in concept_ids, "Should include special parallelograms"
        assert 'trapezoids' in concept_ids, "Should include trapezoids"
        
        # Circle vocabulary and relationships
        assert 'circle_vocabulary' in concept_ids, "Should include circle vocabulary"
        assert 'circle_angles_arcs' in concept_ids, "Should include circle angles and arcs"
        assert 'circle_segment_relationships' in concept_ids, "Should include circle segment relationships"
        
        # Coordinate geometry
        assert 'coordinate_geometry' in concept_ids, "Should include coordinate geometry"
        assert 'coordinate_proofs' in concept_ids, "Should include coordinate proofs"
        
        # Proofs and reasoning
        assert 'geometric_proofs' in concept_ids, "Should include geometric proofs"
        assert 'complex_proofs' in concept_ids, "Should include complex proofs"
        
        # 3D geometry
        assert 'surface_area' in concept_ids, "Should include surface area"
        assert 'volume_3d' in concept_ids, "Should include 3D volume"
        
        # Composite area
        assert 'composite_area' in concept_ids, "Should include composite area calculations"
    
    def test_triangle_concepts_comprehensive(self, comprehensive_geometry_graph):
        """Test comprehensive triangle concept coverage."""
        kg = comprehensive_geometry_graph
        
        triangle_concepts = [c for c in kg.get_all_concepts() if c.category == 'triangles']
        assert len(triangle_concepts) == 9, "Should have 9 comprehensive triangle concepts"
        
        # Check specific triangle concepts
        triangle_names = {c.name for c in triangle_concepts}
        
        assert any('Special Segments' in name for name in triangle_names), "Should include triangle special segments"
        assert any('Centers' in name for name in triangle_names), "Should include triangle centers"
        assert any('Congruent' in name for name in triangle_names), "Should include congruent triangles"
        assert any('Similar' in name for name in triangle_names), "Should include similar triangles"
        assert any('Pythagorean' in name for name in triangle_names), "Should include Pythagorean theorem"
        
        # Check triangle centers concept specifically
        centers_concept = kg.get_concept('triangle_centers')
        assert 'centroid' in centers_concept.description.lower(), "Should mention centroid"
        assert 'orthocenter' in centers_concept.description.lower(), "Should mention orthocenter"
        assert 'incenter' in centers_concept.description.lower(), "Should mention incenter"
        assert 'circumcenter' in centers_concept.description.lower(), "Should mention circumcenter"
    
    def test_quadrilateral_concepts_complete(self, comprehensive_geometry_graph):
        """Test complete quadrilateral concept coverage."""
        kg = comprehensive_geometry_graph
        
        quad_concepts = [c for c in kg.get_all_concepts() if c.category == 'quadrilaterals']
        assert len(quad_concepts) == 3, "Should have 3 quadrilateral concepts"
        
        # Check specific quadrilateral concepts
        parallelogram_concept = kg.get_concept('parallelograms')
        assert parallelogram_concept.difficulty == 4, "Parallelograms should be advanced level"
        
        special_quads = kg.get_concept('rectangles_rhombi_squares')
        assert 'rectangle' in special_quads.description.lower(), "Should mention rectangles"
        assert 'rhombi' in special_quads.description.lower(), "Should mention rhombi"
        assert 'square' in special_quads.description.lower(), "Should mention squares"
        
        trapezoid_concept = kg.get_concept('trapezoids')
        assert 'isosceles' in trapezoid_concept.description.lower(), "Should mention isosceles trapezoids"
    
    def test_circle_concepts_comprehensive(self, comprehensive_geometry_graph):
        """Test comprehensive circle concept coverage."""
        kg = comprehensive_geometry_graph
        
        circle_concepts = [c for c in kg.get_all_concepts() if c.category == 'circles']
        assert len(circle_concepts) == 5, "Should have 5 comprehensive circle concepts"
        
        # Check circle vocabulary
        vocab_concept = kg.get_concept('circle_vocabulary')
        vocab_desc = vocab_concept.description.lower()
        assert 'radius' in vocab_desc, "Should mention radius"
        assert 'diameter' in vocab_desc, "Should mention diameter"
        assert 'chord' in vocab_desc, "Should mention chord"
        assert 'secant' in vocab_desc, "Should mention secant"
        assert 'tangent' in vocab_desc, "Should mention tangent"
        
        # Check angles and arcs
        angles_concept = kg.get_concept('circle_angles_arcs')
        angles_desc = angles_concept.description.lower()
        assert 'central' in angles_desc, "Should mention central angles"
        assert 'inscribed' in angles_desc, "Should mention inscribed angles"
        assert 'arc' in angles_desc, "Should mention arcs"
    
    def test_coordinate_geometry_concepts(self, comprehensive_geometry_graph):
        """Test coordinate geometry concept coverage."""
        kg = comprehensive_geometry_graph
        
        coord_concepts = [c for c in kg.get_all_concepts() if c.category == 'coordinates']
        assert len(coord_concepts) == 3, "Should have 3 coordinate geometry concepts"
        
        # Check basic coordinate geometry
        basic_coord = kg.get_concept('coordinate_geometry')
        coord_desc = basic_coord.description.lower()
        assert 'distance' in coord_desc, "Should mention distance formula"
        assert 'midpoint' in coord_desc, "Should mention midpoint formula"
        assert 'slope' in coord_desc, "Should mention slope"
        
        # Check coordinate proofs
        coord_proofs = kg.get_concept('coordinate_proofs')
        assert coord_proofs.difficulty == 4, "Coordinate proofs should be advanced"
        assert 'prove' in coord_proofs.description.lower(), "Should mention proving properties"
    
    def test_transformation_concepts_complete(self, comprehensive_geometry_graph):
        """Test complete transformation concept coverage."""
        kg = comprehensive_geometry_graph
        
        transform_concepts = [c for c in kg.get_all_concepts() if c.category == 'transformations']
        assert len(transform_concepts) == 3, "Should have 3 transformation concepts"
        
        # Check basic transformations
        basic_transform = kg.get_concept('transformations')
        transform_desc = basic_transform.description.lower()
        assert 'translation' in transform_desc, "Should mention translations"
        assert 'reflection' in transform_desc, "Should mention reflections"
        assert 'rotation' in transform_desc, "Should mention rotations"
        assert 'dilation' in transform_desc, "Should mention dilations"
        
        # Check congruence and similarity through transformations
        congruence_transform = kg.get_concept('transformation_congruence')
        similarity_transform = kg.get_concept('transformation_similarity')
        assert congruence_transform.difficulty == 4, "Should be advanced level"
        assert similarity_transform.difficulty == 5, "Should be expert level"
    
    def test_proof_concepts_comprehensive(self, comprehensive_geometry_graph):
        """Test comprehensive proof concept coverage."""
        kg = comprehensive_geometry_graph
        
        proof_concepts = [c for c in kg.get_all_concepts() if c.category == 'proofs']
        assert len(proof_concepts) == 2, "Should have 2 proof concepts"
        
        # Check basic geometric proofs
        basic_proofs = kg.get_concept('geometric_proofs')
        proof_desc = basic_proofs.description.lower()
        assert 'two-column' in proof_desc, "Should mention two-column proofs"
        assert 'postulate' in proof_desc, "Should mention postulates"
        assert 'theorem' in proof_desc, "Should mention theorems"
        
        # Check complex proofs
        complex_proofs = kg.get_concept('complex_proofs')
        assert complex_proofs.difficulty == 5, "Complex proofs should be expert level"
        assert complex_proofs.time_to_master == 70, "Should take longest time"
    
    def test_3d_geometry_concepts(self, comprehensive_geometry_graph):
        """Test 3D geometry concept coverage."""
        kg = comprehensive_geometry_graph
        
        geometry_3d_concepts = [c for c in kg.get_all_concepts() if c.category == '3d_geometry']
        assert len(geometry_3d_concepts) == 2, "Should have 2 3D geometry concepts"
        
        # Check surface area
        surface_area = kg.get_concept('surface_area')
        surface_desc = surface_area.description.lower()
        assert 'prism' in surface_desc, "Should mention prisms"
        assert 'pyramid' in surface_desc, "Should mention pyramids"
        assert 'cylinder' in surface_desc, "Should mention cylinders"
        assert 'cone' in surface_desc, "Should mention cones"
        assert 'sphere' in surface_desc, "Should mention spheres"
        
        # Check volume
        volume_3d = kg.get_concept('volume_3d')
        volume_desc = volume_3d.description.lower()
        assert 'volume' in volume_desc, "Should mention volume"
        assert 'formula' in volume_desc, "Should mention formulas"
    
    def test_comprehensive_prerequisite_relationships(self, comprehensive_geometry_graph):
        """Test that prerequisite relationships are comprehensive and logical."""
        kg = comprehensive_geometry_graph
        
        # Test triangle progression
        triangle_props = kg.get_prerequisites('triangle_properties')
        assert len(triangle_props) == 1, "Triangle properties should have 1 prerequisite"
        
        triangle_centers = kg.get_prerequisites('triangle_centers')
        centers_prereq_ids = {prereq.id for prereq in triangle_centers}
        assert 'triangle_special_segments' in centers_prereq_ids, "Centers should require special segments"
        
        # Test quadrilateral progression
        special_quads = kg.get_prerequisites('rectangles_rhombi_squares')
        special_prereq_ids = {prereq.id for prereq in special_quads}
        assert 'parallelograms' in special_prereq_ids, "Special parallelograms should require basic parallelograms"
        
        # Test circle progression
        circle_segments = kg.get_prerequisites('circle_segment_relationships')
        segment_prereq_ids = {prereq.id for prereq in circle_segments}
        assert 'circle_angles_arcs' in segment_prereq_ids, "Segment relationships should require angles/arcs"
        
        # Test coordinate geometry progression
        coord_proofs = kg.get_prerequisites('coordinate_proofs')
        coord_prereq_ids = {prereq.id for prereq in coord_proofs}
        assert 'coordinate_geometry' in coord_prereq_ids, "Coordinate proofs should require basic coordinate geometry"
        
        # Test 3D progression
        volume_prereqs = kg.get_prerequisites('volume_3d')
        volume_prereq_ids = {prereq.id for prereq in volume_prereqs}
        assert 'surface_area' in volume_prereq_ids, "Volume should require surface area knowledge"
    
    def test_comprehensive_curriculum_coverage(self, comprehensive_geometry_graph):
        """Test that all essential geometry curriculum areas are covered comprehensively."""
        kg = comprehensive_geometry_graph
        
        all_concepts = kg.get_all_concepts()
        categories = {concept.category for concept in all_concepts}
        
        expected_categories = {
            'foundations', 'proofs', 'constructions', 'angles', 'lines', 'transformations',
            'coordinates', 'triangles', 'quadrilaterals', 'polygons', 'circles', 
            'measurement', '3d_geometry', 'trigonometry'
        }
        assert categories == expected_categories, "Should have all expected categories"
        
        # Check category distribution
        triangles = [c for c in all_concepts if c.category == 'triangles']
        assert len(triangles) == 9, "Should have comprehensive triangle coverage"
        
        circles = [c for c in all_concepts if c.category == 'circles']
        assert len(circles) == 5, "Should have comprehensive circle coverage"
        
        quadrilaterals = [c for c in all_concepts if c.category == 'quadrilaterals']
        assert len(quadrilaterals) == 3, "Should have comprehensive quadrilateral coverage"
        
        coordinates = [c for c in all_concepts if c.category == 'coordinates']
        assert len(coordinates) == 3, "Should have comprehensive coordinate geometry"
        
        transformations = [c for c in all_concepts if c.category == 'transformations']
        assert len(transformations) == 3, "Should have comprehensive transformations"
        
        proofs = [c for c in all_concepts if c.category == 'proofs']
        assert len(proofs) == 2, "Should have proof reasoning concepts"
        
        geometry_3d = [c for c in all_concepts if c.category == '3d_geometry']
        assert len(geometry_3d) == 2, "Should have 3D geometry concepts"
    
    def test_comprehensive_time_and_difficulty(self, comprehensive_geometry_graph):
        """Test that time and difficulty estimates are realistic for comprehensive curriculum."""
        kg = comprehensive_geometry_graph
        
        all_concepts = kg.get_all_concepts()
        
        # Check total time (should be realistic for full geometry course)
        total_time = sum(concept.time_to_master for concept in all_concepts)
        assert 1600 <= total_time <= 2000, f"Total time {total_time} should be 26-33 hours for comprehensive course"
        
        # Check difficulty distribution
        difficulty_counts = {}
        for concept in all_concepts:
            difficulty_counts[concept.difficulty] = difficulty_counts.get(concept.difficulty, 0) + 1
        
        assert difficulty_counts[1] >= 1, "Should have foundation concepts"
        assert difficulty_counts[2] >= 3, "Should have basic concepts"
        assert difficulty_counts[3] >= 5, "Should have intermediate concepts"
        assert difficulty_counts[4] >= 15, "Should have many advanced concepts"
        assert difficulty_counts[5] >= 5, "Should have expert concepts"
        
        # Check that most complex concepts are appropriately difficult
        complex_concepts = ['complex_proofs', 'trigonometry_intro', 'advanced_circle_theorems']
        for concept_id in complex_concepts:
            concept = kg.get_concept(concept_id)
            assert concept.difficulty == 5, f"{concept_id} should be expert level"
    
    def test_comprehensive_educational_sequence(self, comprehensive_geometry_graph):
        """Test that the comprehensive curriculum follows sound educational sequence."""
        kg = comprehensive_geometry_graph
        
        # Foundation should come before advanced
        basic = kg.get_concept('basic_concepts')
        complex_proofs = kg.get_concept('complex_proofs')
        assert basic.difficulty < complex_proofs.difficulty
        
        # Basic triangles before triangle centers
        triangles_basic = kg.get_concept('triangles_basic')
        triangle_centers = kg.get_concept('triangle_centers')
        assert triangles_basic.difficulty < triangle_centers.difficulty
        
        # Circle vocabulary before advanced circle theorems
        circle_vocab = kg.get_concept('circle_vocabulary')
        advanced_circles = kg.get_concept('advanced_circle_theorems')
        assert circle_vocab.difficulty < advanced_circles.difficulty
        
        # Basic coordinate geometry before coordinate transformations
        coord_basic = kg.get_concept('coordinate_geometry')
        coord_transform = kg.get_concept('coordinate_transformations')
        assert coord_basic.difficulty < coord_transform.difficulty
        
        # Surface area before volume
        surface = kg.get_concept('surface_area')
        volume = kg.get_concept('volume_3d')
        assert surface.difficulty <= volume.difficulty


def test_comprehensive_geometry_integration(comprehensive_geometry_graph):
    """Integration test for the complete comprehensive geometry curriculum."""
    kg = comprehensive_geometry_graph
    
    # Test that we can perform common graph operations
    assert len(kg.get_all_concepts()) == 35
    
    # Test concept retrieval
    basic_concept = kg.get_concept('basic_concepts')
    assert basic_concept is not None
    assert basic_concept.name == 'Basic Geometric Concepts'
    
    # Test advanced concept
    complex_proofs = kg.get_concept('complex_proofs')
    assert complex_proofs.difficulty == 5
    assert complex_proofs.time_to_master == 70
    
    # Test prerequisite relationships
    trig_prereqs = kg.get_prerequisites('trigonometry_intro')
    assert len(trig_prereqs) >= 1  # Should have prerequisites
    
    print("\n📐 Comprehensive High School Geometry Knowledge Graph Test Summary:")
    print(f"   📚 Concepts: {len(kg.get_all_concepts())}")
    print(f"   🔗 Total prerequisite relationships: 45+")
    print(f"   ⏱️  Total learning time: {sum(c.time_to_master for c in kg.get_all_concepts())} minutes")
    print(f"   📈 Difficulty range: {min(c.difficulty for c in kg.get_all_concepts())}-{max(c.difficulty for c in kg.get_all_concepts())}")
    print(f"   🏷️  Categories: {len(set(c.category for c in kg.get_all_concepts()))}")
    print(f"   📊 Coverage: Complete curriculum including triangle centers, coordinate proofs, 3D geometry")
    print(f"   🎯 Missing concepts addressed: Triangle special segments, centers, complete quadrilaterals,")
    print(f"      circle vocabulary, coordinate geometry, proofs, 3D geometry, composite areas") 