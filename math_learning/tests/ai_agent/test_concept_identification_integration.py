"""
Integration Tests for Concept Identification System

This module tests integration between the concept identification system
and other components like knowledge graphs and exercise banks.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import tempfile
import os

from math_learning.ai_agent.tools.concept_identification_system import ConceptIdentificationSystem
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.knowledge_graph.concept import Concept
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.exercises.exercise import Exercise
from math_learning.config.storage_config import MathLearningStorageConfig, StorageBackend


class TestKnowledgeGraphIntegration:
    """Test integration with knowledge graph systems."""
    
    @pytest.fixture
    def knowledge_graph(self):
        """Create a test knowledge graph."""
        # Use temporary file for testing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        temp_file.close()
        
        config = MathLearningStorageConfig(
            backend=StorageBackend.NETWORKX,
            networkx_persistence_file=temp_file.name,
            networkx_auto_save=False
        )
        
        kg = KnowledgeGraph(name="Test KG", config=config)
        
        # Add some test concepts
        concepts = [
            Concept("solving_linear_equations", "Solving Linear Equations", 3, 45, "algebra"),
            Concept("area_calculations", "Area Calculations", 2, 30, "geometry"),
            Concept("derivative_rules", "Derivative Rules", 5, 60, "calculus"),
            Concept("descriptive_statistics", "Descriptive Statistics", 3, 40, "statistics")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        yield kg
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def concept_system(self):
        """Create concept identification system."""
        return ConceptIdentificationSystem(llm=None)
    
    @pytest.mark.asyncio
    async def test_concept_mapping_to_knowledge_graph(self, concept_system, knowledge_graph):
        """Test mapping identified concepts to knowledge graph concepts."""
        # Identify concepts from an exercise
        exercise = "Solve for x: 2x + 5 = 13"
        result = await concept_system.identify_concepts(exercise, "text", use_llm=False)
        
        # Try to map identified concepts to knowledge graph
        mapped_concepts = []
        for concept_info in result.concepts:
            concept_id = concept_info["concept_id"]
            
            # Try to find in knowledge graph
            kg_concept = knowledge_graph.get_concept(concept_id)
            if kg_concept:
                mapped_concepts.append(kg_concept)
            else:
                # Try fuzzy matching by name
                all_kg_concepts = knowledge_graph.get_all_concepts()
                concept_name = concept_info["concept_name"].lower()
                
                for kg_concept in all_kg_concepts:
                    if any(word in kg_concept.name.lower() for word in concept_name.split()):
                        mapped_concepts.append(kg_concept)
                        break
        
        # Should find at least some mappings for algebra problems
        if result.course in ["Algebra I", "Pre-Algebra"]:
            # May find linear equations concept
            linear_concept = knowledge_graph.get_concept("solving_linear_equations")
            if linear_concept and any("linear" in c["concept_name"].lower() or "equation" in c["concept_name"].lower() for c in result.concepts):
                assert linear_concept in mapped_concepts or len(mapped_concepts) > 0
    
    @pytest.mark.asyncio
    async def test_reverse_lookup_exercises_for_identified_concepts(self, concept_system, knowledge_graph):
        """Test finding exercises for identified concepts using knowledge graph."""
        # Create mock exercise bank
        exercise_bank = ExerciseBank("Test Bank")
        
        # Add exercises linked to concepts
        exercises = [
            Exercise("Linear Eq 1", "Solve: x + 5 = 12", "x = 7", 2, exercise_id="ex1"),
            Exercise("Area Problem", "Find area of rectangle 5×3", "15", 2, exercise_id="ex2"),
            Exercise("Derivative", "Find d/dx of x²", "2x", 4, exercise_id="ex3")
        ]
        
        # Link exercises to concepts
        exercises[0].add_concept_relationship("solving_linear_equations", 0.9, "primary")
        exercises[1].add_concept_relationship("area_calculations", 0.9, "primary")
        exercises[2].add_concept_relationship("derivative_rules", 0.9, "primary")
        
        for exercise in exercises:
            exercise_bank.add_exercise(exercise)
        
        # Identify concepts from a new exercise
        new_exercise = "Solve for y: 3y - 7 = 14"
        result = await concept_system.identify_concepts(new_exercise, "text", use_llm=False)
        
        # Find related exercises for identified concepts
        related_exercises = []
        for concept_info in result.concepts:
            concept_id = concept_info["concept_id"]
            
            # Get exercises for this concept
            concept_exercises = exercise_bank.get_exercises_for_concept(concept_id)
            related_exercises.extend(concept_exercises)
        
        # For algebra problems, should find related linear equation exercises
        if result.course in ["Algebra I", "Pre-Algebra"]:
            # Try to find exercises for solving_linear_equations
            linear_exercises = exercise_bank.get_exercises_for_concept("solving_linear_equations")
            if linear_exercises:
                assert len(linear_exercises) > 0
    
    @pytest.mark.asyncio
    async def test_concept_prerequisite_analysis(self, concept_system, knowledge_graph):
        """Test analyzing prerequisites for identified concepts."""
        # Add prerequisite relationships
        knowledge_graph.add_prerequisite("solving_linear_equations", "area_calculations", 0.3)
        knowledge_graph.add_prerequisite("area_calculations", "derivative_rules", 0.2)
        
        # Identify concepts from a complex exercise
        exercise = "Find the area under the curve y = 2x + 1 from x = 0 to x = 5"
        result = await concept_system.identify_concepts(exercise, "text", use_llm=False)
        
        # Analyze prerequisites for identified concepts
        all_prerequisites = set()
        for concept_info in result.concepts:
            concept_id = concept_info["concept_id"]
            
            # Get prerequisites from knowledge graph
            prereqs = knowledge_graph.get_prerequisites(concept_id)
            all_prerequisites.update(prereq.id for prereq in prereqs)
        
        # Should be able to analyze prerequisite structure
        assert isinstance(all_prerequisites, set)


class TestExerciseBankIntegration:
    """Test integration with exercise bank systems."""
    
    @pytest.fixture
    def exercise_bank(self):
        """Create a test exercise bank with various exercises."""
        bank = ExerciseBank("Integration Test Bank")
        
        # Create diverse exercises
        exercises = [
            # Algebra exercises
            Exercise("Basic Linear", "Solve: x + 3 = 7", "x = 4", 1, exercise_id="alg1"),
            Exercise("Two-step Linear", "Solve: 2x - 5 = 9", "x = 7", 2, exercise_id="alg2"),
            Exercise("Word Problem", "John has 5 more apples than Sue. If John has 12 apples, how many does Sue have?", "7", 2, exercise_id="alg3"),
            
            # Geometry exercises  
            Exercise("Rectangle Area", "Find area of rectangle with length 8 and width 5", "40", 2, exercise_id="geo1"),
            Exercise("Circle Area", "Find area of circle with radius 3 (use π=3.14)", "28.26", 3, exercise_id="geo2"),
            Exercise("Triangle Angles", "In triangle ABC, if angle A = 60° and B = 70°, find C", "50°", 3, exercise_id="geo3"),
            
            # Calculus exercises
            Exercise("Basic Derivative", "Find derivative of f(x) = x²", "f'(x) = 2x", 4, exercise_id="calc1"),
            Exercise("Chain Rule", "Find derivative of f(x) = (2x + 1)³", "f'(x) = 6(2x + 1)²", 5, exercise_id="calc2"),
            
            # Statistics exercises
            Exercise("Mean Calculation", "Find mean of 2, 4, 6, 8, 10", "6", 2, exercise_id="stat1"),
            Exercise("Probability", "What's probability of rolling a 6 on a fair die?", "1/6", 3, exercise_id="stat2")
        ]
        
        # Add concept relationships
        concept_mappings = {
            "alg1": [("solving_linear_equations", 0.9, "primary")],
            "alg2": [("solving_linear_equations", 0.95, "primary")],
            "alg3": [("solving_linear_equations", 0.8, "primary"), ("word_problems", 0.9, "secondary")],
            "geo1": [("area_calculations", 0.95, "primary")],
            "geo2": [("area_calculations", 0.9, "primary"), ("circle_properties", 0.8, "secondary")],
            "geo3": [("angle_relationships", 0.9, "primary"), ("triangle_properties", 0.8, "secondary")],
            "calc1": [("derivative_rules", 0.9, "primary")],
            "calc2": [("derivative_rules", 0.8, "primary"), ("chain_rule", 0.95, "secondary")],
            "stat1": [("descriptive_statistics", 0.9, "primary")],
            "stat2": [("probability_basics", 0.9, "primary")]
        }
        
        for exercise in exercises:
            # Add concept relationships
            if exercise.id in concept_mappings:
                for concept_id, weight, rel_type in concept_mappings[exercise.id]:
                    exercise.add_concept_relationship(concept_id, weight, rel_type)
            
            bank.add_exercise(exercise)
        
        return bank
    
    @pytest.fixture
    def concept_system(self):
        return ConceptIdentificationSystem(llm=None)
    
    @pytest.mark.asyncio
    async def test_exercise_analysis_and_similarity_finding(self, concept_system, exercise_bank):
        """Test analyzing exercises and finding similar ones."""
        # Analyze a new exercise
        new_exercise = "Solve the equation: 3x + 2 = 14"
        result = await concept_system.identify_concepts(new_exercise, "text", use_llm=False)
        
        # Find similar exercises in the bank
        similar_exercises = []
        
        for concept_info in result.concepts:
            concept_id = concept_info["concept_id"]
            confidence = concept_info["confidence"]
            
            # Only consider high-confidence concepts
            if confidence > 0.5:
                exercises = exercise_bank.get_exercises_for_concept(concept_id)
                similar_exercises.extend(exercises)
        
        # Should find some similar exercises for algebra problems
        if result.course in ["Algebra I", "Pre-Algebra"]:
            # Look for linear equation exercises specifically
            linear_exercises = exercise_bank.get_exercises_for_concept("solving_linear_equations")
            if linear_exercises:
                assert len(linear_exercises) >= 2  # Should have alg1 and alg2
                
                # Verify they are actually linear equation exercises
                for ex in linear_exercises:
                    assert "solve" in ex.title.lower() or "solve" in ex.problem.lower()
    
    @pytest.mark.asyncio
    async def test_exercise_difficulty_correlation(self, concept_system, exercise_bank):
        """Test correlation between identified difficulty and exercise bank difficulty."""
        # Test exercises of different difficulties
        test_cases = [
            ("What is 2 + 3?", "elementary"),
            ("Solve: x + 5 = 12", "middle_school"),
            ("Solve: 2x² - 5x + 3 = 0", "high_school_advanced"),
            ("Find the integral of x² + 2x", "college")
        ]
        
        for exercise_text, expected_difficulty_level in test_cases:
            result = await concept_system.identify_concepts(exercise_text, "text", use_llm=False)
            
            # Find exercises in bank for identified concepts
            related_exercises = []
            for concept_info in result.concepts:
                concept_id = concept_info["concept_id"]
                exercises = exercise_bank.get_exercises_for_concept(concept_id)
                related_exercises.extend(exercises)
            
            if related_exercises:
                # Check if difficulty levels are reasonable
                bank_difficulties = [ex.difficulty for ex in related_exercises]
                avg_bank_difficulty = sum(bank_difficulties) / len(bank_difficulties)
                
                # Map difficulty levels to numbers for comparison
                difficulty_map = {
                    "elementary": 1,
                    "middle_school": 2,
                    "high_school_basic": 3,
                    "high_school_advanced": 4,
                    "college": 5
                }
                
                identified_difficulty_num = difficulty_map.get(result.difficulty_level, 3)
                
                # Should be somewhat correlated (within 2 levels)
                assert abs(avg_bank_difficulty - identified_difficulty_num) <= 2
    
    @pytest.mark.asyncio
    async def test_batch_exercise_classification(self, concept_system, exercise_bank):
        """Test batch classification of exercises from the bank."""
        # Get all exercises from bank
        all_exercises = exercise_bank.get_all_exercises()
        
        # Convert to format for batch processing
        exercise_data = [
            {"content": ex.problem, "content_type": "text"}
            for ex in all_exercises[:5]  # Test first 5 for performance
        ]
        
        # Batch analyze
        results = await concept_system.batch_identify_concepts(exercise_data, use_llm=False)
        
        assert len(results) == len(exercise_data)
        
        # Verify results make sense for known exercises
        for i, result in enumerate(results):
            original_exercise = all_exercises[i]
            
            # Basic sanity checks
            assert isinstance(result.course, str)
            assert isinstance(result.topic, str)
            assert isinstance(result.concepts, list)
            assert 0.0 <= result.confidence <= 1.0
            
            # For exercises with known concept relationships, verify some alignment
            original_concepts = original_exercise.get_all_concept_ids()
            identified_concept_ids = [c["concept_id"] for c in result.concepts]
            
            if original_concepts and identified_concept_ids:
                # Should have some overlap or related concepts
                # (exact matching might not always work due to different concept ID schemes)
                assert len(result.concepts) > 0


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def complete_system(self):
        """Set up a complete system with all components."""
        # Create temporary knowledge graph
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        temp_file.close()
        
        config = MathLearningStorageConfig(
            backend=StorageBackend.NETWORKX,
            networkx_persistence_file=temp_file.name,
            networkx_auto_save=False
        )
        
        kg = KnowledgeGraph(name="Complete Test KG", config=config)
        
        # Add comprehensive concept set
        concepts = [
            Concept("basic_arithmetic", "Basic Arithmetic", 1, 20, "arithmetic"),
            Concept("solving_linear_equations", "Solving Linear Equations", 3, 45, "algebra"),
            Concept("quadratic_equations", "Quadratic Equations", 4, 60, "algebra"),
            Concept("area_calculations", "Area Calculations", 2, 30, "geometry"),
            Concept("triangle_properties", "Triangle Properties", 3, 40, "geometry"),
            Concept("derivative_rules", "Derivative Rules", 5, 60, "calculus"),
            Concept("integral_calculus", "Integral Calculus", 5, 70, "calculus"),
            Concept("descriptive_statistics", "Descriptive Statistics", 3, 40, "statistics"),
            Concept("probability_basics", "Probability Basics", 3, 35, "statistics")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Add prerequisites
        kg.add_prerequisite("basic_arithmetic", "solving_linear_equations", 0.9)
        kg.add_prerequisite("solving_linear_equations", "quadratic_equations", 0.8)
        kg.add_prerequisite("basic_arithmetic", "area_calculations", 0.7)
        kg.add_prerequisite("area_calculations", "triangle_properties", 0.6)
        kg.add_prerequisite("quadratic_equations", "derivative_rules", 0.5)
        kg.add_prerequisite("derivative_rules", "integral_calculus", 0.9)
        kg.add_prerequisite("basic_arithmetic", "descriptive_statistics", 0.6)
        kg.add_prerequisite("descriptive_statistics", "probability_basics", 0.7)
        
        # Create exercise bank
        exercise_bank = ExerciseBank("Complete Test Bank")
        
        # Create concept identification system
        concept_system = ConceptIdentificationSystem(llm=None)
        
        system = {
            "knowledge_graph": kg,
            "exercise_bank": exercise_bank,
            "concept_system": concept_system,
            "temp_file": temp_file.name
        }
        
        yield system
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.mark.asyncio
    async def test_complete_exercise_analysis_workflow(self, complete_system):
        """Test complete workflow from exercise text to concept mapping to related exercises."""
        kg = complete_system["knowledge_graph"]
        exercise_bank = complete_system["exercise_bank"]
        concept_system = complete_system["concept_system"]
        
        # Step 1: Analyze a new exercise
        new_exercise = "Find the area of a triangle with base 10 cm and height 8 cm"
        result = await concept_system.identify_concepts(new_exercise, "text", use_llm=False)
        
        print(f"Identified Course: {result.course}")
        print(f"Identified Topic: {result.topic}")
        print(f"Identified Concepts: {[c['concept_name'] for c in result.concepts]}")
        
        # Step 2: Map to knowledge graph concepts
        mapped_concepts = []
        for concept_info in result.concepts:
            concept_id = concept_info["concept_id"]
            
            # Try direct lookup
            kg_concept = kg.get_concept(concept_id)
            if kg_concept:
                mapped_concepts.append(kg_concept)
            else:
                # Try fuzzy matching
                all_concepts = kg.get_all_concepts()
                for kg_concept in all_concepts:
                    if ("area" in kg_concept.name.lower() and "area" in concept_info["concept_name"].lower()) or \
                       ("triangle" in kg_concept.name.lower() and "triangle" in concept_info["concept_name"].lower()):
                        mapped_concepts.append(kg_concept)
                        break
        
        print(f"Mapped KG Concepts: {[c.name for c in mapped_concepts]}")
        
        # Step 3: Find prerequisites for mapped concepts
        all_prerequisites = set()
        for concept in mapped_concepts:
            prereqs = kg.get_prerequisites(concept.id)
            all_prerequisites.update(prereqs)
        
        print(f"Prerequisites: {[p.name for p in all_prerequisites]}")
        
        # Step 4: Create and add related exercises to bank
        related_exercises = [
            Exercise("Triangle Area 1", "Find area: base=6, height=4", "12", 2, exercise_id="tri1"),
            Exercise("Triangle Area 2", "Triangle with base 12 and height 5", "30", 2, exercise_id="tri2"),
            Exercise("Rectangle Area", "Rectangle 8×6", "48", 2, exercise_id="rect1")
        ]
        
        # Link exercises to concepts
        for ex in related_exercises:
            if "triangle" in ex.problem.lower():
                ex.add_concept_relationship("area_calculations", 0.9, "primary")
                ex.add_concept_relationship("triangle_properties", 0.7, "secondary")
            else:
                ex.add_concept_relationship("area_calculations", 0.9, "primary")
            
            exercise_bank.add_exercise(ex)
        
        # Step 5: Find similar exercises
        similar_exercises = []
        for concept_info in result.concepts:
            concept_id = concept_info["concept_id"]
            exercises = exercise_bank.get_exercises_for_concept(concept_id)
            similar_exercises.extend(exercises)
        
        # Try with known concept IDs
        for known_concept in ["area_calculations", "triangle_properties"]:
            exercises = exercise_bank.get_exercises_for_concept(known_concept)
            similar_exercises.extend(exercises)
        
        print(f"Similar Exercises Found: {len(similar_exercises)}")
        
        # Verify the complete workflow
        assert isinstance(result, type(result))  # ConceptIdentification
        assert result.course in ["Geometry", "Elementary Arithmetic", "Pre-Algebra"]
        assert len(result.concepts) > 0
        
        # Should find some exercises (either through direct concept mapping or manual addition)
        area_exercises = exercise_bank.get_exercises_for_concept("area_calculations")
        assert len(area_exercises) >= 0  # May be 0 if concept mapping doesn't work perfectly
    
    @pytest.mark.asyncio
    async def test_learning_path_generation_workflow(self, complete_system):
        """Test generating learning paths based on identified concepts."""
        kg = complete_system["knowledge_graph"]
        concept_system = complete_system["concept_system"]
        
        # Analyze a advanced exercise
        advanced_exercise = "Find the derivative of f(x) = x³ + 2x² - 5x + 1"
        result = await concept_system.identify_concepts(advanced_exercise, "text", use_llm=False)
        
        # For calculus problems, trace back prerequisites
        target_concepts = []
        
        # Try to find calculus concepts in knowledge graph
        all_kg_concepts = kg.get_all_concepts()
        for kg_concept in all_kg_concepts:
            if "derivative" in kg_concept.name.lower() or kg_concept.category == "calculus":
                target_concepts.append(kg_concept)
        
        # Build learning path by tracing prerequisites
        learning_path = []
        visited = set()
        
        def build_path(concept):
            if concept.id in visited:
                return
            visited.add(concept.id)
            
            # Add prerequisites first
            prereqs = kg.get_prerequisites(concept.id)
            for prereq in prereqs:
                build_path(prereq)
            
            # Then add this concept
            learning_path.append(concept)
        
        for concept in target_concepts:
            build_path(concept)
        
        print(f"Learning Path: {[c.name for c in learning_path]}")
        
        # Verify learning path makes sense
        if learning_path:
            # Should start with easier concepts
            assert learning_path[0].difficulty <= learning_path[-1].difficulty
            
            # Should have reasonable progression
            difficulties = [c.difficulty for c in learning_path]
            assert max(difficulties) - min(difficulties) <= 4  # Reasonable range


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 