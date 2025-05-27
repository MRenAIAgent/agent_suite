#!/usr/bin/env python3
"""
Comprehensive tests for Personal Learning Graph functionality.

This test suite validates:
- LearningGraph core functionality
- PersonalizedLearningSystem features
- Integration scenarios
- Real-world learning journeys
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add the parent directory to the path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.learning_graph.user_model import LearningGraph
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem, LearningPattern, WeaknessProfile, LearningInsight
from math_learning.knowledge_graph.graph import KnowledgeGraph, Concept
from math_learning.algebra_learning import AlgebraLearningSystem


class TestLearningGraph:
    """Test suite for LearningGraph core functionality."""
    
    @pytest.fixture
    def learning_graph(self):
        """Create a fresh learning graph for testing."""
        return LearningGraph("test_user_001", "Test User")
    
    @pytest.fixture
    def sample_concepts(self):
        """Sample concept IDs for testing."""
        return ["NS-01", "NS-02", "NS-03", "ALG-01", "ALG-02"]
    
    def test_initialization(self):
        """Test learning graph initialization."""
        lg = LearningGraph("user123", "John Doe")
        
        assert lg.user_id == "user123"
        assert lg.name == "John Doe"
        assert lg.concept_mastery == {}
        assert lg.exercise_history == {}
    
    def test_set_mastery(self, learning_graph):
        """Test setting mastery levels."""
        # Test basic mastery setting
        learning_graph.set_mastery("NS-01", 0.75, 0.85)
        
        assert learning_graph.get_mastery("NS-01") == 0.75
        assert learning_graph.get_confidence("NS-01") == 0.85
        
        # Test mastery bounds
        learning_graph.set_mastery("NS-02", 1.5, -0.1)  # Out of bounds
        assert learning_graph.get_mastery("NS-02") == 1.0  # Clamped to 1.0
        assert learning_graph.get_confidence("NS-02") == 0.0  # Clamped to 0.0
    
    def test_update_mastery(self, learning_graph):
        """Test mastery updates based on exercise performance."""
        concept_id = "NS-01"
        
        # Start with no mastery
        assert learning_graph.get_mastery(concept_id) == 0.0
        
        # Correct answer should increase mastery
        new_mastery = learning_graph.update_mastery(concept_id, True, 0.5, 1.0)
        assert new_mastery > 0.0
        assert learning_graph.get_mastery(concept_id) == new_mastery
        
        # Incorrect answer should decrease mastery
        current_mastery = learning_graph.get_mastery(concept_id)
        new_mastery = learning_graph.update_mastery(concept_id, False, 0.5, 1.0)
        assert new_mastery < current_mastery
    
    def test_record_exercise_attempt(self, learning_graph):
        """Test recording exercise attempts."""
        exercise_id = "ex_001"
        concept_id = "NS-01"
        
        # Record first attempt
        learning_graph.record_exercise_attempt(exercise_id, concept_id, True, 0.6, 1.0)
        
        # Check exercise history
        assert exercise_id in learning_graph.exercise_history
        assert len(learning_graph.exercise_history[exercise_id]) == 1
        
        attempt = learning_graph.exercise_history[exercise_id][0]
        assert attempt["concept_id"] == concept_id
        assert attempt["result"] == True
        assert attempt["difficulty"] == 0.6
        assert "timestamp" in attempt
        
        # Check mastery was updated
        assert learning_graph.get_mastery(concept_id) > 0.0
        
        # Record second attempt
        learning_graph.record_exercise_attempt(exercise_id, concept_id, False, 0.7, 1.0)
        assert len(learning_graph.exercise_history[exercise_id]) == 2
    
    def test_mastery_queries(self, learning_graph, sample_concepts):
        """Test mastery query methods."""
        # Set up some mastery data
        learning_graph.set_mastery(sample_concepts[0], 0.8)  # Mastered
        learning_graph.set_mastery(sample_concepts[1], 0.6)  # Moderate
        learning_graph.set_mastery(sample_concepts[2], 0.2)  # Struggling
        
        # Test mastered concepts
        mastered = learning_graph.get_mastered_concepts(threshold=0.7)
        assert sample_concepts[0] in mastered
        assert sample_concepts[1] not in mastered
        
        # Test struggling concepts
        struggling = learning_graph.get_struggling_concepts(threshold=0.3)
        assert sample_concepts[2] in struggling
        assert sample_concepts[0] not in struggling
    
    def test_serialization(self, learning_graph):
        """Test serialization and deserialization."""
        # Add some data
        learning_graph.set_mastery("NS-01", 0.75, 0.85)
        learning_graph.record_exercise_attempt("ex_001", "NS-01", True, 0.6, 1.0)
        
        # Test to_dict
        data = learning_graph.to_dict()
        assert data["user_id"] == "test_user_001"
        assert "NS-01" in data["concept_mastery"]
        assert "ex_001" in data["exercise_history"]
        
        # Test from_dict
        new_lg = LearningGraph.from_dict(data)
        assert new_lg.user_id == learning_graph.user_id
        assert new_lg.get_mastery("NS-01") == 0.75
        assert "ex_001" in new_lg.exercise_history
    
    def test_file_operations(self, learning_graph):
        """Test saving and loading from files."""
        # Add some data
        learning_graph.set_mastery("NS-01", 0.75)
        learning_graph.record_exercise_attempt("ex_001", "NS-01", True)
        
        # Test save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            learning_graph.save_to_file(temp_file)
            loaded_lg = LearningGraph.load_from_file(temp_file)
            
            assert loaded_lg.user_id == learning_graph.user_id
            assert loaded_lg.get_mastery("NS-01") == 0.75
            assert "ex_001" in loaded_lg.exercise_history
        finally:
            os.unlink(temp_file)


class TestPersonalizedLearningSystem:
    """Test suite for PersonalizedLearningSystem."""
    
    @pytest.fixture
    def knowledge_graph(self):
        """Create a mock knowledge graph for testing."""
        kg = Mock(spec=KnowledgeGraph)
        
        # Create mock concepts
        concepts = {}
        for i in range(5):
            concept = Mock(spec=Concept)
            concept.id = f"concept_{i}"
            concept.name = f"Concept {i}"
            concept.category = "test_category"
            concept.difficulty = i + 1
            concept.prerequisites = set()
            concept.dependents = set()
            concepts[concept.id] = concept
        
        kg.concepts = concepts
        kg.get_concept = lambda cid: concepts.get(cid)
        kg.get_all_concepts = lambda: list(concepts.values())
        
        return kg
    
    @pytest.fixture
    def personalized_system(self, knowledge_graph):
        """Create a personalized learning system for testing."""
        return PersonalizedLearningSystem(knowledge_graph)
    
    @pytest.fixture
    def learning_graph_with_data(self):
        """Create a learning graph with sample data."""
        lg = LearningGraph("test_user", "Test User")
        
        # Simulate learning history
        concepts = ["concept_0", "concept_1", "concept_2"]
        
        for i, concept_id in enumerate(concepts):
            # Different performance patterns for each concept
            success_rate = 0.8 - (i * 0.2)  # Decreasing success rate
            
            for j in range(10):
                exercise_id = f"ex_{concept_id}_{j}"
                success = j < (success_rate * 10)  # First exercises succeed
                difficulty = 0.5 + (j * 0.05)  # Increasing difficulty
                
                lg.record_exercise_attempt(exercise_id, concept_id, success, difficulty, 1.0)
        
        return lg
    
    def test_analyze_learning_patterns(self, personalized_system, learning_graph_with_data):
        """Test learning pattern analysis."""
        patterns = personalized_system.analyze_learning_patterns(learning_graph_with_data)
        
        # Should have patterns for all concepts with data
        assert len(patterns) == 3
        
        for concept_id, pattern in patterns.items():
            assert isinstance(pattern, LearningPattern)
            assert pattern.concept_id == concept_id
            assert pattern.attempts == 10  # 10 exercises per concept
            assert 0.0 <= pattern.success_rate <= 1.0
            assert 0.0 <= pattern.learning_velocity <= 1.0
            assert 0.0 <= pattern.retention_rate <= 1.0
    
    def test_detect_weaknesses(self, personalized_system, learning_graph_with_data):
        """Test weakness detection."""
        weaknesses = personalized_system.detect_weaknesses(learning_graph_with_data)
        
        # Should detect weaknesses for concepts with low performance
        assert len(weaknesses) > 0
        
        for weakness in weaknesses:
            assert isinstance(weakness, WeaknessProfile)
            assert weakness.concept_id in ["concept_0", "concept_1", "concept_2"]
            assert weakness.weakness_type in ["conceptual", "procedural", "computational"]
            assert 0.0 <= weakness.severity <= 1.0
            assert 0.0 <= weakness.confidence_level <= 1.0
            assert len(weakness.suggested_interventions) > 0
    
    def test_adaptive_recommendations(self, personalized_system, learning_graph_with_data):
        """Test adaptive recommendation generation."""
        recommendations = personalized_system.get_adaptive_recommendations(
            learning_graph_with_data, num_recommendations=5
        )
        
        assert len(recommendations) <= 5
        
        for rec in recommendations:
            assert "type" in rec
            assert "concept_id" in rec
            assert "priority" in rec
            assert "reason" in rec
            assert rec["type"] in ["weakness_remediation", "ready_to_learn", "review"]
            assert 0.0 <= rec["priority"] <= 1.0
    
    def test_learning_insights(self, personalized_system, learning_graph_with_data):
        """Test learning insights generation."""
        insights = personalized_system.generate_learning_insights(learning_graph_with_data)
        
        assert len(insights) > 0
        
        for insight in insights:
            assert isinstance(insight, LearningInsight)
            assert insight.insight_type in ["strength", "weakness", "pattern", "recommendation"]
            assert len(insight.title) > 0
            assert len(insight.description) > 0
            assert 0.0 <= insight.confidence <= 1.0
    
    def test_personalized_path_generation(self, personalized_system, learning_graph_with_data):
        """Test personalized learning path generation."""
        target_concept = "concept_4"  # Advanced concept
        
        path = personalized_system.generate_personalized_path(
            learning_graph_with_data, target_concept, max_concepts=5
        )
        
        assert len(path) <= 5
        
        for concept_id, priority in path:
            assert isinstance(concept_id, str)
            assert 0.0 <= priority <= 1.0


class TestIntegrationScenarios:
    """Integration tests for complete learning scenarios."""
    
    @pytest.fixture
    def algebra_system(self):
        """Create algebra learning system for integration tests."""
        return AlgebraLearningSystem()
    
    def test_complete_learning_journey(self, algebra_system):
        """Test a complete learning journey from start to mastery."""
        # Initialize systems
        learning_graph = LearningGraph("integration_test_user")
        personalized_system = PersonalizedLearningSystem(algebra_system.knowledge_graph)
        
        # Get some basic concepts
        concepts = algebra_system.knowledge_graph.get_all_concepts()
        basic_concepts = [c for c in concepts if c.difficulty <= 2][:3]
        
        # Simulate learning progression
        for week in range(4):
            for concept in basic_concepts:
                # Simulate improving performance over time
                base_success_rate = 0.3 + (week * 0.2)  # Improve each week
                
                for exercise_num in range(5):
                    success_rate = min(0.95, base_success_rate + (exercise_num * 0.1))
                    success = success_rate > 0.5  # Simplified success determination
                    
                    exercise_id = f"week_{week}_concept_{concept.id}_ex_{exercise_num}"
                    difficulty = 0.4 + (week * 0.1)  # Increasing difficulty
                    
                    learning_graph.record_exercise_attempt(
                        exercise_id, concept.id, success, difficulty, 1.0
                    )
        
        # Analyze the learning journey
        patterns = personalized_system.analyze_learning_patterns(learning_graph)
        weaknesses = personalized_system.detect_weaknesses(learning_graph)
        recommendations = personalized_system.get_adaptive_recommendations(learning_graph)
        insights = personalized_system.generate_learning_insights(learning_graph)
        
        # Validate results
        assert len(patterns) == len(basic_concepts)
        assert len(recommendations) > 0
        assert len(insights) > 0
        
        # Check that mastery improved over time
        for concept in basic_concepts:
            mastery = learning_graph.get_mastery(concept.id)
            assert mastery > 0.0  # Should have some mastery
    
    def test_weakness_remediation_cycle(self, algebra_system):
        """Test the cycle of detecting and addressing weaknesses."""
        learning_graph = LearningGraph("weakness_test_user")
        personalized_system = PersonalizedLearningSystem(algebra_system.knowledge_graph)
        
        # Get a concept to work with
        concepts = algebra_system.knowledge_graph.get_all_concepts()
        target_concept = next(c for c in concepts if c.difficulty <= 2)
        
        # Phase 1: Create a weakness by failing exercises
        for i in range(10):
            exercise_id = f"failing_ex_{i}"
            learning_graph.record_exercise_attempt(
                exercise_id, target_concept.id, False, 0.5, 1.0
            )
        
        # Detect weaknesses
        weaknesses = personalized_system.detect_weaknesses(learning_graph)
        assert len(weaknesses) > 0
        
        # Find weakness for our target concept
        target_weakness = next(
            (w for w in weaknesses if w.concept_id == target_concept.id), None
        )
        assert target_weakness is not None
        assert target_weakness.severity > 0.5  # Should be significant
        
        # Phase 2: Address the weakness with successful exercises
        for i in range(15):
            exercise_id = f"remediation_ex_{i}"
            learning_graph.record_exercise_attempt(
                exercise_id, target_concept.id, True, 0.4, 1.0  # Easier exercises
            )
        
        # Check that weakness is reduced
        new_weaknesses = personalized_system.detect_weaknesses(learning_graph)
        new_target_weakness = next(
            (w for w in new_weaknesses if w.concept_id == target_concept.id), None
        )
        
        # Weakness should be reduced or eliminated
        if new_target_weakness:
            assert new_target_weakness.severity < target_weakness.severity
    
    def test_adaptive_difficulty_progression(self, algebra_system):
        """Test that the system can track difficulty progression."""
        learning_graph = LearningGraph("difficulty_test_user")
        personalized_system = PersonalizedLearningSystem(algebra_system.knowledge_graph)
        
        # Get a concept
        concepts = algebra_system.knowledge_graph.get_all_concepts()
        concept = next(c for c in concepts if c.difficulty <= 2)
        
        # Simulate progressive difficulty increase
        difficulties = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        for i, difficulty in enumerate(difficulties):
            # Success rate should be high for easier exercises
            success_rate = max(0.1, 1.0 - difficulty)
            success = success_rate > 0.5
            
            exercise_id = f"progressive_ex_{i}"
            learning_graph.record_exercise_attempt(
                exercise_id, concept.id, success, difficulty, 1.0
            )
        
        # Analyze patterns
        patterns = personalized_system.analyze_learning_patterns(learning_graph)
        concept_pattern = patterns.get(concept.id)
        
        assert concept_pattern is not None
        assert len(concept_pattern.difficulty_progression) > 0
        
        # Check that mastery reflects the difficulty progression
        final_mastery = learning_graph.get_mastery(concept.id)
        assert 0.0 <= final_mastery <= 1.0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_empty_learning_graph(self):
        """Test behavior with empty learning graph."""
        lg = LearningGraph("empty_user")
        kg = Mock(spec=KnowledgeGraph)
        kg.concepts = {}
        kg.get_concept = lambda cid: None
        kg.get_all_concepts = lambda: []
        
        ps = PersonalizedLearningSystem(kg)
        
        # Should handle empty data gracefully
        patterns = ps.analyze_learning_patterns(lg)
        assert patterns == {}
        
        weaknesses = ps.detect_weaknesses(lg)
        assert weaknesses == []
        
        recommendations = ps.get_adaptive_recommendations(lg)
        assert len(recommendations) == 0
        
        insights = ps.generate_learning_insights(lg)
        assert len(insights) == 0
    
    def test_invalid_mastery_values(self):
        """Test handling of invalid mastery values."""
        lg = LearningGraph("test_user")
        
        # Test extreme values
        lg.set_mastery("concept_1", -5.0, 2.0)  # Out of bounds
        assert lg.get_mastery("concept_1") == 0.0  # Should be clamped
        assert lg.get_confidence("concept_1") == 1.0  # Should be clamped
    
    def test_missing_concept_data(self, algebra_system):
        """Test behavior when concept data is missing."""
        lg = LearningGraph("test_user")
        ps = PersonalizedLearningSystem(algebra_system.knowledge_graph)
        
        # Record exercise for non-existent concept
        lg.record_exercise_attempt("ex_1", "non_existent_concept", True, 0.5, 1.0)
        
        # Should handle gracefully
        patterns = ps.analyze_learning_patterns(lg)
        # Should still create pattern for the concept even if not in knowledge graph
        assert "non_existent_concept" in patterns
    
    def test_single_exercise_patterns(self, algebra_system):
        """Test pattern analysis with minimal data."""
        lg = LearningGraph("minimal_user")
        ps = PersonalizedLearningSystem(algebra_system.knowledge_graph)
        
        # Record just one exercise
        lg.record_exercise_attempt("single_ex", "concept_1", True, 0.5, 1.0)
        
        # Should handle minimal data
        patterns = ps.analyze_learning_patterns(lg)
        # Might not have enough data for meaningful patterns
        # But should not crash


class TestPerformanceAndScalability:
    """Test performance with larger datasets."""
    
    def test_large_exercise_history(self, algebra_system):
        """Test performance with large exercise history."""
        lg = LearningGraph("performance_user")
        ps = PersonalizedLearningSystem(algebra_system.knowledge_graph)
        
        # Create large dataset
        concepts = algebra_system.knowledge_graph.get_all_concepts()
        test_concepts = list(concepts)[:5]  # Limit for test performance
        
        # Record many exercises
        for concept in test_concepts:
            for i in range(100):  # 100 exercises per concept
                exercise_id = f"perf_ex_{concept.id}_{i}"
                success = i % 3 != 0  # 2/3 success rate
                difficulty = 0.3 + (i / 100) * 0.4  # Progressive difficulty
                
                lg.record_exercise_attempt(exercise_id, concept.id, success, difficulty, 1.0)
        
        # Test that analysis still works efficiently
        import time
        start_time = time.time()
        
        patterns = ps.analyze_learning_patterns(lg)
        weaknesses = ps.detect_weaknesses(lg)
        recommendations = ps.get_adaptive_recommendations(lg)
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0
        
        # Should produce meaningful results
        assert len(patterns) == len(test_concepts)
        assert len(recommendations) > 0
    
    def test_temporal_analysis_performance(self, algebra_system):
        """Test performance of temporal analysis over long periods."""
        lg = LearningGraph("temporal_user")
        ps = PersonalizedLearningSystem(algebra_system.knowledge_graph)
        
        # Simulate learning over many sessions
        concepts = list(algebra_system.knowledge_graph.get_all_concepts())[:3]
        
        # Create exercises spread over time
        base_time = datetime.now() - timedelta(days=365)  # One year ago
        
        for day in range(0, 365, 7):  # Weekly sessions
            session_time = base_time + timedelta(days=day)
            
            for concept in concepts:
                for ex_num in range(5):  # 5 exercises per session
                    exercise_id = f"temporal_ex_{day}_{concept.id}_{ex_num}"
                    success = (day + ex_num) % 4 != 0  # Varying success
                    difficulty = 0.4 + (day / 365) * 0.3  # Progressive difficulty
                    
                    lg.record_exercise_attempt(exercise_id, concept.id, success, difficulty, 1.0)
        
        # Test temporal analysis
        patterns = ps.analyze_learning_patterns(lg)
        insights = ps.generate_learning_insights(lg)
        
        # Should handle long-term data effectively
        assert len(patterns) == len(concepts)
        assert len(insights) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 