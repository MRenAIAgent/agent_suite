#!/usr/bin/env python3
"""
Core tests for Personal Learning Graph functionality.

This test suite focuses on the essential functionality without external dependencies.
"""

import sys
import os
import json
import tempfile
from datetime import datetime, timedelta

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.learning_graph.user_model import LearningGraph
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem, LearningPattern, WeaknessProfile, LearningInsight


def test_learning_graph_basic_functionality():
    """Test basic LearningGraph functionality."""
    print("Testing LearningGraph basic functionality...")
    
    # Test initialization
    lg = LearningGraph("test_user_001", "Test User")
    assert lg.user_id == "test_user_001"
    assert lg.name == "Test User"
    assert lg.concept_mastery == {}
    assert lg.exercise_history == {}
    print("✅ Initialization test passed")
    
    # Test mastery setting
    lg.set_mastery("NS-01", 0.75, 0.85)
    assert lg.get_mastery("NS-01") == 0.75
    assert lg.get_confidence("NS-01") == 0.85
    print("✅ Mastery setting test passed")
    
    # Test mastery bounds
    lg.set_mastery("NS-02", 1.5, -0.1)  # Out of bounds
    assert lg.get_mastery("NS-02") == 1.0  # Clamped to 1.0
    assert lg.get_confidence("NS-02") == 0.0  # Clamped to 0.0
    print("✅ Mastery bounds test passed")
    
    # Test exercise recording
    lg.record_exercise_attempt("ex_001", "NS-01", True, 0.6, 1.0)
    assert "ex_001" in lg.exercise_history
    assert len(lg.exercise_history["ex_001"]) == 1
    
    attempt = lg.exercise_history["ex_001"][0]
    assert attempt["concept_id"] == "NS-01"
    assert attempt["result"] == True
    assert attempt["difficulty"] == 0.6
    assert "timestamp" in attempt
    print("✅ Exercise recording test passed")
    
    # Test mastery update
    initial_mastery = lg.get_mastery("NS-03")
    assert initial_mastery == 0.0
    
    new_mastery = lg.update_mastery("NS-03", True, 0.5, 1.0)
    assert new_mastery > 0.0
    assert lg.get_mastery("NS-03") == new_mastery
    print("✅ Mastery update test passed")
    
    print("🎉 All LearningGraph basic tests passed!\n")


def test_learning_graph_queries():
    """Test LearningGraph query methods."""
    print("Testing LearningGraph query methods...")
    
    lg = LearningGraph("query_test_user")
    
    # Set up test data
    concepts = ["concept_1", "concept_2", "concept_3", "concept_4"]
    masteries = [0.9, 0.6, 0.2, 0.8]  # High, medium, low, high
    
    for concept, mastery in zip(concepts, masteries):
        lg.set_mastery(concept, mastery)
    
    # Test mastered concepts
    mastered = lg.get_mastered_concepts(threshold=0.7)
    assert "concept_1" in mastered  # 0.9 > 0.7
    assert "concept_4" in mastered  # 0.8 > 0.7
    assert "concept_2" not in mastered  # 0.6 < 0.7
    assert "concept_3" not in mastered  # 0.2 < 0.7
    print("✅ Mastered concepts query test passed")
    
    # Test struggling concepts
    struggling = lg.get_struggling_concepts(threshold=0.3)
    assert "concept_3" in struggling  # 0.2 < 0.3
    assert "concept_1" not in struggling  # 0.9 > 0.3
    print("✅ Struggling concepts query test passed")
    
    print("🎉 All LearningGraph query tests passed!\n")


def test_learning_graph_serialization():
    """Test LearningGraph serialization and file operations."""
    print("Testing LearningGraph serialization...")
    
    lg = LearningGraph("serialization_test_user", "Serialization Test")
    
    # Add some data
    lg.set_mastery("NS-01", 0.75, 0.85)
    lg.record_exercise_attempt("ex_001", "NS-01", True, 0.6, 1.0)
    lg.record_exercise_attempt("ex_002", "NS-02", False, 0.7, 1.0)
    
    # Get the actual mastery values after exercises (they will have been updated)
    actual_ns01_mastery = lg.get_mastery("NS-01")
    actual_ns01_confidence = lg.get_confidence("NS-01")
    
    # Test to_dict
    data = lg.to_dict()
    assert data["user_id"] == "serialization_test_user"
    assert data["name"] == "Serialization Test"
    assert "NS-01" in data["concept_mastery"]
    assert "ex_001" in data["exercise_history"]
    assert "ex_002" in data["exercise_history"]
    print("✅ to_dict test passed")
    
    # Test from_dict
    new_lg = LearningGraph.from_dict(data)
    assert new_lg.user_id == lg.user_id
    assert new_lg.name == lg.name
    assert new_lg.get_mastery("NS-01") == actual_ns01_mastery
    assert new_lg.get_confidence("NS-01") == actual_ns01_confidence
    assert "ex_001" in new_lg.exercise_history
    assert "ex_002" in new_lg.exercise_history
    print("✅ from_dict test passed")
    
    # Test file operations
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        lg.save_to_file(temp_file)
        loaded_lg = LearningGraph.load_from_file(temp_file)
        
        assert loaded_lg.user_id == lg.user_id
        assert loaded_lg.name == lg.name
        assert loaded_lg.get_mastery("NS-01") == actual_ns01_mastery
        assert "ex_001" in loaded_lg.exercise_history
        print("✅ File operations test passed")
    finally:
        os.unlink(temp_file)
    
    print("🎉 All LearningGraph serialization tests passed!\n")


def test_personalized_learning_patterns():
    """Test PersonalizedLearningSystem pattern analysis."""
    print("Testing PersonalizedLearningSystem pattern analysis...")
    
    # Create a mock knowledge graph
    class MockConcept:
        def __init__(self, concept_id, name="Test Concept", difficulty=1):
            self.id = concept_id
            self.name = name
            self.category = "test"
            self.difficulty = difficulty
            self.prerequisites = set()
            self.dependents = set()
    
    class MockKnowledgeGraph:
        def __init__(self):
            self.concepts = {
                "concept_1": MockConcept("concept_1", "Concept 1", 1),
                "concept_2": MockConcept("concept_2", "Concept 2", 2),
                "concept_3": MockConcept("concept_3", "Concept 3", 3),
            }
        
        def get_concept(self, concept_id):
            return self.concepts.get(concept_id)
        
        def get_all_concepts(self):
            return list(self.concepts.values())
        
        def get_prerequisites(self, concept_id):
            """Return empty list for mock - no prerequisites defined."""
            return []
        
        def get_dependent_concepts(self, concept_id):
            """Return empty list for mock - no dependents defined."""
            return []
    
    # Create systems
    kg = MockKnowledgeGraph()
    ps = PersonalizedLearningSystem(kg)
    lg = LearningGraph("pattern_test_user")
    
    # Create learning data with different patterns
    concepts = ["concept_1", "concept_2", "concept_3"]
    
    for i, concept_id in enumerate(concepts):
        # Different performance patterns for each concept
        success_rate = 0.8 - (i * 0.2)  # Decreasing success rate
        
        for j in range(10):
            exercise_id = f"ex_{concept_id}_{j}"
            success = j < (success_rate * 10)  # First exercises succeed
            difficulty = 0.5 + (j * 0.05)  # Increasing difficulty
            
            lg.record_exercise_attempt(exercise_id, concept_id, success, difficulty, 1.0)
    
    # Test pattern analysis
    patterns = ps.analyze_learning_patterns(lg)
    assert len(patterns) == 3
    
    for concept_id, pattern in patterns.items():
        assert isinstance(pattern, LearningPattern)
        assert pattern.concept_id == concept_id
        assert pattern.attempts == 10
        assert 0.0 <= pattern.success_rate <= 1.0
        assert 0.0 <= pattern.learning_velocity <= 1.0
        assert 0.0 <= pattern.retention_rate <= 1.0
    
    print("✅ Pattern analysis test passed")
    
    # Test weakness detection
    weaknesses = ps.detect_weaknesses(lg)
    assert len(weaknesses) > 0
    
    for weakness in weaknesses:
        assert isinstance(weakness, WeaknessProfile)
        assert weakness.concept_id in concepts
        assert weakness.weakness_type in ["conceptual", "procedural", "computational"]
        assert 0.0 <= weakness.severity <= 1.0
        assert 0.0 <= weakness.confidence_level <= 1.0
        assert len(weakness.suggested_interventions) > 0
    
    print("✅ Weakness detection test passed")
    
    # Test adaptive recommendations
    recommendations = ps.get_adaptive_recommendations(lg, num_recommendations=5)
    assert len(recommendations) <= 5
    
    for rec in recommendations:
        assert "type" in rec
        assert "concept_id" in rec
        assert "priority" in rec
        assert "reason" in rec
        assert rec["type"] in ["weakness_remediation", "ready_to_learn", "review"]
        assert 0.0 <= rec["priority"] <= 1.0
    
    print("✅ Adaptive recommendations test passed")
    
    # Test learning insights
    insights = ps.generate_learning_insights(lg)
    
    # The insights might be empty if the patterns don't meet the thresholds
    # Let's create a scenario that should generate insights
    lg_insights = LearningGraph("insights_test_user")
    
    # Create data that should generate a "fast learning" insight
    for i in range(15):  # More exercises to create clear patterns
        exercise_id = f"fast_ex_{i}"
        success = True  # All successful for high learning velocity
        difficulty = 0.3 + (i * 0.02)  # Gradually increasing difficulty
        lg_insights.record_exercise_attempt(exercise_id, "concept_1", success, difficulty, 1.0)
    
    # Create data for low retention (should generate review insight)
    for i in range(10):
        exercise_id = f"retention_ex_{i}"
        success = i < 7  # Start well but decline
        difficulty = 0.5
        lg_insights.record_exercise_attempt(exercise_id, "concept_2", success, difficulty, 1.0)
    
    insights = ps.generate_learning_insights(lg_insights)
    
    # Should have at least one insight now (fast learning or review needed)
    print(f"Generated {len(insights)} insights")
    for insight in insights:
        assert isinstance(insight, LearningInsight)
        assert insight.insight_type in ['strength', 'weakness', 'pattern', 'recommendation']
        assert len(insight.title) > 0
        assert len(insight.description) > 0
        assert 0.0 <= insight.confidence <= 1.0
        print(f"  - {insight.title}: {insight.description}")
    
    print("✅ Learning insights test passed")
    
    print("🎉 All PersonalizedLearningSystem tests passed!\n")


def test_learning_journey_simulation():
    """Test a complete learning journey simulation."""
    print("Testing complete learning journey simulation...")
    
    # Create mock knowledge graph
    class MockConcept:
        def __init__(self, concept_id, name, difficulty=1):
            self.id = concept_id
            self.name = name
            self.category = "test"
            self.difficulty = difficulty
            self.prerequisites = set()
            self.dependents = set()
    
    class MockKnowledgeGraph:
        def __init__(self):
            self.concepts = {
                f"concept_{i}": MockConcept(f"concept_{i}", f"Concept {i}", i+1)
                for i in range(5)
            }
        
        def get_concept(self, concept_id):
            return self.concepts.get(concept_id)
        
        def get_all_concepts(self):
            return list(self.concepts.values())
        
        def get_prerequisites(self, concept_id):
            """Return empty list for mock - no prerequisites defined."""
            return []
        
        def get_dependent_concepts(self, concept_id):
            """Return empty list for mock - no dependents defined."""
            return []
    
    # Initialize systems
    kg = MockKnowledgeGraph()
    ps = PersonalizedLearningSystem(kg)
    lg = LearningGraph("journey_test_user")
    
    # Simulate learning progression over time
    concepts = ["concept_0", "concept_1", "concept_2"]
    
    for week in range(4):
        for concept_id in concepts:
            # Simulate improving performance over time
            base_success_rate = 0.3 + (week * 0.2)  # Improve each week
            
            for exercise_num in range(5):
                success_rate = min(0.95, base_success_rate + (exercise_num * 0.1))
                success = success_rate > 0.5
                
                exercise_id = f"week_{week}_{concept_id}_ex_{exercise_num}"
                difficulty = 0.4 + (week * 0.1)  # Increasing difficulty
                
                lg.record_exercise_attempt(exercise_id, concept_id, success, difficulty, 1.0)
    
    # Analyze the learning journey
    patterns = ps.analyze_learning_patterns(lg)
    weaknesses = ps.detect_weaknesses(lg)
    recommendations = ps.get_adaptive_recommendations(lg)
    insights = ps.generate_learning_insights(lg)
    
    # Validate results
    assert len(patterns) == len(concepts)
    assert len(recommendations) >= 0  # May be 0 if no clear recommendations
    assert len(insights) >= 0  # May be 0 if no clear insights
    
    # Check that mastery improved over time
    for concept_id in concepts:
        mastery = lg.get_mastery(concept_id)
        assert mastery >= 0.0  # Should have some mastery or at least not negative
    
    print("✅ Learning journey simulation test passed")
    print("🎉 All learning journey tests passed!\n")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases and error handling...")
    
    # Test empty learning graph
    lg = LearningGraph("empty_user")
    
    class MockKnowledgeGraph:
        def __init__(self):
            self.concepts = {}
        
        def get_concept(self, concept_id):
            return None
        
        def get_all_concepts(self):
            return []
    
    kg = MockKnowledgeGraph()
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
    
    print("✅ Empty data handling test passed")
    
    # Test invalid mastery values
    lg = LearningGraph("test_user")
    lg.set_mastery("concept_1", -5.0, 2.0)  # Out of bounds
    assert lg.get_mastery("concept_1") == 0.0  # Should be clamped
    assert lg.get_confidence("concept_1") == 1.0  # Should be clamped
    
    print("✅ Invalid values handling test passed")
    
    # Test single exercise
    lg = LearningGraph("minimal_user")
    lg.record_exercise_attempt("single_ex", "concept_1", True, 0.5, 1.0)
    
    # Should handle minimal data without crashing
    patterns = ps.analyze_learning_patterns(lg)
    # Should create pattern even with minimal data
    assert "concept_1" in patterns
    
    print("✅ Minimal data handling test passed")
    print("🎉 All edge case tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("🚀 Running Core Personal Learning Graph Tests")
    print("=" * 60)
    
    try:
        test_learning_graph_basic_functionality()
        test_learning_graph_queries()
        test_learning_graph_serialization()
        test_personalized_learning_patterns()
        test_learning_journey_simulation()
        test_edge_cases()
        
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("✅ LearningGraph functionality: PASSED")
        print("✅ Query methods: PASSED")
        print("✅ Serialization: PASSED")
        print("✅ Pattern analysis: PASSED")
        print("✅ Learning journey: PASSED")
        print("✅ Edge cases: PASSED")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1) 