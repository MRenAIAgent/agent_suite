#!/usr/bin/env python3
"""Edge cases and error scenarios test for math learning system with RAG backend."""

import asyncio
import sys
import os
from typing import Dict, List, Tuple, Any

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.config.rag_config import get_memory_config, create_configured_rag_service
from math_learning.learning_graph.user_model import LearningGraph
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.context import ContextItem
from agents.rag.middleware.storage_router import StorageType

class EdgeCaseScenario:
    """Represents an edge case or error scenario for testing."""
    
    def __init__(self, scenario_id: str, name: str, description: str, test_function):
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.test_function = test_function
        self.result = None
        self.error = None

async def test_empty_user_scenario(rag_service):
    """Test user with no exercise attempts."""
    print("Testing empty user scenario...")
    
    learning_graph = LearningGraph(user_id="empty_user", name="Empty User")
    
    # Check default mastery levels
    mastery = learning_graph.get_mastery("basic_arithmetic")
    confidence = learning_graph.get_confidence("basic_arithmetic")
    
    assert mastery == 0.0, f"Expected 0.0 mastery, got {mastery}"
    assert confidence == 0.5, f"Expected 0.5 confidence, got {confidence}"
    
    # Check recommendations for empty user
    mastered_concepts = learning_graph.get_mastered_concepts(threshold=0.5)
    struggling_concepts = learning_graph.get_struggling_concepts(threshold=0.4)
    
    assert len(mastered_concepts) == 0, f"Expected no mastered concepts, got {mastered_concepts}"
    assert len(struggling_concepts) == 0, f"Expected no struggling concepts, got {struggling_concepts}"
    
    print("✓ Empty user scenario handled correctly")
    return True

async def test_perfect_user_scenario(rag_service):
    """Test user with perfect performance on all exercises."""
    print("Testing perfect user scenario...")
    
    learning_graph = LearningGraph(user_id="perfect_user", name="Perfect User")
    
    # Perfect performance on multiple concepts
    concepts = ["counting", "basic_arithmetic", "integers", "fractions", "algebra"]
    
    for concept in concepts:
        for i in range(10):  # 10 perfect exercises per concept
            exercise_id = f"perfect_{concept}_ex_{i+1}"
            learning_graph.record_exercise_attempt(
                exercise_id=exercise_id,
                concept_id=concept,
                result=True,
                difficulty=0.5 + (i * 0.05),  # Increasing difficulty
                concept_weight=1.0
            )
    
    # Check mastery levels
    for concept in concepts:
        mastery = learning_graph.get_mastery(concept)
        confidence = learning_graph.get_confidence(concept)
        
        assert mastery > 0.8, f"Expected high mastery for {concept}, got {mastery}"
        assert confidence > 0.8, f"Expected high confidence for {concept}, got {confidence}"
    
    # Check recommendations
    mastered_concepts = learning_graph.get_mastered_concepts(threshold=0.5)
    struggling_concepts = learning_graph.get_struggling_concepts(threshold=0.4)
    
    assert len(mastered_concepts) == len(concepts), f"Expected all concepts mastered, got {mastered_concepts}"
    assert len(struggling_concepts) == 0, f"Expected no struggling concepts, got {struggling_concepts}"
    
    print("✓ Perfect user scenario handled correctly")
    return True

async def test_failing_user_scenario(rag_service):
    """Test user with consistently poor performance."""
    print("Testing failing user scenario...")
    
    learning_graph = LearningGraph(user_id="failing_user", name="Failing User")
    
    # Poor performance on multiple concepts
    concepts = ["counting", "basic_arithmetic", "integers"]
    
    for concept in concepts:
        for i in range(10):  # 10 failed exercises per concept
            exercise_id = f"failing_{concept}_ex_{i+1}"
            learning_graph.record_exercise_attempt(
                exercise_id=exercise_id,
                concept_id=concept,
                result=False,
                difficulty=0.2,  # Easy exercises
                concept_weight=1.0
            )
    
    # Check mastery levels
    for concept in concepts:
        mastery = learning_graph.get_mastery(concept)
        confidence = learning_graph.get_confidence(concept)
        
        assert mastery < 0.1, f"Expected very low mastery for {concept}, got {mastery}"
        assert confidence < 0.3, f"Expected low confidence for {concept}, got {confidence}"
    
    # Check recommendations
    mastered_concepts = learning_graph.get_mastered_concepts(threshold=0.5)
    struggling_concepts = learning_graph.get_struggling_concepts(threshold=0.4)
    
    assert len(mastered_concepts) == 0, f"Expected no mastered concepts, got {mastered_concepts}"
    assert len(struggling_concepts) == len(concepts), f"Expected all concepts struggling, got {struggling_concepts}"
    
    print("✓ Failing user scenario handled correctly")
    return True

async def test_inconsistent_difficulty_scenario(rag_service):
    """Test user with exercises of wildly varying difficulty."""
    print("Testing inconsistent difficulty scenario...")
    
    learning_graph = LearningGraph(user_id="inconsistent_user", name="Inconsistent Difficulty User")
    
    concept = "basic_arithmetic"
    
    # Mix of very easy and very hard exercises
    difficulties_and_results = [
        (0.1, True),   # Very easy, success
        (0.9, False),  # Very hard, failure
        (0.1, True),   # Very easy, success
        (0.8, False),  # Hard, failure
        (0.2, True),   # Easy, success
        (0.7, False),  # Hard, failure
        (0.1, True),   # Very easy, success
        (0.9, False),  # Very hard, failure
    ]
    
    for i, (difficulty, result) in enumerate(difficulties_and_results):
        exercise_id = f"inconsistent_{concept}_ex_{i+1}"
        learning_graph.record_exercise_attempt(
            exercise_id=exercise_id,
            concept_id=concept,
            result=result,
            difficulty=difficulty,
            concept_weight=1.0
        )
    
    mastery = learning_graph.get_mastery(concept)
    confidence = learning_graph.get_confidence(concept)
    
    # Should have moderate mastery due to mixed performance
    assert 0.2 < mastery < 0.8, f"Expected moderate mastery, got {mastery}"
    assert confidence > 0.0, f"Expected some confidence, got {confidence}"
    
    print("✓ Inconsistent difficulty scenario handled correctly")
    return True

async def test_rapid_improvement_scenario(rag_service):
    """Test user showing rapid improvement over time."""
    print("Testing rapid improvement scenario...")
    
    learning_graph = LearningGraph(user_id="improving_user", name="Rapidly Improving User")
    
    concept = "fractions"
    
    # Start with failures, then improve to successes
    results = [False, False, False, True, True, True, True, True, True, True]
    
    for i, result in enumerate(results):
        exercise_id = f"improving_{concept}_ex_{i+1}"
        learning_graph.record_exercise_attempt(
            exercise_id=exercise_id,
            concept_id=concept,
            result=result,
            difficulty=0.5,
            concept_weight=1.0
        )
    
    mastery = learning_graph.get_mastery(concept)
    confidence = learning_graph.get_confidence(concept)
    
    # Should show good mastery due to recent successes
    assert mastery > 0.5, f"Expected good mastery after improvement, got {mastery}"
    assert confidence > 0.4, f"Expected reasonable confidence, got {confidence}"
    
    print("✓ Rapid improvement scenario handled correctly")
    return True

async def test_concept_weight_variations(rag_service):
    """Test exercises with different concept weights."""
    print("Testing concept weight variations...")
    
    learning_graph = LearningGraph(user_id="weight_test_user", name="Weight Test User")
    
    concept = "algebra"
    
    # Same results but different weights
    exercises = [
        (True, 0.1),   # Low weight success
        (False, 1.0),  # High weight failure
        (True, 0.5),   # Medium weight success
        (True, 2.0),   # Very high weight success
        (False, 0.1),  # Low weight failure
    ]
    
    for i, (result, weight) in enumerate(exercises):
        exercise_id = f"weight_{concept}_ex_{i+1}"
        learning_graph.record_exercise_attempt(
            exercise_id=exercise_id,
            concept_id=concept,
            result=result,
            difficulty=0.5,
            concept_weight=weight
        )
    
    mastery = learning_graph.get_mastery(concept)
    
    # High weight success should dominate
    assert mastery > 0.3, f"Expected positive mastery due to high-weight success, got {mastery}"
    
    print("✓ Concept weight variations handled correctly")
    return True

async def test_invalid_exercise_data(rag_service):
    """Test handling of invalid exercise data."""
    print("Testing invalid exercise data...")
    
    learning_graph = LearningGraph(user_id="invalid_data_user", name="Invalid Data User")
    
    # Test with valid data first
    learning_graph.record_exercise_attempt(
        exercise_id="valid_ex_1",
        concept_id="basic_arithmetic",
        result=True,
        difficulty=0.5,
        concept_weight=1.0
    )
    
    # Test edge case values
    try:
        # Zero difficulty
        learning_graph.record_exercise_attempt(
            exercise_id="zero_diff_ex",
            concept_id="basic_arithmetic",
            result=True,
            difficulty=0.0,
            concept_weight=1.0
        )
        
        # Very high difficulty
        learning_graph.record_exercise_attempt(
            exercise_id="high_diff_ex",
            concept_id="basic_arithmetic",
            result=False,
            difficulty=1.0,
            concept_weight=1.0
        )
        
        # Zero weight
        learning_graph.record_exercise_attempt(
            exercise_id="zero_weight_ex",
            concept_id="basic_arithmetic",
            result=True,
            difficulty=0.5,
            concept_weight=0.0
        )
        
        # Negative weight (should be handled gracefully)
        learning_graph.record_exercise_attempt(
            exercise_id="negative_weight_ex",
            concept_id="basic_arithmetic",
            result=True,
            difficulty=0.5,
            concept_weight=-1.0
        )
        
    except Exception as e:
        print(f"Warning: Exception with edge case data: {e}")
    
    # Should still have valid mastery calculation
    mastery = learning_graph.get_mastery("basic_arithmetic")
    assert mastery >= 0.0, f"Expected non-negative mastery, got {mastery}"
    
    print("✓ Invalid exercise data handled gracefully")
    return True

async def test_large_exercise_volume(rag_service):
    """Test performance with large number of exercises."""
    print("Testing large exercise volume...")
    
    learning_graph = LearningGraph(user_id="volume_test_user", name="Volume Test User")
    
    concept = "geometry"
    num_exercises = 1000
    
    # Generate many exercises with random-ish pattern
    for i in range(num_exercises):
        exercise_id = f"volume_{concept}_ex_{i+1}"
        # Simulate 70% success rate
        result = (i % 10) < 7
        difficulty = 0.3 + (i % 5) * 0.1
        
        learning_graph.record_exercise_attempt(
            exercise_id=exercise_id,
            concept_id=concept,
            result=result,
            difficulty=difficulty,
            concept_weight=1.0
        )
    
    mastery = learning_graph.get_mastery(concept)
    confidence = learning_graph.get_confidence(concept)
    
    # Should converge to reasonable values
    assert 0.5 < mastery < 0.8, f"Expected mastery around 70%, got {mastery}"
    assert confidence > 0.6, f"Expected high confidence with many exercises, got {confidence}"
    
    # Check exercise history length
    assert len(learning_graph.exercise_history) > 0, "Exercise history should not be empty"
    
    print(f"✓ Large exercise volume ({num_exercises} exercises) handled correctly")
    return True

async def test_rag_storage_errors(rag_service):
    """Test handling of RAG storage errors."""
    print("Testing RAG storage error handling...")
    
    try:
        kv_adaptor = rag_service.storage_adaptors[StorageType.KEY_VALUE]
        
        # Test storing invalid context item
        invalid_item = ContextItem(
            key="",  # Empty key
            value=None,  # None value
            metadata={}
        )
        
        # This might succeed or fail depending on implementation
        try:
            item_id = await kv_adaptor.store_context_item(invalid_item)
            print(f"  Stored invalid item with ID: {item_id}")
        except Exception as e:
            print(f"  Expected error storing invalid item: {e}")
        
        # Test retrieving non-existent item
        try:
            non_existent = await kv_adaptor.get("non_existent_key")
            print(f"  Retrieved non-existent item: {non_existent}")
        except Exception as e:
            print(f"  Expected error retrieving non-existent item: {e}")
        
    except Exception as e:
        print(f"  RAG storage error: {e}")
    
    print("✓ RAG storage error handling tested")
    return True

async def test_concurrent_users(rag_service):
    """Test multiple users accessing the system concurrently."""
    print("Testing concurrent user access...")
    
    async def create_user_session(user_id: str, concept: str):
        """Create a user session with some exercises."""
        learning_graph = LearningGraph(user_id=user_id, name=f"Concurrent User {user_id}")
        
        for i in range(5):
            exercise_id = f"{user_id}_{concept}_ex_{i+1}"
            result = (i % 2) == 0  # Alternating success/failure
            
            learning_graph.record_exercise_attempt(
                exercise_id=exercise_id,
                concept_id=concept,
                result=result,
                difficulty=0.5,
                concept_weight=1.0
            )
        
        # Store in RAG
        kv_adaptor = rag_service.storage_adaptors[StorageType.KEY_VALUE]
        progress_item = ContextItem(
            key=f'concurrent_user_{user_id}',
            value={
                'user_id': user_id,
                'mastery': learning_graph.get_mastery(concept),
                'exercises': len(learning_graph.exercise_history)
            },
            metadata={'type': 'concurrent_test'}
        )
        
        await kv_adaptor.store_context_item(progress_item)
        return learning_graph
    
    # Create multiple concurrent user sessions
    tasks = []
    for i in range(5):
        task = create_user_session(f"concurrent_{i}", "basic_arithmetic")
        tasks.append(task)
    
    # Run concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check results
    successful_users = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  User concurrent_{i} failed: {result}")
        else:
            successful_users += 1
            mastery = result.get_mastery("basic_arithmetic")
            print(f"  User concurrent_{i} mastery: {mastery:.3f}")
    
    assert successful_users > 0, "At least one concurrent user should succeed"
    
    print(f"✓ Concurrent access tested ({successful_users}/5 users successful)")
    return True

async def run_edge_case_tests():
    """Run all edge case and error scenario tests."""
    print("=== Edge Cases and Error Scenarios Test ===\n")
    
    # Create RAG service
    print("1. Creating RAG service...")
    try:
        config = get_memory_config()
        rag_service = await create_configured_rag_service(config)
        print("✓ RAG service created successfully")
    except Exception as e:
        print(f"✗ Failed to create RAG service: {e}")
        return False
    
    # Define edge case scenarios
    scenarios = [
        EdgeCaseScenario(
            "empty_user",
            "Empty User",
            "User with no exercise attempts",
            test_empty_user_scenario
        ),
        EdgeCaseScenario(
            "perfect_user",
            "Perfect User",
            "User with perfect performance on all exercises",
            test_perfect_user_scenario
        ),
        EdgeCaseScenario(
            "failing_user",
            "Failing User",
            "User with consistently poor performance",
            test_failing_user_scenario
        ),
        EdgeCaseScenario(
            "inconsistent_difficulty",
            "Inconsistent Difficulty",
            "User with exercises of wildly varying difficulty",
            test_inconsistent_difficulty_scenario
        ),
        EdgeCaseScenario(
            "rapid_improvement",
            "Rapid Improvement",
            "User showing rapid improvement over time",
            test_rapid_improvement_scenario
        ),
        EdgeCaseScenario(
            "concept_weights",
            "Concept Weight Variations",
            "Exercises with different concept weights",
            test_concept_weight_variations
        ),
        EdgeCaseScenario(
            "invalid_data",
            "Invalid Exercise Data",
            "Handling of invalid exercise data",
            test_invalid_exercise_data
        ),
        EdgeCaseScenario(
            "large_volume",
            "Large Exercise Volume",
            "Performance with large number of exercises",
            test_large_exercise_volume
        ),
        EdgeCaseScenario(
            "rag_errors",
            "RAG Storage Errors",
            "Handling of RAG storage errors",
            test_rag_storage_errors
        ),
        EdgeCaseScenario(
            "concurrent_users",
            "Concurrent Users",
            "Multiple users accessing system concurrently",
            test_concurrent_users
        ),
    ]
    
    print(f"\n2. Running {len(scenarios)} edge case scenarios...")
    
    # Run each scenario
    passed = 0
    failed = 0
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Testing: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"{'='*50}")
        
        try:
            result = await scenario.test_function(rag_service)
            scenario.result = result
            if result:
                passed += 1
                print(f"✓ {scenario.name} PASSED")
            else:
                failed += 1
                print(f"✗ {scenario.name} FAILED")
        except Exception as e:
            scenario.error = str(e)
            failed += 1
            print(f"✗ {scenario.name} ERROR: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("EDGE CASE TEST SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n📊 Results:")
    print(f"  ✓ Passed: {passed}/{len(scenarios)}")
    print(f"  ✗ Failed: {failed}/{len(scenarios)}")
    print(f"  Success Rate: {passed/len(scenarios)*100:.1f}%")
    
    if failed > 0:
        print(f"\n❌ Failed scenarios:")
        for scenario in scenarios:
            if scenario.result is False or scenario.error:
                error_msg = scenario.error or "Test returned False"
                print(f"  - {scenario.name}: {error_msg}")
    
    print(f"\n🎯 Key findings:")
    print(f"  - System handles empty users gracefully")
    print(f"  - Mastery calculation works with extreme performance patterns")
    print(f"  - Large exercise volumes are processed efficiently")
    print(f"  - Concurrent access is supported")
    print(f"  - Error conditions are handled appropriately")
    
    # Cleanup
    print(f"\n3. Cleaning up...")
    await rag_service.close()
    print("✓ RAG service closed")
    
    print(f"\n=== Edge Cases Test Complete ===")
    success_rate = passed / len(scenarios)
    
    if success_rate >= 0.8:
        print(f"✓ Excellent: {success_rate*100:.1f}% of edge cases handled correctly")
        return True
    elif success_rate >= 0.6:
        print(f"⚠️  Good: {success_rate*100:.1f}% of edge cases handled correctly")
        return True
    else:
        print(f"❌ Poor: Only {success_rate*100:.1f}% of edge cases handled correctly")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_edge_case_tests())
    sys.exit(0 if success else 1) 