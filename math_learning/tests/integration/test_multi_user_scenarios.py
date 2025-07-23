#!/usr/bin/env python3
"""Comprehensive multi-user test scenarios for math learning system with RAG backend."""

import asyncio
import sys
import os
from typing import Dict, List, Tuple, Any
import time
import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from math_learning.config.rag_config import get_memory_config, create_simple_rag_service
from math_learning.learning_graph.user_model import LearningGraph
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.context import ContextItem
from agents.rag.middleware.storage_router import StorageType

class UserScenario:
    """Represents a user learning scenario with specific error patterns."""
    
    def __init__(self, user_id: str, name: str, description: str, exercise_patterns: Dict[str, List[bool]]):
        self.user_id = user_id
        self.name = name
        self.description = description
        self.exercise_patterns = exercise_patterns
        self.learning_graph = None
        self.missed_concepts = []
        self.recommended_exercises = []
        self.recommended_concepts = []

async def setup_math_concepts(rag_service):
    """Set up the math concept knowledge graph."""
    print("Setting up math concept knowledge graph...")
    
    graph_adaptor = rag_service.storage_adaptors[StorageType.GRAPH]
    
    # Extended math concept hierarchy
    concepts = [
        # Foundation concepts
        {"id": "counting", "name": "Counting", "difficulty": 0.1, "category": "foundation"},
        {"id": "basic_arithmetic", "name": "Basic Arithmetic", "difficulty": 0.3, "category": "foundation"},
        {"id": "number_sense", "name": "Number Sense", "difficulty": 0.2, "category": "foundation"},
        
        # Intermediate concepts
        {"id": "integers", "name": "Integer Operations", "difficulty": 0.5, "category": "intermediate"},
        {"id": "fractions", "name": "Fractions", "difficulty": 0.7, "category": "intermediate"},
        {"id": "decimals", "name": "Decimals", "difficulty": 0.6, "category": "intermediate"},
        {"id": "percentages", "name": "Percentages", "difficulty": 0.65, "category": "intermediate"},
        
        # Advanced concepts
        {"id": "algebra", "name": "Basic Algebra", "difficulty": 0.8, "category": "advanced"},
        {"id": "geometry", "name": "Basic Geometry", "difficulty": 0.75, "category": "advanced"},
        {"id": "equations", "name": "Linear Equations", "difficulty": 0.85, "category": "advanced"},
    ]
    
    concept_entities = {}
    for concept in concepts:
        entity = Entity(
            type='math_concept',
            properties={
                'concept_id': concept['id'],
                'name': concept['name'],
                'difficulty': concept['difficulty'],
                'category': concept['category'],
                'domain': 'mathematics'
            }
        )
        entity_id = await graph_adaptor.store_entity(entity)
        concept_entities[concept['id']] = entity_id
    
    # Define prerequisite relationships
    prerequisites = [
        # Foundation prerequisites
        ("basic_arithmetic", "counting"),
        ("number_sense", "counting"),
        ("integers", "basic_arithmetic"),
        ("integers", "number_sense"),
        
        # Intermediate prerequisites
        ("fractions", "basic_arithmetic"),
        ("fractions", "integers"),
        ("decimals", "fractions"),
        ("decimals", "basic_arithmetic"),
        ("percentages", "fractions"),
        ("percentages", "decimals"),
        
        # Advanced prerequisites
        ("algebra", "integers"),
        ("algebra", "fractions"),
        ("geometry", "basic_arithmetic"),
        ("geometry", "fractions"),
        ("equations", "algebra"),
        ("equations", "integers"),
    ]
    
    for concept, prereq in prerequisites:
        if concept in concept_entities and prereq in concept_entities:
            relationship = Relationship(
                source_id=concept_entities[prereq],
                target_id=concept_entities[concept],
                type='prerequisite',
                properties={'strength': 0.9}
            )
            await graph_adaptor.store_relationship(relationship)
    
    print(f"✓ Set up {len(concepts)} concepts and {len(prerequisites)} prerequisite relationships")
    return concept_entities

def create_user_scenarios() -> List[UserScenario]:
    """Create diverse user scenarios with different learning patterns and error types."""
    
    scenarios = [
        # Scenario 1: Strong foundation, struggles with fractions
        UserScenario(
            user_id="student_001",
            name="Alex - Foundation Strong, Fraction Struggles",
            description="Strong in basic concepts but struggles specifically with fractions",
            exercise_patterns={
                "counting": [True, True, True, True, True, True, True, True],  # 100% - excellent
                "basic_arithmetic": [True, True, True, True, False, True, True, True],  # 87.5% - very good
                "number_sense": [True, True, True, True, True, True, False, True],  # 87.5% - very good
                "integers": [True, True, False, True, True, True, False, True],  # 75% - good
                "fractions": [False, False, True, False, False, False, True, False, False, False],  # 20% - struggling
                "decimals": [False, True, False, False, True, False, False, True],  # 37.5% - poor (depends on fractions)
                "percentages": [False, False, True, False, False, True, False, False],  # 25% - poor
                "algebra": [False, False, False, True, False, False, False, False],  # 12.5% - very poor
            }
        ),
        
        # Scenario 2: Weak foundation, tries to jump ahead
        UserScenario(
            user_id="student_002", 
            name="Blake - Weak Foundation, Jumping Ahead",
            description="Weak foundation but attempting advanced concepts",
            exercise_patterns={
                "counting": [True, False, True, False, True, False, True, False],  # 50% - weak
                "basic_arithmetic": [False, True, False, True, False, False, True, False, False, True],  # 40% - poor
                "number_sense": [False, False, True, False, False, True, False, False],  # 25% - very poor
                "integers": [False, False, False, True, False, False, False, True, False, False],  # 20% - very poor
                "fractions": [False, False, False, False, True, False, False, False],  # 12.5% - extremely poor
                "decimals": [False, False, False, False, False, True, False, False],  # 12.5% - extremely poor
                "algebra": [False, False, False, False, False, False, False, False, False, False],  # 0% - impossible without foundation
                "geometry": [True, False, False, False, True, False, False, False],  # 25% - some spatial intuition
                "equations": [False, False, False, False, False, False, False, False],  # 0% - impossible
            }
        ),
        
        # Scenario 3: Inconsistent performer with specific concept failures
        UserScenario(
            user_id="student_003",
            name="Casey - Inconsistent Performance", 
            description="Shows inconsistent performance with specific concept weaknesses",
            exercise_patterns={
                "counting": [True, False, True, True, False, True, False, True],  # 62.5% - inconsistent
                "basic_arithmetic": [False, True, True, False, True, False, True, True, False, True],  # 60% - inconsistent
                "number_sense": [True, False, True, False, True, True, False, False],  # 50% - moderate
                "integers": [True, False, False, True, False, True, False, False, True, False],  # 40% - poor
                "fractions": [False, True, False, True, False, False, False, True, False, False],  # 30% - poor
                "decimals": [True, False, True, False, False, False, True, False],  # 37.5% - poor
                "percentages": [False, False, True, False, True, False, False, False, True, False],  # 30% - poor
                "algebra": [False, False, False, True, False, False, False, False],  # 12.5% - very poor
                "geometry": [True, False, False, False, True, False, True, False],  # 37.5% - poor
            }
        ),
        
        # Scenario 4: Advanced learner with specific gaps
        UserScenario(
            user_id="student_004",
            name="Dana - Advanced with Specific Gaps",
            description="Advanced learner with unexpected gaps in intermediate concepts",
            exercise_patterns={
                "counting": [True, True, True, True, True, True, True, True],  # 100% - excellent
                "basic_arithmetic": [True, True, True, True, True, True, False, True, True, True],  # 90% - excellent
                "number_sense": [True, True, True, True, True, True, True, False, True, True],  # 90% - excellent
                "integers": [True, True, True, True, False, True, True, True],  # 87.5% - very good
                "fractions": [True, True, False, True, True, True, False, True, True, True],  # 80% - good
                "decimals": [False, False, True, False, False, True, False, False, True, False],  # 30% - unexpected gap
                "percentages": [False, True, False, False, True, False, True, False, False, True],  # 40% - struggling
                "algebra": [True, True, True, False, True, True, False, True, True, False],  # 70% - good despite gaps
                "geometry": [True, True, True, True, True, False, True, True, True, True],  # 90% - excellent
                "equations": [True, False, True, True, False, True, False, True],  # 62.5% - moderate
            }
        ),
        
        # Scenario 5: Beginner making steady progress
        UserScenario(
            user_id="student_005",
            name="Eli - Steady Beginner Progress",
            description="Beginner making steady, methodical progress",
            exercise_patterns={
                "counting": [False, True, True, True, True, True, False, True],  # 75% - learning well
                "basic_arithmetic": [False, False, True, True, True, False, True, True, True, False],  # 60% - improving
                "number_sense": [True, False, True, True, False, True, True, False],  # 62.5% - developing
                "integers": [False, False, False, True, True, False, True, True],  # 50% - early stage
                "fractions": [False, False, False, False, True, False, False, True],  # 25% - just starting
            }
        ),
        
        # Scenario 6: Arithmetic strong, geometry weak
        UserScenario(
            user_id="student_006",
            name="Finn - Arithmetic Strong, Geometry Weak",
            description="Excellent at computational math but struggles with spatial concepts",
            exercise_patterns={
                "counting": [True, True, True, True, True, True, True, True],  # 100% - perfect
                "basic_arithmetic": [True, True, True, True, True, False, True, True, True, True],  # 90% - excellent
                "number_sense": [True, True, True, True, False, True, True, True],  # 87.5% - very good
                "integers": [True, True, True, False, True, True, True, True, True, False],  # 80% - good
                "fractions": [True, True, False, True, True, True, True, False, True, True],  # 80% - good
                "decimals": [True, True, True, True, False, True, True, True],  # 87.5% - very good
                "percentages": [True, False, True, True, True, True, False, True, True, True],  # 80% - good
                "algebra": [True, True, False, True, True, False, True, True],  # 75% - good
                "geometry": [False, False, True, False, False, False, True, False, False, False],  # 20% - struggling
                "equations": [True, False, True, True, False, True, True, False],  # 62.5% - moderate
            }
        ),
        
        # Scenario 7: Memory issues - forgets previous concepts
        UserScenario(
            user_id="student_007",
            name="Grace - Memory Retention Issues",
            description="Learns concepts but struggles to retain them over time",
            exercise_patterns={
                "counting": [True, True, False, True, False, True, False, False],  # 50% - retention issues
                "basic_arithmetic": [True, True, True, False, False, True, False, False, True, False],  # 50% - inconsistent retention
                "number_sense": [True, False, True, False, False, True, False, True],  # 50% - moderate retention
                "integers": [True, True, False, False, True, False, False, False, True, False],  # 40% - poor retention
                "fractions": [True, False, False, False, True, False, False, False],  # 25% - very poor retention
                "decimals": [False, True, False, False, False, True, False, False],  # 25% - very poor retention
                "percentages": [False, False, True, False, False, False, True, False],  # 25% - very poor retention
            }
        ),
        
        # Scenario 8: Concept confusion - mixes up similar concepts
        UserScenario(
            user_id="student_008",
            name="Henry - Concept Confusion",
            description="Confuses similar mathematical concepts and operations",
            exercise_patterns={
                "counting": [True, True, True, False, True, True, True, False],  # 75% - basic counting ok
                "basic_arithmetic": [True, False, True, False, True, False, True, False, False, True],  # 50% - operation confusion
                "number_sense": [False, True, False, True, False, True, False, True],  # 50% - place value confusion
                "integers": [False, False, True, False, True, False, False, True, False, False],  # 30% - sign confusion
                "fractions": [False, True, False, False, False, True, False, False, True, False],  # 30% - fraction operation confusion
                "decimals": [True, False, False, True, False, False, True, False],  # 37.5% - decimal place confusion
                "percentages": [False, False, False, True, False, True, False, False],  # 25% - percent conversion confusion
                "algebra": [False, False, False, False, True, False, False, False],  # 12.5% - variable confusion
            }
        ),
        
        # Scenario 9: Test anxiety - performs worse under pressure
        UserScenario(
            user_id="student_009",
            name="Iris - Test Anxiety",
            description="Understands concepts but performs poorly due to anxiety",
            exercise_patterns={
                "counting": [True, False, True, True, False, True, True, False],  # 62.5% - anxiety affects performance
                "basic_arithmetic": [True, True, False, True, False, False, True, True, False, True],  # 60% - inconsistent due to stress
                "number_sense": [False, True, True, False, True, False, True, True],  # 62.5% - stress-related errors
                "integers": [True, False, False, True, True, False, False, True, False, True],  # 50% - anxiety impact
                "fractions": [False, True, False, False, True, False, True, False],  # 37.5% - high stress concept
                "decimals": [True, False, True, False, False, True, False, False],  # 37.5% - anxiety-prone area
                "percentages": [False, False, True, True, False, False, True, False],  # 37.5% - stress affects calculation
                "algebra": [False, True, False, False, False, True, False, False],  # 25% - high anxiety with variables
            }
        ),
        
        # Scenario 10: Overconfident - rushes through without careful thought
        UserScenario(
            user_id="student_010",
            name="Jack - Overconfident Rusher",
            description="Overconfident student who makes careless errors by rushing",
            exercise_patterns={
                "counting": [True, True, False, True, True, True, False, True],  # 75% - careless counting errors
                "basic_arithmetic": [True, True, True, False, True, False, True, True, False, True],  # 70% - calculation mistakes
                "number_sense": [True, False, True, True, True, False, True, True],  # 75% - place value errors
                "integers": [True, True, False, True, False, True, True, False, True, False],  # 60% - sign errors from rushing
                "fractions": [False, True, True, False, True, False, True, True, False, True],  # 60% - operation mistakes
                "decimals": [True, False, True, True, False, True, False, True],  # 62.5% - decimal point errors
                "percentages": [True, True, False, False, True, True, False, True],  # 62.5% - conversion errors
                "algebra": [False, True, False, True, True, False, True, False],  # 50% - variable manipulation errors
                "geometry": [True, False, True, False, True, True, False, True],  # 62.5% - measurement errors
            }
        ),
    ]
    
    return scenarios

async def analyze_missed_concepts(scenario: UserScenario, concept_entities: Dict[str, str], rag_service) -> List[str]:
    """Analyze which concepts the user is missing based on their performance."""
    missed_concepts = []
    
    # Identify concepts with low mastery (< 50%)
    for concept_id in scenario.exercise_patterns.keys():
        mastery = scenario.learning_graph.get_mastery(concept_id)
        if mastery < 0.5:
            missed_concepts.append(concept_id)
    
    # Check for prerequisite gaps
    graph_adaptor = rag_service.storage_adaptors[StorageType.GRAPH]
    
    for concept_id in missed_concepts:
        if concept_id in concept_entities:
            # Find prerequisites for this concept
            try:
                neighbors = await graph_adaptor.query_graph(
                    'get_neighbors',
                    {'entity_id': concept_entities[concept_id], 'direction': 'incoming'}
                )
                
                # Check if prerequisites are also missed
                for neighbor in neighbors:
                    # This is a simplified check - in practice you'd need to map back to concept IDs
                    pass
                    
            except Exception as e:
                print(f"Warning: Could not query prerequisites for {concept_id}: {e}")
    
    scenario.missed_concepts = missed_concepts
    return missed_concepts

def recommend_exercises(scenario: UserScenario) -> List[Dict[str, Any]]:
    """Recommend specific exercises based on missed concepts and performance patterns."""
    recommendations = []
    
    for concept_id in scenario.missed_concepts:
        mastery = scenario.learning_graph.get_mastery(concept_id)
        confidence = scenario.learning_graph.get_confidence(concept_id)
        
        # Determine exercise difficulty based on current performance
        if mastery < 0.2:
            difficulty = 0.2  # Very easy
            exercise_count = 8
        elif mastery < 0.4:
            difficulty = 0.3  # Easy
            exercise_count = 6
        else:
            difficulty = 0.4  # Moderate
            exercise_count = 4
        
        recommendations.append({
            'concept_id': concept_id,
            'difficulty': difficulty,
            'exercise_count': exercise_count,
            'focus_areas': get_focus_areas(concept_id, mastery),
            'current_mastery': mastery,
            'confidence': confidence
        })
    
    scenario.recommended_exercises = recommendations
    return recommendations

def get_focus_areas(concept_id: str, mastery: float) -> List[str]:
    """Get specific focus areas for a concept based on mastery level."""
    focus_map = {
        'counting': ['number recognition', 'sequence', 'one-to-one correspondence'],
        'basic_arithmetic': ['addition facts', 'subtraction facts', 'mental math'],
        'number_sense': ['place value', 'number comparison', 'estimation'],
        'integers': ['negative numbers', 'integer operations', 'number line'],
        'fractions': ['fraction concepts', 'equivalent fractions', 'fraction operations'],
        'decimals': ['decimal place value', 'decimal operations', 'fraction-decimal conversion'],
        'percentages': ['percent concepts', 'percent calculations', 'percent applications'],
        'algebra': ['variables', 'expressions', 'simple equations'],
        'geometry': ['shapes', 'measurements', 'spatial reasoning'],
        'equations': ['solving equations', 'graphing', 'systems of equations'],
    }
    
    areas = focus_map.get(concept_id, ['general practice'])
    
    # Return fewer focus areas for very low mastery
    if mastery < 0.3:
        return areas[:1]  # Focus on just one area
    elif mastery < 0.5:
        return areas[:2]  # Focus on two areas
    else:
        return areas  # All areas
    
def recommend_learning_concepts(scenario: UserScenario) -> List[Dict[str, Any]]:
    """Recommend which concepts to learn next based on current mastery and prerequisites."""
    recommendations = []
    
    # Concept progression map
    progression = {
        'counting': ['basic_arithmetic', 'number_sense'],
        'basic_arithmetic': ['integers', 'fractions'],
        'number_sense': ['integers', 'decimals'],
        'integers': ['fractions', 'algebra'],
        'fractions': ['decimals', 'percentages', 'algebra'],
        'decimals': ['percentages', 'algebra'],
        'percentages': ['algebra'],
        'algebra': ['equations'],
        'geometry': ['equations'],
    }
    
    # Find concepts that are ready to learn (prerequisites met)
    for concept_id, next_concepts in progression.items():
        current_mastery = scenario.learning_graph.get_mastery(concept_id)
        
        if current_mastery >= 0.6:  # Good mastery of prerequisite
            for next_concept in next_concepts:
                next_mastery = scenario.learning_graph.get_mastery(next_concept)
                
                if next_mastery < 0.5:  # Not yet mastered
                    recommendations.append({
                        'concept_id': next_concept,
                        'prerequisite': concept_id,
                        'prerequisite_mastery': current_mastery,
                        'current_mastery': next_mastery,
                        'readiness_score': current_mastery * 0.8 + (0.6 - next_mastery) * 0.2
                    })
    
    # Sort by readiness score
    recommendations.sort(key=lambda x: x['readiness_score'], reverse=True)
    
    # Remove duplicates and keep top 3
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec['concept_id'] not in seen:
            seen.add(rec['concept_id'])
            unique_recommendations.append(rec)
            if len(unique_recommendations) >= 3:
                break
    
    scenario.recommended_concepts = unique_recommendations
    return unique_recommendations

async def run_user_scenario(scenario: UserScenario, concept_entities: Dict[str, str], rag_service):
    """Run a complete learning scenario for a user."""
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"{'='*60}")
    
    # Create learning graph for this user
    scenario.learning_graph = LearningGraph(user_id=scenario.user_id, name=f"{scenario.name} Learning Graph")
    
    # Simulate exercises
    print("\n📚 Simulating learning exercises...")
    total_exercises = 0
    
    for concept_id, results in scenario.exercise_patterns.items():
        print(f"\n  Learning {concept_id}:")
        
        for i, result in enumerate(results):
            exercise_id = f"{concept_id}_exercise_{i+1}"
            scenario.learning_graph.record_exercise_attempt(
                exercise_id=exercise_id,
                concept_id=concept_id,
                result=result,
                difficulty=0.5,
                concept_weight=1.0
            )
            
            # Get final mastery levels
            mastery = scenario.learning_graph.get_mastery(concept_id)
            confidence = scenario.learning_graph.get_confidence(concept_id)
            print(f"    Exercise {i+1}: {'✓' if result else '✗'} (mastery: {mastery:.3f}, confidence: {confidence:.3f})")
            total_exercises += 1
    
    # Analyze missed concepts
    print("\n🔍 Analyzing missed concepts...")
    missed_concepts = await analyze_missed_concepts(scenario, concept_entities, rag_service)
    
    if missed_concepts:
        print(f"  Missed concepts: {', '.join(missed_concepts)}")
        for concept in missed_concepts:
            mastery = scenario.learning_graph.get_mastery(concept)
            print(f"    - {concept}: {mastery:.1%} mastery")
    else:
        print("  ✓ No significantly missed concepts!")
    
    # Recommend exercises
    print("\n💡 Recommending exercises...")
    exercise_recommendations = recommend_exercises(scenario)
    
    if exercise_recommendations:
        for rec in exercise_recommendations:
            print(f"  📝 {rec['concept_id']}:")
            print(f"    - {rec['exercise_count']} exercises at difficulty {rec['difficulty']}")
            print(f"    - Focus areas: {', '.join(rec['focus_areas'])}")
            print(f"    - Current mastery: {rec['current_mastery']:.1%}")
    else:
        print("  ✓ No additional exercises needed!")
    
    # Recommend learning concepts
    print("\n🎯 Recommending next learning concepts...")
    concept_recommendations = recommend_learning_concepts(scenario)
    
    if concept_recommendations:
        for rec in concept_recommendations:
            print(f"  📈 Ready to learn {rec['concept_id']}:")
            print(f"    - Prerequisite: {rec['prerequisite']} ({rec['prerequisite_mastery']:.1%} mastery)")
            print(f"    - Readiness score: {rec['readiness_score']:.3f}")
    else:
        print("  ⏳ Focus on current concepts before advancing!")
    
    # Store learning progress in RAG context system
    total_exercises_all = sum(len(results) for results in scenario.exercise_patterns.values())
    total_correct_all = sum(sum(results) for results in scenario.exercise_patterns.values())
    overall_success_rate_all = (total_correct_all / total_exercises_all) * 100 if total_exercises_all > 0 else 0
    
    progress_context = ContextItem(
        key=f"learning_progress_{scenario.user_id}",
        value={
            "user_id": scenario.user_id,
            "total_exercises": total_exercises_all,
            "total_correct": total_correct_all,
            "overall_success_rate": overall_success_rate_all,
            "concept_mastery": {concept: scenario.learning_graph.get_mastery(concept) for concept in scenario.exercise_patterns.keys()},
            "struggling_concepts": [concept for concept in scenario.exercise_patterns.keys() if scenario.learning_graph.get_mastery(concept) < 0.5],
            "mastered_concepts": [concept for concept in scenario.exercise_patterns.keys() if scenario.learning_graph.get_mastery(concept) >= 0.5],
            "timestamp": time.time()
        },
        metadata={
            "type": "learning_progress",
            "user_id": scenario.user_id,
            "scenario": scenario.name
        }
    )
    await rag_service.store_context(progress_context)
    
    # Summary
    print(f"\n📊 Summary for {scenario.name}:")
    print(f"  - Total exercises completed: {total_exercises}")
    print(f"  - Concepts with issues: {len(missed_concepts)}")
    print(f"  - Exercise recommendations: {len(exercise_recommendations)}")
    print(f"  - Next concept recommendations: {len(concept_recommendations)}")
    
    return scenario

@pytest.mark.asyncio
async def test_multi_user_scenarios() -> bool:
    """Test multiple user scenarios with different learning patterns."""
    print("🧮 Testing Multi-User Math Learning Scenarios with RAG Backend")
    print("=" * 70)
    
    try:
        # Create RAG service with memory backends
        rag_service = await create_simple_rag_service()
        
        print("✅ RAG service created successfully")
        
        # Create user scenarios
        scenarios = create_user_scenarios()
        print(f"📚 Created {len(scenarios)} user scenarios")
        
        # Test each scenario
        all_results = []
        for scenario in scenarios:
            try:
                result = await run_user_scenario(rag_service, scenario)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error testing {scenario.name}: {e}")
                return False
        
        # Generate comprehensive summary
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE MULTI-USER ANALYSIS")
        print(f"{'='*70}")
        
        # Overall statistics
        total_users = len(all_results)
        total_exercises = sum(result["total_exercises"] for result in all_results)
        
        print(f"Total users tested: {total_users}")
        print(f"Total exercises completed: {total_exercises}")
        print(f"Average exercises per user: {total_exercises / total_users:.1f}")
        
        # Concept analysis across all users
        all_concepts = set()
        for result in all_results:
            all_concepts.update(result["concept_results"].keys())
        
        concept_stats = {}
        for concept in all_concepts:
            masteries = []
            success_rates = []
            for result in all_results:
                if concept in result["concept_results"]:
                    masteries.append(result["concept_results"][concept]["mastery"] * 100)  # Convert to percentage
                    success_rates.append(result["concept_results"][concept]["success_rate"])
            
            if masteries:
                concept_stats[concept] = {
                    "avg_mastery": sum(masteries) / len(masteries),
                    "avg_success_rate": sum(success_rates) / len(success_rates),
                    "users_tested": len(masteries),
                    "struggling_users": len([m for m in masteries if m < 30]),
                    "proficient_users": len([m for m in masteries if m >= 60])
                }
        
        print(f"\n--- Concept Difficulty Analysis ---")
        sorted_concepts = sorted(concept_stats.items(), key=lambda x: x[1]["avg_mastery"])
        
        print("Most challenging concepts:")
        for concept, stats in sorted_concepts[:3]:
            print(f"  • {concept.replace('_', ' ').title()}: "
                  f"{stats['avg_mastery']:.1f}% avg mastery, "
                  f"{stats['struggling_users']}/{stats['users_tested']} users struggling")
        
        print("\nEasiest concepts:")
        for concept, stats in sorted_concepts[-3:]:
            print(f"  • {concept.replace('_', ' ').title()}: "
                  f"{stats['avg_mastery']:.1f}% avg mastery, "
                  f"{stats['proficient_users']}/{stats['users_tested']} users proficient")
        
        # User performance patterns
        print(f"\n--- User Performance Patterns ---")
        
        # Categorize users by overall performance
        high_performers = []
        moderate_performers = []
        struggling_learners = []
        
        for result in all_results:
            masteries = [cr["mastery"] * 100 for cr in result["concept_results"].values()]  # Convert to percentage
            avg_mastery = sum(masteries) / len(masteries) if masteries else 0
            
            if avg_mastery >= 60:
                high_performers.append((result["name"], avg_mastery))
            elif avg_mastery >= 35:
                moderate_performers.append((result["name"], avg_mastery))
            else:
                struggling_learners.append((result["name"], avg_mastery))
        
        print(f"High performers ({len(high_performers)}):")
        for name, avg in sorted(high_performers, key=lambda x: x[1], reverse=True):
            print(f"  • {name}: {avg:.1f}% average mastery")
        
        print(f"\nModerate performers ({len(moderate_performers)}):")
        for name, avg in sorted(moderate_performers, key=lambda x: x[1], reverse=True):
            print(f"  • {name}: {avg:.1f}% average mastery")
        
        print(f"\nStruggling learners ({len(struggling_learners)}):")
        for name, avg in sorted(struggling_learners, key=lambda x: x[1], reverse=True):
            print(f"  • {name}: {avg:.1f}% average mastery")
        
        # Learning pattern insights
        print(f"\n--- Learning Pattern Insights ---")
        
        # Foundation vs advanced performance
        foundation_concepts = ["counting", "basic_arithmetic", "number_sense"]
        advanced_concepts = ["algebra", "geometry", "equations"]
        
        foundation_strong = 0
        advanced_strong = 0
        
        for result in all_results:
            foundation_masteries = []
            advanced_masteries = []
            
            for concept, cr in result["concept_results"].items():
                if concept in foundation_concepts:
                    foundation_masteries.append(cr["mastery"] * 100)  # Convert to percentage
                elif concept in advanced_concepts:
                    advanced_masteries.append(cr["mastery"] * 100)  # Convert to percentage
            
            if foundation_masteries:
                avg_foundation = sum(foundation_masteries) / len(foundation_masteries)
                if avg_foundation >= 60:
                    foundation_strong += 1
            
            if advanced_masteries:
                avg_advanced = sum(advanced_masteries) / len(advanced_masteries)
                if avg_advanced >= 60:
                    advanced_strong += 1
        
        print(f"Users with strong foundation: {foundation_strong}/{total_users}")
        print(f"Users ready for advanced concepts: {advanced_strong}/{total_users}")
        
        # Common struggling areas
        struggling_concepts = {}
        for result in all_results:
            for concept, mastery in result["struggling_concepts"]:
                if concept not in struggling_concepts:
                    struggling_concepts[concept] = 0
                struggling_concepts[concept] += 1
        
        if struggling_concepts:
            print(f"\nMost common struggling areas:")
            sorted_struggling = sorted(struggling_concepts.items(), key=lambda x: x[1], reverse=True)
            for concept, count in sorted_struggling[:5]:
                print(f"  • {concept.replace('_', ' ').title()}: {count} users struggling")
        
        # Cleanup
        await rag_service.close()
        print(f"\n✅ All {total_users} user scenarios tested successfully!")
        print("🧹 Resources cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_user_scenario(rag_service, scenario) -> Dict[str, Any]:
    """Test a specific user scenario and return detailed results."""
    print(f"\n{'='*60}")
    print(f"Testing: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"{'='*60}")
    
    # Create learning graph for this user
    learning_graph = LearningGraph(user_id=scenario.user_id, name=scenario.name)
    
    # Store all math concepts in the knowledge graph
    concepts = [
        "counting", "basic_arithmetic", "number_sense", "integers", 
        "fractions", "decimals", "percentages", "algebra", "geometry", "equations"
    ]
    
    concept_entities = {}
    for concept in concepts:
        entity = Entity(
            type="math_concept",
            properties={
                "name": concept.replace("_", " ").title(),
                "description": get_concept_description(concept),
                "difficulty": get_concept_difficulty(concept),
                "category": "mathematics"
            },
            id=concept
        )
        
        # Store in RAG system using store_knowledge
        await rag_service.store_knowledge(entity)
        concept_entities[concept] = entity
    
    # Create prerequisite relationships
    prerequisites = {
        "basic_arithmetic": ["counting"],
        "number_sense": ["counting", "basic_arithmetic"],
        "integers": ["number_sense"],
        "fractions": ["basic_arithmetic", "number_sense"],
        "decimals": ["fractions"],
        "percentages": ["decimals"],
        "algebra": ["integers", "fractions"],
        "geometry": ["basic_arithmetic", "number_sense"],
        "equations": ["algebra"]
    }
    
    for concept, prereqs in prerequisites.items():
        for prereq in prereqs:
            relationship = Relationship(
                source_id=prereq,
                target_id=concept,
                type="prerequisite_for",
                properties={"strength": 1.0},
                id=f"{prereq}_prereq_{concept}"
            )
            # Store relationship in RAG system
            await rag_service.store_knowledge(relationship)
    
    # Simulate exercises for each concept
    total_exercises = 0
    concept_results = {}
    
    for concept, pattern in scenario.exercise_patterns.items():
        if concept not in concepts:
            continue
            
        print(f"\n--- Testing {concept.replace('_', ' ').title()} ---")
        
        correct_count = 0
        total_count = len(pattern)
        
        for i, is_correct in enumerate(pattern):
            exercise_id = f"{concept}_exercise_{i+1}"
            
            # Record the exercise attempt
            learning_graph.record_exercise_attempt(
                exercise_id=exercise_id,
                concept_id=concept,
                result=is_correct,
                difficulty=0.5,
                concept_weight=1.0
            )
            
            if is_correct:
                correct_count += 1
            total_exercises += 1
        
        # Get final mastery levels
        mastery = learning_graph.get_mastery(concept)
        confidence = learning_graph.get_confidence(concept)
        success_rate = (correct_count / total_count) * 100
        
        concept_results[concept] = {
            "success_rate": success_rate,
            "mastery": mastery,
            "confidence": confidence,
            "total_exercises": total_count,
            "correct_answers": correct_count,
            "status": get_mastery_status(mastery)
        }
        
        print(f"  Exercises: {correct_count}/{total_count} ({success_rate:.1f}%)")
        print(f"  Mastery: {mastery*100:.1f}% (confidence: {confidence*100:.1f}%)")
        print(f"  Status: {get_mastery_status(mastery*100)}")
    
    # Store learning progress in RAG context system
    total_exercises_all = sum(len(results) for results in scenario.exercise_patterns.values())
    total_correct_all = sum(sum(results) for results in scenario.exercise_patterns.values())
    overall_success_rate_all = (total_correct_all / total_exercises_all) * 100 if total_exercises_all > 0 else 0
    
    progress_context = ContextItem(
        key=f"learning_progress_{scenario.user_id}",
        value={
            "user_id": scenario.user_id,
            "total_exercises": total_exercises_all,
            "total_correct": total_correct_all,
            "overall_success_rate": overall_success_rate_all,
            "concept_mastery": {concept: learning_graph.get_mastery(concept) for concept in scenario.exercise_patterns.keys()},
            "struggling_concepts": [concept for concept in scenario.exercise_patterns.keys() if learning_graph.get_mastery(concept) < 0.5],
            "mastered_concepts": [concept for concept in scenario.exercise_patterns.keys() if learning_graph.get_mastery(concept) >= 0.5],
            "timestamp": time.time()
        },
        metadata={
            "type": "learning_progress",
            "user_id": scenario.user_id,
            "scenario": scenario.name
        }
    )
    await rag_service.store_context(progress_context)
    
    # Analyze learning patterns
    struggling_concepts = []
    strong_concepts = []
    moderate_concepts = []
    
    for concept, results in concept_results.items():
        mastery = results["mastery"] * 100  # Convert to percentage
        if mastery < 30:
            struggling_concepts.append((concept, mastery))
        elif mastery > 70:
            strong_concepts.append((concept, mastery))
        else:
            moderate_concepts.append((concept, mastery))
    
    # Sort by mastery level
    struggling_concepts.sort(key=lambda x: x[1])
    strong_concepts.sort(key=lambda x: x[1], reverse=True)
    moderate_concepts.sort(key=lambda x: x[1])
    
    # Generate recommendations based on learning patterns
    recommendations = generate_learning_recommendations(
        concept_results, struggling_concepts, strong_concepts, prerequisites
    )
    
    # Print analysis
    print(f"\n--- Learning Analysis for {scenario.name} ---")
    print(f"Total exercises completed: {total_exercises}")
    
    if strong_concepts:
        print(f"\nStrong concepts ({len(strong_concepts)}):")
        for concept, mastery in strong_concepts:
            print(f"  • {concept.replace('_', ' ').title()}: {mastery:.1f}%")
    
    if moderate_concepts:
        print(f"\nDeveloping concepts ({len(moderate_concepts)}):")
        for concept, mastery in moderate_concepts:
            print(f"  • {concept.replace('_', ' ').title()}: {mastery:.1f}%")
    
    if struggling_concepts:
        print(f"\nStruggling concepts ({len(struggling_concepts)}):")
        for concept, mastery in struggling_concepts:
            print(f"  • {concept.replace('_', ' ').title()}: {mastery:.1f}%")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return {
        "user_id": scenario.user_id,
        "name": scenario.name,
        "description": scenario.description,
        "total_exercises": total_exercises,
        "concept_results": concept_results,
        "struggling_concepts": struggling_concepts,
        "strong_concepts": strong_concepts,
        "moderate_concepts": moderate_concepts,
        "recommendations": recommendations
    }

def get_concept_difficulty(concept: str) -> int:
    """Get difficulty level for a math concept (1-10 scale)."""
    difficulty_map = {
        "counting": 1,
        "basic_arithmetic": 2,
        "number_sense": 3,
        "integers": 4,
        "fractions": 6,
        "decimals": 6,
        "percentages": 7,
        "algebra": 8,
        "geometry": 7,
        "equations": 9
    }
    return difficulty_map.get(concept, 5)

def get_concept_description(concept: str) -> str:
    """Get description for a math concept."""
    descriptions = {
        "counting": "Basic counting and number recognition",
        "basic_arithmetic": "Addition, subtraction, multiplication, division",
        "number_sense": "Understanding place value and number relationships",
        "integers": "Working with positive and negative whole numbers",
        "fractions": "Understanding and operating with parts of a whole",
        "decimals": "Working with decimal numbers and place value",
        "percentages": "Understanding and calculating percentages",
        "algebra": "Working with variables and algebraic expressions",
        "geometry": "Shapes, spatial relationships, and measurements",
        "equations": "Solving mathematical equations and inequalities"
    }
    return descriptions.get(concept, "Mathematical concept")

def get_mastery_status(mastery: float) -> str:
    """Get mastery status based on mastery level."""
    if mastery >= 80:
        return "Excellent"
    elif mastery >= 60:
        return "Good"
    elif mastery >= 40:
        return "Developing"
    elif mastery >= 20:
        return "Struggling"
    else:
        return "Needs Support"

def generate_learning_recommendations(
    concept_results: Dict[str, Any], 
    struggling_concepts: List[Tuple[str, float]], 
    strong_concepts: List[Tuple[str, float]],
    prerequisites: Dict[str, List[str]]
) -> List[str]:
    """Generate personalized learning recommendations."""
    recommendations = []
    
    # If there are struggling concepts, prioritize foundation
    if struggling_concepts:
        weakest_concept, weakest_mastery = struggling_concepts[0]
        
        # Check if prerequisites are strong enough
        if weakest_concept in prerequisites:
            prereqs = prerequisites[weakest_concept]
            weak_prereqs = []
            for prereq in prereqs:
                if prereq in concept_results:
                    prereq_mastery = concept_results[prereq]["mastery"] * 100  # Convert to percentage
                    if prereq_mastery < 50:
                        weak_prereqs.append((prereq, prereq_mastery))
            
            if weak_prereqs:
                weakest_prereq = min(weak_prereqs, key=lambda x: x[1])
                recommendations.append(
                    f"Focus on strengthening {weakest_prereq[0].replace('_', ' ')} "
                    f"({weakest_prereq[1]:.1f}% mastery) before advancing to "
                    f"{weakest_concept.replace('_', ' ')}"
                )
            else:
                recommendations.append(
                    f"Practice more {weakest_concept.replace('_', ' ')} exercises "
                    f"(current mastery: {weakest_mastery:.1f}%)"
                )
        else:
            recommendations.append(
                f"Focus on {weakest_concept.replace('_', ' ')} fundamentals "
                f"(current mastery: {weakest_mastery:.1f}%)"
            )
    
    # Suggest next concepts to learn based on strong areas
    if strong_concepts:
        strongest_concept, strongest_mastery = strong_concepts[0]
        
        # Find concepts that have this as a prerequisite
        next_concepts = []
        for concept, prereqs in prerequisites.items():
            if strongest_concept in prereqs and concept not in concept_results:
                next_concepts.append(concept)
        
        if next_concepts:
            recommendations.append(
                f"Ready to start learning {next_concepts[0].replace('_', ' ')} "
                f"(strong foundation in {strongest_concept.replace('_', ' ')})"
            )
    
    # Pattern-specific recommendations
    mastery_values = [results["mastery"] * 100 for results in concept_results.values()]  # Convert to percentage
    avg_mastery = sum(mastery_values) / len(mastery_values) if mastery_values else 0
    
    if avg_mastery < 30:
        recommendations.append("Consider reviewing fundamental concepts before advancing")
    elif avg_mastery > 70:
        recommendations.append("Excellent progress! Ready for more challenging material")
    
    # Check for inconsistent performance
    if len(mastery_values) > 1:
        mastery_range = max(mastery_values) - min(mastery_values)
        if mastery_range > 40:
            recommendations.append("Focus on consistency across all concepts")
    
    # If no specific recommendations, provide general guidance
    if not recommendations:
        recommendations.append("Continue practicing current concepts to build confidence")
    
    return recommendations

if __name__ == "__main__":
    success = asyncio.run(test_multi_user_scenarios())
    sys.exit(0 if success else 1) 