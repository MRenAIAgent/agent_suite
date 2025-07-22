#!/usr/bin/env python3
"""
Demonstration of Personal Learning Graph functionality.

This script shows how the personal learning system works with real learning data.
"""

import sys
import os
from datetime import datetime, timedelta
import random

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.learning_graph.user_model import LearningGraph
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem


def create_mock_knowledge_graph():
    """Create a simple mock knowledge graph for demonstration."""
    
    class MockConcept:
        def __init__(self, concept_id, name, difficulty=1, category="math"):
            self.id = concept_id
            self.name = name
            self.category = category
            self.difficulty = difficulty
            self.prerequisites = set()
            self.dependents = set()
    
    class MockKnowledgeGraph:
        def __init__(self):
            self.concepts = {
                "basic_addition": MockConcept("basic_addition", "Basic Addition", 1, "arithmetic"),
                "basic_subtraction": MockConcept("basic_subtraction", "Basic Subtraction", 1, "arithmetic"),
                "multiplication": MockConcept("multiplication", "Multiplication", 2, "arithmetic"),
                "division": MockConcept("division", "Division", 2, "arithmetic"),
                "fractions": MockConcept("fractions", "Fractions", 3, "arithmetic"),
                "linear_equations": MockConcept("linear_equations", "Linear Equations", 4, "algebra"),
                "quadratic_equations": MockConcept("quadratic_equations", "Quadratic Equations", 5, "algebra"),
            }
        
        def get_concept(self, concept_id):
            return self.concepts.get(concept_id)
        
        def get_all_concepts(self):
            return list(self.concepts.values())
        
        def get_prerequisites(self, concept_id):
            # Simple prerequisite relationships
            prereqs = {
                "multiplication": ["basic_addition"],
                "division": ["multiplication"],
                "fractions": ["division"],
                "linear_equations": ["fractions"],
                "quadratic_equations": ["linear_equations"],
            }
            return [self.concepts[pid] for pid in prereqs.get(concept_id, []) if pid in self.concepts]
        
        def get_dependent_concepts(self, concept_id):
            # Find concepts that depend on this one
            dependents = []
            for cid, concept in self.concepts.items():
                prereqs = self.get_prerequisites(cid)
                if any(p.id == concept_id for p in prereqs):
                    dependents.append(concept)
            return dependents
    
    return MockKnowledgeGraph()


def simulate_learning_session(lg, concept_id, session_num, base_success_rate=0.7):
    """Simulate a learning session for a specific concept."""
    print(f"\n📚 Learning Session {session_num} - {concept_id}")
    print("-" * 50)
    
    exercises_per_session = 5
    for i in range(exercises_per_session):
        exercise_id = f"session_{session_num}_{concept_id}_ex_{i+1}"
        
        # Simulate varying difficulty and success
        difficulty = 0.3 + (i * 0.1) + (session_num * 0.05)  # Progressive difficulty
        success_probability = base_success_rate + (i * 0.05) - (difficulty * 0.2)
        
        # Add some randomness to make it more realistic
        success = random.random() < success_probability
        
        # Record the exercise attempt
        lg.record_exercise_attempt(exercise_id, concept_id, success, difficulty, 1.0)
        
        # Show progress
        status = "✅ Correct" if success else "❌ Incorrect"
        print(f"  Exercise {i+1}: {status} (difficulty: {difficulty:.2f})")
    
    # Show updated mastery
    mastery = lg.get_mastery(concept_id)
    confidence = lg.get_confidence(concept_id)
    print(f"  📊 Updated Mastery: {mastery:.3f} (confidence: {confidence:.3f})")


def demonstrate_personal_learning():
    """Demonstrate the personal learning graph system."""
    print("🎓 Personal Learning Graph Demonstration")
    print("=" * 60)
    
    # Create systems
    kg = create_mock_knowledge_graph()
    ps = PersonalizedLearningSystem(kg)
    lg = LearningGraph("demo_student", "Demo Student")
    
    print(f"👤 Student: {lg.name} (ID: {lg.user_id})")
    print(f"📚 Available Concepts: {len(kg.get_all_concepts())}")
    
    # Simulate learning progression over multiple sessions
    concepts_to_learn = ["basic_addition", "multiplication", "fractions"]
    
    for session in range(1, 4):  # 3 learning sessions
        print(f"\n🗓️  Learning Session {session}")
        print("=" * 40)
        
        for concept_id in concepts_to_learn:
            # Improve success rate over time
            base_success = 0.4 + (session * 0.2)
            simulate_learning_session(lg, concept_id, session, base_success)
    
    # Analyze learning patterns
    print("\n🔍 Learning Analysis")
    print("=" * 40)
    
    patterns = ps.analyze_learning_patterns(lg)
    print(f"📈 Learning Patterns Found: {len(patterns)}")
    
    for concept_id, pattern in patterns.items():
        concept = kg.get_concept(concept_id)
        concept_name = concept.name if concept else concept_id
        print(f"\n📊 {concept_name}:")
        print(f"  • Attempts: {pattern.attempts}")
        print(f"  • Success Rate: {pattern.success_rate:.1%}")
        print(f"  • Learning Velocity: {pattern.learning_velocity:.3f}")
        print(f"  • Retention Rate: {pattern.retention_rate:.3f}")
        print(f"  • Difficulty Progression: {len(pattern.difficulty_progression)} data points")
    
    # Detect weaknesses
    weaknesses = ps.detect_weaknesses(lg)
    print(f"\n⚠️  Weaknesses Detected: {len(weaknesses)}")
    
    for weakness in weaknesses:
        concept = kg.get_concept(weakness.concept_id)
        concept_name = concept.name if concept else weakness.concept_id
        print(f"\n🔴 {concept_name}:")
        print(f"  • Type: {weakness.weakness_type}")
        print(f"  • Severity: {weakness.severity:.1%}")
        print(f"  • Confidence: {weakness.confidence_level:.1%}")
        print(f"  • Interventions: {', '.join(weakness.suggested_interventions[:2])}")
    
    # Get recommendations
    recommendations = ps.get_adaptive_recommendations(lg, num_recommendations=5)
    print(f"\n💡 Adaptive Recommendations: {len(recommendations)}")
    
    for i, rec in enumerate(recommendations, 1):
        concept = kg.get_concept(rec['concept_id'])
        concept_name = concept.name if concept else rec['concept_id']
        print(f"\n{i}. {concept_name}")
        print(f"   • Type: {rec['type']}")
        print(f"   • Priority: {rec['priority']:.1%}")
        print(f"   • Reason: {rec['reason']}")
    
    # Generate insights
    insights = ps.generate_learning_insights(lg)
    print(f"\n🧠 Learning Insights: {len(insights)}")
    
    for insight in insights:
        print(f"\n💭 {insight.title}")
        print(f"   {insight.description}")
        print(f"   (Confidence: {insight.confidence:.1%})")
    
    # Show mastery overview
    print(f"\n📈 Final Mastery Overview")
    print("=" * 40)
    
    for concept_id in concepts_to_learn:
        concept = kg.get_concept(concept_id)
        concept_name = concept.name if concept else concept_id
        mastery = lg.get_mastery(concept_id)
        confidence = lg.get_confidence(concept_id)
        
        mastery_level = "🟢 Mastered" if mastery >= 0.7 else "🟡 Learning" if mastery >= 0.4 else "🔴 Struggling"
        print(f"{concept_name}: {mastery:.1%} {mastery_level} (confidence: {confidence:.1%})")
    
    # Test serialization
    print(f"\n💾 Testing Data Persistence")
    print("=" * 40)
    
    # Save to file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        lg.save_to_file(temp_file)
        print(f"✅ Learning data saved to temporary file")
        
        # Load from file
        loaded_lg = LearningGraph.load_from_file(temp_file)
        print(f"✅ Learning data loaded successfully")
        print(f"   • User: {loaded_lg.name}")
        print(f"   • Concepts tracked: {len(loaded_lg.concept_mastery)}")
        print(f"   • Exercise history: {len(loaded_lg.exercise_history)} exercises")
        
    finally:
        os.unlink(temp_file)
        print(f"✅ Temporary file cleaned up")
    
    print(f"\n🎉 Demonstration Complete!")
    print("=" * 60)
    print("The personal learning graph system successfully:")
    print("✅ Tracked learning progress across multiple concepts")
    print("✅ Analyzed learning patterns and performance")
    print("✅ Detected weaknesses and areas for improvement")
    print("✅ Generated adaptive recommendations")
    print("✅ Provided learning insights")
    print("✅ Persisted data to file and loaded it back")


if __name__ == "__main__":
    demonstrate_personal_learning() 