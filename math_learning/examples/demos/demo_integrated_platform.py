"""
Comprehensive Demo of Integrated Math Learning Platform.

This demo showcases the complete integration of:
- Advanced Exercise System (4-level selection)
- AI Tutoring Agent
- Knowledge Graph
- Learning Graph
- Platform Analytics
"""

import asyncio
import sys
import os
from typing import Dict, List, Any

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.integration import IntegratedMathLearningPlatform
from math_learning.knowledge_graph.algebra_graph import build_algebra_knowledge_graph
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.exercises.algebra_exercises import create_elementary_exercises
from math_learning.exercises.middle_algebra_exercises import create_middle_school_exercises


class MockLLM:
    """Mock LLM for demonstration purposes."""
    
    def chat_completion(self, model: str, messages: List[Dict[str, str]]) -> str:
        """Mock chat completion."""
        return "This is a helpful tutoring response based on your question."


async def demo_integrated_platform():
    """Demonstrate the integrated math learning platform."""
    
    print("🚀 INTEGRATED MATH LEARNING PLATFORM DEMO")
    print("=" * 60)
    
    # Initialize platform with mock LLM
    print("\n📚 Initializing Platform...")
    mock_llm = MockLLM()
    platform = IntegratedMathLearningPlatform(llm=mock_llm)
    
    # Load knowledge graph and exercises
    print("📊 Loading Knowledge Graph...")
    knowledge_graph = build_algebra_knowledge_graph()
    platform.knowledge_graph = knowledge_graph
    
    print("📝 Loading Exercise Bank...")
    elementary_bank = create_elementary_exercises()
    middle_bank = create_middle_school_exercises()
    
    # Combine exercise banks
    for exercise in elementary_bank.get_all_exercises():
        platform.exercise_bank.add_exercise(exercise)
    for exercise in middle_bank.get_all_exercises():
        platform.exercise_bank.add_exercise(exercise)
    
    print(f"✅ Loaded {len(platform.exercise_bank.exercises)} exercises")
    print(f"✅ Loaded {len(platform.knowledge_graph.get_all_concepts())} concepts")
    
    # Register students
    print("\n👥 Registering Students...")
    students = [
        ("alice_2024", "Alice Johnson"),
        ("bob_2024", "Bob Smith"),
        ("carol_2024", "Carol Davis")
    ]
    
    for student_id, student_name in students:
        result = platform.register_student(student_id, student_name)
        if result["success"]:
            print(f"✅ Registered: {student_name} ({student_id})")
        else:
            print(f"❌ Failed to register: {student_name}")
    
    # Demo 1: Level 1 Exercise Selection (Basic Concept + Difficulty)
    print("\n" + "="*60)
    print("🎯 DEMO 1: Level 1 Exercise Selection")
    print("="*60)
    
    alice_exercises = platform.get_personalized_exercises(
        student_id="alice_2024",
        concept="addition",
        difficulty=2.0,
        count=3,
        session_type="practice"
    )
    
    print(f"\n📋 Alice's Level 1 Exercises (addition, difficulty 2.0):")
    if alice_exercises["success"]:
        for i, ex in enumerate(alice_exercises["exercises"], 1):
            print(f"  {i}. {ex['title']} (Difficulty: {ex['difficulty']})")
        print(f"   Generation Method: {alice_exercises['generation_method']}")
        print(f"   Total Exercises: {alice_exercises['total_exercises']}")
    else:
        print(f"   Error: {alice_exercises['error']}")
    
    # Demo 2: Simulate some learning progress
    print("\n" + "="*60)
    print("🎓 DEMO 2: Learning Progress Simulation")
    print("="*60)
    
    # Simulate Alice completing some exercises
    alice_progress = []
    if alice_exercises["success"]:
        for exercise in alice_exercises["exercises"][:2]:
            result = platform.record_exercise_completion(
                student_id="alice_2024",
                exercise_id=exercise["id"],
                success=True,
                time_spent=300  # 5 minutes
            )
            alice_progress.append(result)
            if result["success"]:
                print(f"✅ Alice completed: {exercise['title']}")
                for concept_id, mastery in result["mastery_updates"].items():
                    print(f"   {concept_id}: {mastery:.1%} mastery")
    
    # Demo 3: Level 2 Adaptive Exercise Selection
    print("\n" + "="*60)
    print("🧠 DEMO 3: Level 2 Adaptive Exercise Selection")
    print("="*60)
    
    adaptive_exercises = platform.get_personalized_exercises(
        student_id="alice_2024",
        concept="adaptive",
        count=4,
        session_type="practice"
    )
    
    print(f"\n📋 Alice's Adaptive Exercises:")
    if adaptive_exercises["success"]:
        for i, ex in enumerate(adaptive_exercises["exercises"], 1):
            print(f"  {i}. {ex['title']}")
            if "selection_reason" in ex:
                print(f"     Reason: {ex['selection_reason']}")
        print(f"   Generation Method: {adaptive_exercises['generation_method']}")
    else:
        print(f"   Error: {adaptive_exercises['error']}")
    
    # Demo 4: AI Tutoring Session
    print("\n" + "="*60)
    print("🤖 DEMO 4: AI Tutoring Session")
    print("="*60)
    
    # Start tutoring session
    session_context = {
        "current_topic": "algebra",
        "goal": "understand_variables",
        "session_type": "tutoring"
    }
    
    session_result = await platform.start_tutoring_session("alice_2024", session_context)
    if session_result["success"]:
        print("✅ AI Tutoring Session Started")
        print(f"   Context: {session_context}")
        
        # Get tutoring response
        student_input = "I'm confused about what variables represent in algebra"
        tutor_response = await platform.get_ai_tutoring_response("alice_2024", student_input)
        
        if tutor_response["success"]:
            print(f"\n💬 Student: {student_input}")
            print(f"🎓 AI Tutor: {tutor_response['tutor_response']}")
        else:
            print(f"❌ Tutoring Error: {tutor_response['error']}")
    else:
        print(f"❌ Session Error: {session_result['error']}")
    
    # Demo 5: Student Analytics
    print("\n" + "="*60)
    print("📊 DEMO 5: Student Analytics")
    print("="*60)
    
    alice_analytics = platform.get_student_analytics("alice_2024")
    if alice_analytics["success"]:
        analytics = alice_analytics["analytics"]
        print(f"\n📈 Alice's Learning Analytics:")
        print(f"   Total Exercises: {analytics['total_exercises_attempted']}")
        print(f"   Success Rate: {analytics['success_rate']:.1%}")
        print(f"   Average Mastery: {analytics['average_mastery']:.1%}")
        print(f"   Concepts Tracked: {analytics['concepts_tracked']}")
        
        print(f"\n🎯 Mastery Levels:")
        for concept_id, mastery in analytics['mastery_levels'].items():
            print(f"   {concept_id}: {mastery:.1%}")
        
        print(f"\n🔍 Knowledge Gaps: {len(analytics['knowledge_gaps'])}")
        for gap in analytics['knowledge_gaps'][:3]:
            if isinstance(gap, dict):
                print(f"   {gap.get('concept_id', 'Unknown')}: {gap.get('mastery', 0):.1%} mastery")
    else:
        print(f"❌ Analytics Error: {alice_analytics['error']}")
    
    # Demo 6: Multi-Student Comparison
    print("\n" + "="*60)
    print("👥 DEMO 6: Multi-Student Platform Analytics")
    print("="*60)
    
    # Register some progress for other students
    for student_id in ["bob_2024", "carol_2024"]:
        exercises = platform.get_personalized_exercises(
            student_id=student_id,
            concept="counting",
            difficulty=1.5,
            count=2
        )
        
        if exercises["success"] and exercises["exercises"]:
            platform.record_exercise_completion(
                student_id=student_id,
                exercise_id=exercises["exercises"][0]["id"],
                success=True
            )
    
    # Get platform analytics
    platform_analytics = platform.get_platform_analytics()
    if platform_analytics["success"]:
        analytics = platform_analytics["platform_analytics"]
        print(f"\n🏫 Platform Analytics:")
        print(f"   Total Students: {analytics['total_students']}")
        print(f"   Total Sessions: {analytics['total_sessions']}")
        print(f"   Exercises Completed: {analytics['total_exercises_completed']}")
        print(f"   Exercise Bank Size: {platform_analytics['exercise_bank_size']}")
        print(f"   Knowledge Graph Concepts: {platform_analytics['knowledge_graph_concepts']}")
        print(f"   Active Students: {platform_analytics['active_students']}")
        print(f"   AI Tutors Active: {platform_analytics['ai_tutors_active']}")
    
    # Demo 7: Different Session Types
    print("\n" + "="*60)
    print("🎮 DEMO 7: Different Session Types")
    print("="*60)
    
    session_types = ["practice", "assessment", "review", "challenge"]
    
    for session_type in session_types:
        exercises = platform.get_personalized_exercises(
            student_id="alice_2024",
            concept="adaptive",
            count=2,
            session_type=session_type
        )
        
        print(f"\n📚 {session_type.title()} Session:")
        if exercises["success"]:
            print(f"   Method: {exercises['generation_method']}")
            print(f"   Exercises: {exercises['total_exercises']}")
            for ex in exercises["exercises"]:
                print(f"   - {ex['title']}")
        else:
            print(f"   Error: {exercises['error']}")
    
    print("\n" + "="*60)
    print("🎉 INTEGRATED PLATFORM DEMO COMPLETE!")
    print("="*60)
    
    print("\n✨ Summary of Demonstrated Features:")
    print("   ✅ 4-Level Exercise Selection System")
    print("   ✅ AI-Powered Tutoring Integration")
    print("   ✅ Adaptive Learning Path Generation")
    print("   ✅ Real-time Progress Tracking")
    print("   ✅ Comprehensive Analytics")
    print("   ✅ Multi-Student Platform Management")
    print("   ✅ Session Type Customization")
    print("   ✅ Knowledge Gap Analysis")
    
    return platform


if __name__ == "__main__":
    # Run the demo
    platform = asyncio.run(demo_integrated_platform()) 