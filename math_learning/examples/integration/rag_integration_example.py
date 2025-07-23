"""
RAG Integration Example for Math Learning System.

This example demonstrates how to use the RAG-enhanced knowledge graph
and learning graph together to create an intelligent, personalized
math learning experience.
"""

import asyncio
import random
from typing import Dict, List, Any

from math_learning.knowledge_graph.rag_enhanced_algebra_graph import (
    create_rag_enhanced_algebra_graph, 
    RagEnhancedAlgebraGraph
)
from math_learning.learning_graph.rag_enhanced_user_model import (
    RagEnhancedLearningGraph
)


class IntelligentMathTutor:
    """
    An intelligent math tutor that combines RAG-enhanced knowledge graphs
    and learning graphs for personalized education.
    """
    
    def __init__(self, student_id: str, student_name: str):
        """
        Initialize the intelligent math tutor.
        
        Args:
            student_id: Unique identifier for the student
            student_name: Name of the student
        """
        self.student_id = student_id
        self.student_name = student_name
        self.knowledge_graph: RagEnhancedAlgebraGraph = None
        self.learning_graph: RagEnhancedLearningGraph = None
        self.current_session_id: str = None
    
    async def initialize(self):
        """Initialize the tutor with RAG-enhanced components."""
        print(f"🚀 Initializing intelligent tutor for {self.student_name}...")
        
        # Initialize knowledge graph
        self.knowledge_graph = await create_rag_enhanced_algebra_graph()
        print("✅ Knowledge graph loaded with RAG capabilities")
        
        # Initialize learning graph
        self.learning_graph = RagEnhancedLearningGraph(
            user_id=self.student_id, 
            name=f"{self.student_name}'s Learning Journey"
        )
        print("✅ Learning graph initialized with RAG analytics")
    
    async def start_learning_session(self, learning_goals: List[str] = None) -> str:
        """
        Start a new learning session with intelligent goal setting.
        
        Args:
            learning_goals: Optional explicit learning goals
            
        Returns:
            Session ID
        """
        if not learning_goals:
            # Use RAG to suggest learning goals based on current progress
            recommendations = await self.learning_graph.get_personalized_recommendations()
            learning_goals = [
                f"Focus on {concept}" for concept in recommendations["focus_areas"][:2]
            ]
            if recommendations["next_concepts"]:
                learning_goals.append(f"Explore {recommendations['next_concepts'][0]}")
        
        self.current_session_id = await self.learning_graph.start_learning_session(learning_goals)
        
        print(f"📚 Started learning session: {self.current_session_id}")
        print(f"🎯 Goals: {', '.join(learning_goals)}")
        
        return self.current_session_id
    
    async def get_next_exercise(self, concept_id: str = None) -> Dict[str, Any]:
        """
        Get the next recommended exercise using RAG intelligence.
        
        Args:
            concept_id: Optional specific concept to practice
            
        Returns:
            Exercise recommendation with adaptive difficulty
        """
        if not concept_id:
            # Get recommendations from learning graph
            recommendations = await self.learning_graph.get_personalized_recommendations(
                self.current_session_id
            )
            
            if recommendations["focus_areas"]:
                concept_id = recommendations["focus_areas"][0]
            else:
                concept_id = "NS-01"  # Default to basic concept
        
        # Get adaptive difficulty
        difficulty = await self.learning_graph.get_adaptive_difficulty(concept_id)
        
        # Get teaching recommendations from knowledge graph
        student_profile = {
            "learning_style": "visual",  # Could be determined from past performance
            "difficulty_preference": "moderate",
            "interests": ["games", "real-world applications"]
        }
        
        teaching_rec = await self.knowledge_graph.get_teaching_recommendations(
            concept_id, student_profile
        )
        
        # Simulate exercise generation (in real system, this would be more sophisticated)
        exercise = {
            "exercise_id": f"ex_{concept_id}_{random.randint(1000, 9999)}",
            "concept_id": concept_id,
            "difficulty": difficulty,
            "type": "practice",
            "teaching_strategies": teaching_rec.get("teaching_strategies", [])[:2],
            "examples": teaching_rec.get("examples", [])[:1],
            "personalized_tips": teaching_rec.get("personalized_tips", []),
            "estimated_time": max(2, int(5 * difficulty))  # 2-5 minutes
        }
        
        return exercise
    
    async def submit_exercise_result(self, exercise: Dict[str, Any], 
                                   result: bool, time_spent: int,
                                   hints_used: int = 0, 
                                   confidence: int = None) -> Dict[str, Any]:
        """
        Submit exercise result and get intelligent feedback.
        
        Args:
            exercise: The exercise that was completed
            result: Whether the exercise was completed correctly
            time_spent: Time spent in seconds
            hints_used: Number of hints used
            confidence: Student's confidence rating (1-5)
            
        Returns:
            Feedback and next steps
        """
        # Record the attempt with enhanced tracking
        await self.learning_graph.record_enhanced_exercise_attempt(
            session_id=self.current_session_id,
            exercise_id=exercise["exercise_id"],
            concept_id=exercise["concept_id"],
            result=result,
            difficulty=exercise["difficulty"],
            time_spent=time_spent,
            hints_used=hints_used,
            student_confidence=confidence
        )
        
        # Generate intelligent feedback
        feedback = {
            "result": result,
            "concept_id": exercise["concept_id"],
            "feedback_message": "",
            "next_steps": [],
            "encouragement": "",
            "mastery_update": ""
        }
        
        # Get current mastery level
        current_mastery = self.learning_graph.traditional_graph.get_mastery(exercise["concept_id"])
        
        if result:
            feedback["feedback_message"] = "Great job! You got it right! 🎉"
            feedback["encouragement"] = "Your understanding is improving!"
            
            if current_mastery > 0.8:
                feedback["next_steps"].append("Ready for more challenging problems")
                feedback["mastery_update"] = f"Excellent mastery of {exercise['concept_id']}!"
            else:
                feedback["next_steps"].append("Continue practicing to build confidence")
        else:
            feedback["feedback_message"] = "Not quite right, but that's okay! Learning takes practice. 💪"
            feedback["encouragement"] = "Every mistake is a learning opportunity!"
            
            # Get similar concepts for additional support
            similar_concepts = await self.knowledge_graph.find_similar_concepts(
                f"help with {exercise['concept_id']}", top_k=2
            )
            
            if similar_concepts:
                feedback["next_steps"].append("Try reviewing related concepts")
                feedback["next_steps"].extend([
                    f"Review: {concept.get('concept_id', 'related concept')}" 
                    for concept in similar_concepts[:1]
                ])
        
        # Add time-based feedback
        expected_time = exercise.get("estimated_time", 3) * 60  # Convert to seconds
        if time_spent > expected_time * 1.5:
            feedback["next_steps"].append("Take your time - accuracy is more important than speed")
        elif time_spent < expected_time * 0.5 and result:
            feedback["next_steps"].append("Great speed! Try a more challenging problem")
        
        return feedback
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive learning insights using RAG analysis.
        
        Returns:
            Learning insights and recommendations
        """
        insights = await self.learning_graph.generate_learning_insights()
        
        # Get learning path suggestions
        recommendations = await self.learning_graph.get_personalized_recommendations()
        
        # Find concepts the student might enjoy based on interests
        interesting_concepts = await self.knowledge_graph.find_similar_concepts(
            "fun engaging math concepts for visual learners", top_k=3
        )
        
        return {
            "strengths": [i for i in insights if i.insight_type == "strength"],
            "areas_for_improvement": [i for i in insights if i.insight_type == "weakness"],
            "learning_patterns": [i for i in insights if i.insight_type == "pattern"],
            "recommendations": recommendations,
            "interesting_concepts": [
                concept.get("concept_id", "unknown") for concept in interesting_concepts
            ],
            "total_insights": len(insights)
        }
    
    async def end_session(self, mood_rating: int = None, 
                         difficulty_rating: int = None, 
                         notes: str = "") -> Dict[str, Any]:
        """
        End the current learning session with comprehensive summary.
        
        Args:
            mood_rating: Student's mood rating (1-5)
            difficulty_rating: Perceived difficulty rating (1-5)
            notes: Optional session notes
            
        Returns:
            Session summary with insights
        """
        if not self.current_session_id:
            raise ValueError("No active session to end")
        
        summary = await self.learning_graph.end_learning_session(
            self.current_session_id, mood_rating, difficulty_rating, notes
        )
        
        # Get additional insights
        insights = await self.get_learning_insights()
        
        # Combine summaries
        comprehensive_summary = {
            **summary,
            "learning_insights": insights,
            "session_highlights": [],
            "next_session_suggestions": []
        }
        
        # Add session highlights
        if summary["success_rate"] > 0.8:
            comprehensive_summary["session_highlights"].append("Excellent performance! 🌟")
        elif summary["success_rate"] > 0.6:
            comprehensive_summary["session_highlights"].append("Good progress made! 👍")
        else:
            comprehensive_summary["session_highlights"].append("Keep practicing - you're learning! 💪")
        
        if summary["duration_minutes"] > 30:
            comprehensive_summary["session_highlights"].append("Great dedication to learning!")
        
        # Add next session suggestions
        if insights["areas_for_improvement"]:
            comprehensive_summary["next_session_suggestions"].append(
                f"Focus on {insights['areas_for_improvement'][0].concept_id}"
            )
        
        if insights["interesting_concepts"]:
            comprehensive_summary["next_session_suggestions"].append(
                f"Explore {insights['interesting_concepts'][0]} for variety"
            )
        
        self.current_session_id = None
        return comprehensive_summary
    
    async def close(self):
        """Close all RAG services and cleanup resources."""
        if self.knowledge_graph:
            await self.knowledge_graph.close()
        if self.learning_graph:
            await self.learning_graph.close()


async def demo_intelligent_tutoring():
    """
    Demonstrate the intelligent tutoring system with RAG integration.
    """
    print("🎓 Math Learning System with RAG Integration Demo")
    print("=" * 50)
    
    # Create tutor for a student
    tutor = IntelligentMathTutor("student_alice_001", "Alice")
    
    try:
        # Initialize the system
        await tutor.initialize()
        
        # Start a learning session
        print("\n📖 Starting Learning Session")
        print("-" * 30)
        session_id = await tutor.start_learning_session([
            "Master counting and place value",
            "Build confidence with number sense"
        ])
        
        # Simulate a few exercises
        print("\n🎯 Exercise Practice")
        print("-" * 20)
        
        for i in range(3):
            # Get next exercise
            exercise = await tutor.get_next_exercise()
            print(f"\n📝 Exercise {i+1}: {exercise['concept_id']}")
            print(f"   Difficulty: {exercise['difficulty']:.2f}")
            print(f"   Tips: {exercise['personalized_tips'][:1]}")
            
            # Simulate student attempt
            result = random.choice([True, True, False])  # 67% success rate
            time_spent = random.randint(30, 180)  # 30 seconds to 3 minutes
            hints_used = random.randint(0, 2)
            confidence = random.randint(2, 5)
            
            # Submit result and get feedback
            feedback = await tutor.submit_exercise_result(
                exercise, result, time_spent, hints_used, confidence
            )
            
            print(f"   Result: {'✅ Correct' if result else '❌ Incorrect'}")
            print(f"   Feedback: {feedback['feedback_message']}")
            print(f"   Next steps: {feedback['next_steps'][:1]}")
        
        # Get learning insights
        print("\n🧠 Learning Insights")
        print("-" * 20)
        insights = await tutor.get_learning_insights()
        print(f"Strengths: {len(insights['strengths'])} identified")
        print(f"Areas for improvement: {len(insights['areas_for_improvement'])} identified")
        print(f"Recommended focus: {insights['recommendations']['focus_areas'][:2]}")
        
        # End session
        print("\n📊 Session Summary")
        print("-" * 18)
        summary = await tutor.end_session(
            mood_rating=4,
            difficulty_rating=3,
            notes="Enjoyed the visual examples!"
        )
        
        print(f"Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Highlights: {summary['session_highlights']}")
        print(f"Next session: {summary['next_session_suggestions'][:1]}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await tutor.close()


async def demo_rag_features():
    """
    Demonstrate specific RAG features in isolation.
    """
    print("\n🔍 RAG Features Demo")
    print("=" * 20)
    
    # Create knowledge graph
    knowledge_graph = await create_rag_enhanced_algebra_graph()
    
    try:
        # Demonstrate semantic search
        print("\n🔎 Semantic Search")
        similar_concepts = await knowledge_graph.find_similar_concepts(
            "basic number understanding", top_k=3
        )
        print("Similar concepts found:")
        for concept in similar_concepts:
            print(f"  - {concept.get('concept_id', 'unknown')} (score: {concept.get('relevance_score', 0):.3f})")
        
        # Demonstrate learning path suggestions
        print("\n🛤️ Learning Path Suggestions")
        path = await knowledge_graph.get_learning_path_suggestions(
            "NS-01", "understand algebraic expressions"
        )
        print(f"Suggested path: {' → '.join(path)}")
        
        # Demonstrate teaching recommendations
        print("\n👨‍🏫 Teaching Recommendations")
        student_profile = {
            "learning_style": "kinesthetic",
            "difficulty_preference": "challenging",
            "interests": ["sports", "technology"]
        }
        recommendations = await knowledge_graph.get_teaching_recommendations(
            "NS-02", student_profile
        )
        print(f"Strategies: {recommendations['teaching_strategies'][:2]}")
        print(f"Personalized tips: {recommendations['personalized_tips']}")
        
    finally:
        await knowledge_graph.close()


if __name__ == "__main__":
    async def main():
        await demo_intelligent_tutoring()
        await demo_rag_features()
    
    asyncio.run(main()) 