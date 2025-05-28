"""
RAG-Enhanced User Model for Personalized Math Learning.

This module extends the traditional learning graph with RAG capabilities
for intelligent learning recommendations, progress tracking, and personalized content delivery.
"""

import asyncio
import json
import datetime
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field

from agents.rag.api.rag_service import RagService
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.document import Document
from agents.rag.models.context import ContextItem

from .user_model import LearningGraph


@dataclass
class LearningSession:
    """Represents a learning session with RAG-enhanced tracking."""
    session_id: str
    user_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    concepts_practiced: List[str] = field(default_factory=list)
    exercises_completed: List[Dict[str, Any]] = field(default_factory=list)
    learning_goals: List[str] = field(default_factory=list)
    session_notes: str = ""
    mood_rating: Optional[int] = None  # 1-5 scale
    difficulty_rating: Optional[int] = None  # 1-5 scale


@dataclass
class LearningInsight:
    """AI-generated learning insights from RAG analysis."""
    insight_type: str  # "strength", "weakness", "recommendation", "pattern"
    concept_id: str
    description: str
    confidence_score: float
    supporting_evidence: List[str] = field(default_factory=list)
    actionable_steps: List[str] = field(default_factory=list)


class RagEnhancedLearningGraph:
    """RAG-enhanced learning graph with intelligent personalization and analytics."""
    
    def __init__(self, user_id: str = "", name: str = ""):
        """
        Initialize a RAG-enhanced learning graph for a user.
        
        Args:
            user_id: Identifier for the user
            name: Optional name for the learning graph
        """
        self.traditional_graph = LearningGraph(user_id, name)
        self.rag_service = RagService()
        self.user_id = user_id
        self.name = name
        self.learning_sessions: Dict[str, LearningSession] = {}
        self.learning_insights: List[LearningInsight] = []
        self._setup_storage()
    
    def _setup_storage(self):
        """Setup RAG storage adaptors."""
        # Use memory storage by default for testing and development
        print("Using memory storage for RAG learning graph")
        # No additional storage setup needed - RagService uses memory by default
    
    async def start_learning_session(self, learning_goals: List[str] = None) -> str:
        """
        Start a new learning session with RAG-enhanced tracking.
        
        Args:
            learning_goals: Optional list of learning goals for this session
            
        Returns:
            Session ID
        """
        session_id = f"session_{self.user_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = LearningSession(
            session_id=session_id,
            user_id=self.user_id,
            start_time=datetime.datetime.now(),
            learning_goals=learning_goals or []
        )
        
        self.learning_sessions[session_id] = session
        
        # Store session context in RAG
        context_item = ContextItem(
            key=f"session_{session_id}",
            value={
                "session_id": session_id,
                "user_id": self.user_id,
                "start_time": session.start_time.isoformat(),
                "learning_goals": session.learning_goals,
                "type": "learning_session"
            },
            metadata={
                "user_id": self.user_id,
                "session_type": "math_learning",
                "status": "active"
            }
        )
        
        await self.rag_service.store_context(context_item)
        
        return session_id
    
    async def record_enhanced_exercise_attempt(self, session_id: str, exercise_id: str, 
                                             concept_id: str, result: bool,
                                             difficulty: float = 0.5, 
                                             concept_weight: float = 1.0,
                                             time_spent: Optional[int] = None,
                                             hints_used: int = 0,
                                             student_confidence: Optional[int] = None) -> None:
        """
        Record an enhanced exercise attempt with additional RAG-tracked metrics.
        
        Args:
            session_id: Current learning session ID
            exercise_id: The ID of the exercise
            concept_id: The ID of the concept being tested
            result: Whether the exercise was completed correctly
            difficulty: The difficulty of the exercise (0.0-1.0)
            concept_weight: How strongly the exercise tests the concept (0.0-1.0)
            time_spent: Time spent on exercise in seconds
            hints_used: Number of hints the student used
            student_confidence: Student's self-reported confidence (1-5)
        """
        # Record in traditional graph
        self.traditional_graph.record_exercise_attempt(
            exercise_id, concept_id, result, difficulty, concept_weight
        )
        
        # Enhanced tracking
        exercise_data = {
            "exercise_id": exercise_id,
            "concept_id": concept_id,
            "result": result,
            "difficulty": difficulty,
            "concept_weight": concept_weight,
            "time_spent": time_spent,
            "hints_used": hints_used,
            "student_confidence": student_confidence,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to session
        if session_id in self.learning_sessions:
            session = self.learning_sessions[session_id]
            session.exercises_completed.append(exercise_data)
            if concept_id not in session.concepts_practiced:
                session.concepts_practiced.append(concept_id)
        
        # Store detailed exercise data in RAG
        document_content = f"""
        Exercise Attempt Analysis
        User: {self.user_id}
        Session: {session_id}
        Exercise: {exercise_id}
        Concept: {concept_id}
        Result: {'Correct' if result else 'Incorrect'}
        Difficulty: {difficulty}/1.0
        Time Spent: {time_spent or 'Not recorded'} seconds
        Hints Used: {hints_used}
        Student Confidence: {student_confidence or 'Not reported'}/5
        
        Performance Context:
        - Current mastery level: {self.traditional_graph.get_mastery(concept_id):.2f}
        - Previous attempts on this concept: {len([h for h in self.traditional_graph.exercise_history.values() for attempt in h if attempt['concept_id'] == concept_id])}
        """
        
        document = Document(
            content=document_content,
            metadata={
                "user_id": self.user_id,
                "session_id": session_id,
                "exercise_id": exercise_id,
                "concept_id": concept_id,
                "result": result,
                "type": "exercise_attempt",
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        
        await self.rag_service.store_document(document)
    
    async def get_personalized_recommendations(self, session_id: str = None) -> Dict[str, Any]:
        """
        Get AI-powered personalized learning recommendations using RAG.
        
        Args:
            session_id: Optional current session ID for context
            
        Returns:
            Personalized recommendations including next concepts, exercises, and strategies
        """
        # Build query based on current learning state
        struggling_concepts = self.traditional_graph.get_struggling_concepts()
        mastered_concepts = self.traditional_graph.get_mastered_concepts()
        
        query = f"""
        personalized learning recommendations for user {self.user_id}
        struggling with: {', '.join(struggling_concepts[:3])}
        mastered: {', '.join(mastered_concepts[-3:])}
        learning style preferences and optimal next steps
        """
        
        # Retrieve relevant recommendations
        results = await self.rag_service.retrieve(
            query=query,
            top_k=10,
            filter_criteria={"user_id": self.user_id, "type": "exercise_attempt"}
        )
        
        recommendations = {
            "next_concepts": [],
            "recommended_exercises": [],
            "learning_strategies": [],
            "focus_areas": struggling_concepts[:3],
            "confidence_boosters": mastered_concepts[-2:],
            "estimated_study_time": 0
        }
        
        # Analyze patterns from RAG results
        concept_performance = {}
        for result in results:
            if hasattr(result, 'metadata') and 'concept_id' in result.metadata:
                concept_id = result.metadata['concept_id']
                if concept_id not in concept_performance:
                    concept_performance[concept_id] = []
                concept_performance[concept_id].append(result.metadata.get('result', False))
        
        # Generate recommendations based on analysis
        for concept_id, results_list in concept_performance.items():
            success_rate = sum(results_list) / len(results_list) if results_list else 0
            
            if success_rate < 0.6 and concept_id not in recommendations["focus_areas"]:
                recommendations["focus_areas"].append(concept_id)
            elif success_rate > 0.8 and len(recommendations["next_concepts"]) < 3:
                # Look for concepts that build on this mastered concept
                recommendations["next_concepts"].append(f"advanced_{concept_id}")
        
        # Add learning strategies based on performance patterns
        if struggling_concepts:
            recommendations["learning_strategies"].extend([
                "Break down complex problems into smaller steps",
                "Use visual aids and concrete examples",
                "Practice with easier problems first to build confidence"
            ])
        
        # Estimate study time based on concept difficulty and current mastery
        total_time = 0
        for concept_id in recommendations["focus_areas"]:
            current_mastery = self.traditional_graph.get_mastery(concept_id)
            estimated_time = max(15, int(30 * (1 - current_mastery)))  # 15-30 minutes per concept
            total_time += estimated_time
        
        recommendations["estimated_study_time"] = total_time
        
        return recommendations
    
    async def generate_learning_insights(self) -> List[LearningInsight]:
        """
        Generate AI-powered learning insights using RAG analysis.
        
        Returns:
            List of learning insights with actionable recommendations
        """
        insights = []
        
        # Query for patterns in learning data
        query = f"""
        learning patterns and insights for user {self.user_id}
        performance trends, strengths, weaknesses, and improvement opportunities
        """
        
        results = await self.rag_service.retrieve(
            query=query,
            top_k=20,
            filter_criteria={"user_id": self.user_id}
        )
        
        # Analyze concept mastery patterns
        concept_mastery = self.traditional_graph.concept_mastery
        
        for concept_id, mastery_data in concept_mastery.items():
            mastery_level = mastery_data["mastery"]
            confidence = mastery_data["confidence"]
            
            if mastery_level > 0.8 and confidence > 0.8:
                # Strength identified
                insight = LearningInsight(
                    insight_type="strength",
                    concept_id=concept_id,
                    description=f"Strong mastery of {concept_id} with high confidence",
                    confidence_score=min(mastery_level, confidence),
                    supporting_evidence=[
                        f"Mastery level: {mastery_level:.2f}",
                        f"Confidence: {confidence:.2f}"
                    ],
                    actionable_steps=[
                        f"Use {concept_id} as a foundation for learning related advanced concepts",
                        "Consider peer tutoring opportunities in this area"
                    ]
                )
                insights.append(insight)
            
            elif mastery_level < 0.4:
                # Weakness identified
                insight = LearningInsight(
                    insight_type="weakness",
                    concept_id=concept_id,
                    description=f"Struggling with {concept_id} - needs focused attention",
                    confidence_score=1.0 - mastery_level,
                    supporting_evidence=[
                        f"Low mastery level: {mastery_level:.2f}",
                        f"Multiple incorrect attempts detected"
                    ],
                    actionable_steps=[
                        f"Schedule extra practice sessions for {concept_id}",
                        "Review prerequisite concepts",
                        "Try alternative learning approaches (visual, hands-on, etc.)"
                    ]
                )
                insights.append(insight)
        
        # Analyze time-based patterns
        recent_sessions = [s for s in self.learning_sessions.values() 
                          if s.start_time > datetime.datetime.now() - datetime.timedelta(days=7)]
        
        if recent_sessions:
            avg_concepts_per_session = sum(len(s.concepts_practiced) for s in recent_sessions) / len(recent_sessions)
            
            if avg_concepts_per_session > 5:
                insight = LearningInsight(
                    insight_type="pattern",
                    concept_id="session_management",
                    description="Learning sessions may be too broad - consider focusing on fewer concepts",
                    confidence_score=0.7,
                    supporting_evidence=[
                        f"Average {avg_concepts_per_session:.1f} concepts per session",
                        "Research suggests 2-3 concepts per session for optimal retention"
                    ],
                    actionable_steps=[
                        "Limit each session to 2-3 related concepts",
                        "Spend more time on practice and reinforcement",
                        "Take breaks between different concept areas"
                    ]
                )
                insights.append(insight)
        
        self.learning_insights = insights
        return insights
    
    async def get_adaptive_difficulty(self, concept_id: str) -> float:
        """
        Get adaptive difficulty recommendation for a concept using RAG analysis.
        
        Args:
            concept_id: The concept to get difficulty for
            
        Returns:
            Recommended difficulty level (0.0-1.0)
        """
        current_mastery = self.traditional_graph.get_mastery(concept_id)
        confidence = self.traditional_graph.get_confidence(concept_id)
        
        # Query for similar learning patterns
        query = f"""
        optimal difficulty progression for concept {concept_id}
        current mastery {current_mastery:.2f} confidence {confidence:.2f}
        adaptive learning difficulty recommendations
        """
        
        results = await self.rag_service.retrieve(
            query=query,
            top_k=5,
            filter_criteria={"concept_id": concept_id}
        )
        
        # Base difficulty on current mastery and confidence
        base_difficulty = current_mastery * 0.7 + confidence * 0.3
        
        # Adjust based on recent performance
        recent_attempts = []
        for exercise_history in self.traditional_graph.exercise_history.values():
            for attempt in exercise_history:
                if (attempt['concept_id'] == concept_id and 
                    datetime.datetime.fromisoformat(attempt['timestamp']) > 
                    datetime.datetime.now() - datetime.timedelta(days=3)):
                    recent_attempts.append(attempt['result'])
        
        if recent_attempts:
            recent_success_rate = sum(recent_attempts) / len(recent_attempts)
            if recent_success_rate > 0.8:
                # Increase difficulty if doing well
                base_difficulty = min(1.0, base_difficulty + 0.1)
            elif recent_success_rate < 0.5:
                # Decrease difficulty if struggling
                base_difficulty = max(0.1, base_difficulty - 0.15)
        
        return base_difficulty
    
    async def end_learning_session(self, session_id: str, mood_rating: int = None, 
                                 difficulty_rating: int = None, notes: str = "") -> Dict[str, Any]:
        """
        End a learning session and generate session summary.
        
        Args:
            session_id: The session to end
            mood_rating: Student's mood rating (1-5)
            difficulty_rating: Student's difficulty rating (1-5)
            notes: Optional session notes
            
        Returns:
            Session summary with insights and recommendations
        """
        if session_id not in self.learning_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.learning_sessions[session_id]
        session.end_time = datetime.datetime.now()
        session.mood_rating = mood_rating
        session.difficulty_rating = difficulty_rating
        session.session_notes = notes
        
        # Calculate session metrics
        session_duration = (session.end_time - session.start_time).total_seconds() / 60  # minutes
        concepts_practiced = len(session.concepts_practiced)
        exercises_completed = len(session.exercises_completed)
        success_rate = sum(1 for ex in session.exercises_completed if ex['result']) / max(1, exercises_completed)
        
        # Store session summary in RAG
        summary_content = f"""
        Learning Session Summary
        User: {self.user_id}
        Session: {session_id}
        Duration: {session_duration:.1f} minutes
        Concepts Practiced: {concepts_practiced}
        Exercises Completed: {exercises_completed}
        Success Rate: {success_rate:.2f}
        Mood Rating: {mood_rating or 'Not provided'}/5
        Difficulty Rating: {difficulty_rating or 'Not provided'}/5
        
        Session Notes: {notes}
        
        Concepts Covered: {', '.join(session.concepts_practiced)}
        """
        
        document = Document(
            content=summary_content,
            metadata={
                "user_id": self.user_id,
                "session_id": session_id,
                "type": "session_summary",
                "duration_minutes": session_duration,
                "success_rate": success_rate,
                "mood_rating": mood_rating,
                "difficulty_rating": difficulty_rating
            }
        )
        
        await self.rag_service.store_document(document)
        
        # Generate session insights
        insights = await self.generate_learning_insights()
        
        return {
            "session_id": session_id,
            "duration_minutes": session_duration,
            "concepts_practiced": concepts_practiced,
            "exercises_completed": exercises_completed,
            "success_rate": success_rate,
            "mood_rating": mood_rating,
            "difficulty_rating": difficulty_rating,
            "insights": [
                {
                    "type": insight.insight_type,
                    "description": insight.description,
                    "actionable_steps": insight.actionable_steps
                }
                for insight in insights[-3:]  # Last 3 insights
            ]
        }
    
    async def close(self):
        """Close the RAG service and cleanup resources."""
        await self.rag_service.close()


# Example usage
async def main():
    """Example usage of the RAG-enhanced learning graph."""
    # Create enhanced learning graph
    learning_graph = RagEnhancedLearningGraph(user_id="student_123", name="Alice's Learning Journey")
    
    try:
        # Start a learning session
        session_id = await learning_graph.start_learning_session(
            learning_goals=["Master basic algebra", "Improve problem-solving speed"]
        )
        print(f"Started session: {session_id}")
        
        # Record some exercise attempts
        await learning_graph.record_enhanced_exercise_attempt(
            session_id=session_id,
            exercise_id="ex_001",
            concept_id="NS-01",
            result=True,
            difficulty=0.3,
            time_spent=45,
            hints_used=1,
            student_confidence=4
        )
        
        await learning_graph.record_enhanced_exercise_attempt(
            session_id=session_id,
            exercise_id="ex_002",
            concept_id="NS-02",
            result=False,
            difficulty=0.6,
            time_spent=120,
            hints_used=3,
            student_confidence=2
        )
        
        # Get personalized recommendations
        print("\n=== Personalized Recommendations ===")
        recommendations = await learning_graph.get_personalized_recommendations(session_id)
        print(f"Focus areas: {recommendations['focus_areas']}")
        print(f"Estimated study time: {recommendations['estimated_study_time']} minutes")
        print(f"Learning strategies: {recommendations['learning_strategies'][:2]}")
        
        # Get adaptive difficulty
        print("\n=== Adaptive Difficulty ===")
        difficulty = await learning_graph.get_adaptive_difficulty("NS-02")
        print(f"Recommended difficulty for NS-02: {difficulty:.2f}")
        
        # End session with feedback
        print("\n=== Session Summary ===")
        summary = await learning_graph.end_learning_session(
            session_id=session_id,
            mood_rating=4,
            difficulty_rating=3,
            notes="Found place value challenging but counting was easy"
        )
        print(f"Session duration: {summary['duration_minutes']:.1f} minutes")
        print(f"Success rate: {summary['success_rate']:.2f}")
        print(f"Key insights: {len(summary['insights'])} generated")
        
    finally:
        await learning_graph.close()


if __name__ == "__main__":
    asyncio.run(main()) 