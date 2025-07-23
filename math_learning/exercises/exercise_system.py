"""
Exercise System module.

This module provides a comprehensive exercise system that integrates:
1. Static exercise lists with difficulty scoring
2. Concept-based exercise selection
3. AI-powered learning path optimization
4. Adaptive difficulty adjustment
"""

import json
import random
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime

from .exercise import Exercise
from .exercise_bank import ExerciseBank
from .exercise_selector import ExerciseSelector, DifficultyLevel
from ..knowledge_graph.graph import KnowledgeGraph
from ..learning_graph.user_model import LearningGraph

# Import LLM if available
try:
    from llm.llm import LLMBase
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMBase = None


@dataclass
class ExerciseRecommendation:
    """Comprehensive exercise recommendation with metadata."""
    exercise: Exercise
    concept_id: str
    difficulty_score: float
    confidence_score: float
    reason: str
    learning_objective: str
    estimated_success_rate: float
    priority: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['exercise'] = self.exercise.to_dict()
        return result


@dataclass
class LearningSession:
    """Represents a learning session with selected exercises."""
    session_id: str
    student_id: str
    target_concepts: List[str]
    exercises: List[ExerciseRecommendation]
    session_type: str  # "practice", "assessment", "review", "challenge"
    estimated_duration: int  # minutes
    learning_objectives: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['exercises'] = [ex.to_dict() for ex in self.exercises]
        result['created_at'] = self.created_at.isoformat()
        return result


class ExerciseSystem:
    """
    Comprehensive exercise system that provides:
    1. Basic concept + difficulty exercise selection
    2. Adaptive exercise selection based on learning progress
    3. AI-powered learning path optimization
    4. Session planning and management
    """
    
    def __init__(
        self, 
        exercise_bank: ExerciseBank,
        knowledge_graph: KnowledgeGraph,
        llm: Optional[LLMBase] = None
    ):
        """
        Initialize the exercise system.
        
        Args:
            exercise_bank: Exercise bank with all available exercises
            knowledge_graph: Knowledge graph for prerequisite analysis
            llm: Optional language model for AI-powered recommendations
        """
        self.exercise_bank = exercise_bank
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.selector = ExerciseSelector(exercise_bank, knowledge_graph)
        
        # Session management
        self.active_sessions: Dict[str, LearningSession] = {}
        
    # Level 1: Basic concept + difficulty selection
    def get_exercises_by_concept_difficulty(
        self, 
        concept_id: str, 
        difficulty_score: float, 
        count: int = 5
    ) -> List[Exercise]:
        """
        Level 1: Get exercises for a specific concept and difficulty.
        
        Args:
            concept_id: Target concept ID
            difficulty_score: Target difficulty (1.0-5.0)
            count: Number of exercises to return
            
        Returns:
            List of exercises matching criteria
        """
        return self.selector.select_exercises_by_concept_and_difficulty(
            concept_id, difficulty_score, count
        )
    
    # Level 2: Adaptive selection with learning graph
    def get_adaptive_exercises(
        self,
        learning_graph: LearningGraph,
        target_concepts: Optional[List[str]] = None,
        count: int = 5,
        difficulty_adjustment: float = 0.0
    ) -> List[ExerciseRecommendation]:
        """
        Level 2: Get adaptively selected exercises based on learning progress.
        
        Args:
            learning_graph: Student's learning progress
            target_concepts: Optional specific concepts to focus on
            count: Number of exercises to return
            difficulty_adjustment: Manual difficulty adjustment (-1.0 to 1.0)
            
        Returns:
            List of exercise recommendations with metadata
        """
        # Get basic adaptive selection
        adaptive_exercises = self.selector.select_adaptive_exercises(
            learning_graph, target_concepts, count, difficulty_adjustment
        )
        
        # Convert to recommendations with enhanced metadata
        recommendations = []
        for i, (exercise, reason) in enumerate(adaptive_exercises):
            # Find the primary concept for this exercise
            primary_concepts = exercise.get_primary_concepts()
            concept_id = primary_concepts[0] if primary_concepts else "unknown"
            
            # Calculate confidence and success rate
            mastery = learning_graph.get_mastery(concept_id)
            confidence = learning_graph.get_confidence(concept_id)
            
            success_rate = self._estimate_success_rate(exercise, mastery, confidence)
            
            recommendation = ExerciseRecommendation(
                exercise=exercise,
                concept_id=concept_id,
                difficulty_score=float(exercise.difficulty),
                confidence_score=confidence,
                reason=reason,
                learning_objective=f"Practice {concept_id}",
                estimated_success_rate=success_rate,
                priority=count - i  # Higher priority for earlier selections
            )
            recommendations.append(recommendation)
            
        return recommendations
    
    # Level 3: AI-powered learning path optimization
    def get_ai_optimized_exercises(
        self,
        learning_graph: LearningGraph,
        session_type: str = "practice",
        duration_minutes: int = 30,
        learning_goals: Optional[List[str]] = None
    ) -> LearningSession:
        """
        Level 3: Get AI-optimized exercise selection for optimal learning.
        
        Args:
            learning_graph: Student's learning progress
            session_type: Type of session ("practice", "assessment", "review", "challenge")
            duration_minutes: Target session duration
            learning_goals: Optional specific learning goals
            
        Returns:
            Complete learning session with optimized exercises
        """
        # Calculate target exercise count based on duration
        avg_time_per_exercise = 5  # minutes
        target_count = max(3, min(15, duration_minutes // avg_time_per_exercise))
        
        if self.llm and LLM_AVAILABLE:
            # Use AI for optimization
            recommendations = self._ai_optimize_exercise_selection(
                learning_graph, session_type, target_count, learning_goals
            )
        else:
            # Fallback to rule-based optimization
            recommendations = self._rule_based_optimize_exercise_selection(
                learning_graph, session_type, target_count, learning_goals
            )
        
        # Create learning session
        session = LearningSession(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            student_id=learning_graph.user_id,
            target_concepts=list(set([rec.concept_id for rec in recommendations])),
            exercises=recommendations,
            session_type=session_type,
            estimated_duration=sum([rec.exercise.estimated_time for rec in recommendations]),
            learning_objectives=[rec.learning_objective for rec in recommendations],
            created_at=datetime.now()
        )
        
        # Store active session
        self.active_sessions[session.session_id] = session
        
        return session
    
    def create_learning_path_session(
        self,
        learning_graph: LearningGraph,
        target_concept: str,
        session_count: int = 5
    ) -> List[LearningSession]:
        """
        Create a series of learning sessions to reach a target concept.
        
        Args:
            learning_graph: Student's learning progress
            target_concept: The concept to work towards
            session_count: Number of sessions to create
            
        Returns:
            List of learning sessions forming a learning path
        """
        sessions = []
        
        # Get learning path to target concept
        path_exercises = self.selector.select_learning_path_exercises(
            learning_graph, self.knowledge_graph, count=session_count * 5
        )
        
        # Group exercises into sessions
        exercises_per_session = len(path_exercises) // session_count
        
        for i in range(session_count):
            start_idx = i * exercises_per_session
            end_idx = start_idx + exercises_per_session if i < session_count - 1 else len(path_exercises)
            
            session_exercises = path_exercises[start_idx:end_idx]
            
            # Convert to recommendations
            recommendations = []
            for j, (exercise, concept_id, reason) in enumerate(session_exercises):
                mastery = learning_graph.get_mastery(concept_id)
                confidence = learning_graph.get_confidence(concept_id)
                success_rate = self._estimate_success_rate(exercise, mastery, confidence)
                
                recommendation = ExerciseRecommendation(
                    exercise=exercise,
                    concept_id=concept_id,
                    difficulty_score=float(exercise.difficulty),
                    confidence_score=confidence,
                    reason=reason,
                    learning_objective=f"Progress towards {target_concept}",
                    estimated_success_rate=success_rate,
                    priority=len(session_exercises) - j
                )
                recommendations.append(recommendation)
            
            # Create session
            session = LearningSession(
                session_id=f"path_session_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                student_id=learning_graph.user_id,
                target_concepts=list(set([rec.concept_id for rec in recommendations])),
                exercises=recommendations,
                session_type="learning_path",
                estimated_duration=sum([rec.exercise.estimated_time for rec in recommendations]),
                learning_objectives=[f"Session {i+1}: Progress towards {target_concept}"],
                created_at=datetime.now()
            )
            
            sessions.append(session)
            
        return sessions
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a learning session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
            
        session = self.active_sessions[session_id]
        
        # Calculate analytics
        total_exercises = len(session.exercises)
        difficulty_distribution = {}
        concept_distribution = {}
        
        for rec in session.exercises:
            # Difficulty distribution
            difficulty_level = DifficultyLevel(round(rec.difficulty_score)).name
            difficulty_distribution[difficulty_level] = difficulty_distribution.get(difficulty_level, 0) + 1
            
            # Concept distribution
            concept_distribution[rec.concept_id] = concept_distribution.get(rec.concept_id, 0) + 1
        
        if total_exercises > 0:
            avg_success_rate = sum([rec.estimated_success_rate for rec in session.exercises]) / total_exercises
            avg_difficulty = sum([rec.difficulty_score for rec in session.exercises]) / total_exercises
        else:
            avg_success_rate = 0.0
            avg_difficulty = 0.0
        
        return {
            "session_id": session_id,
            "total_exercises": total_exercises,
            "estimated_duration": session.estimated_duration,
            "average_success_rate": avg_success_rate,
            "average_difficulty": avg_difficulty,
            "difficulty_distribution": difficulty_distribution,
            "concept_distribution": concept_distribution,
            "learning_objectives": session.learning_objectives,
            "session_type": session.session_type
        }
    
    def _estimate_success_rate(self, exercise: Exercise, mastery: float, confidence: float) -> float:
        """Estimate success rate for an exercise given student's mastery and confidence."""
        
        # Base success rate from mastery
        base_rate = mastery
        
        # Adjust for exercise difficulty vs mastery
        difficulty_factor = exercise.difficulty / 5.0  # Normalize to 0-1
        difficulty_adjustment = (mastery - difficulty_factor) * 0.3
        
        # Adjust for confidence
        confidence_adjustment = (confidence - 0.5) * 0.2
        
        # Combine factors
        estimated_rate = base_rate + difficulty_adjustment + confidence_adjustment
        
        # Clamp to reasonable range
        return max(0.1, min(0.95, estimated_rate))
    
    def _ai_optimize_exercise_selection(
        self,
        learning_graph: LearningGraph,
        session_type: str,
        target_count: int,
        learning_goals: Optional[List[str]] = None
    ) -> List[ExerciseRecommendation]:
        """Use AI to optimize exercise selection."""
        
        if not self.llm:
            return self._rule_based_optimize_exercise_selection(
                learning_graph, session_type, target_count, learning_goals
            )
        
        # Prepare context for AI
        mastery_summary = {
            concept_id: {
                "mastery": learning_graph.get_mastery(concept_id),
                "confidence": learning_graph.get_confidence(concept_id)
            }
            for concept_id in learning_graph.concept_mastery.keys()
        }
        
        # Create AI prompt
        prompt = f"""
        You are an AI math tutor optimizing exercise selection for a student.
        
        Student Progress:
        {json.dumps(mastery_summary, indent=2)}
        
        Session Type: {session_type}
        Target Exercise Count: {target_count}
        Learning Goals: {learning_goals or "General improvement"}
        
        Available Concepts: {[concept.name for concept in self.knowledge_graph.get_all_concepts()[:10]]}
        
        Please recommend {target_count} concepts to focus on, ordered by priority.
        Consider:
        1. Concepts where the student is struggling (low mastery or confidence)
        2. Prerequisites for concepts the student wants to learn
        3. Appropriate difficulty progression
        4. Session type requirements
        
        Respond with a JSON list of concept names only.
        """
        
        try:
            # Get AI response
            response = self.llm.generate(prompt)
            
            # Parse AI response to get concept recommendations
            # This is a simplified implementation - in practice, you'd want more robust parsing
            ai_concepts = json.loads(response)
            
            # Convert concept names to IDs and get exercises
            recommended_concepts = []
            for concept_name in ai_concepts:
                for concept in self.knowledge_graph.get_all_concepts():
                    if concept_name.lower() in concept.name.lower():
                        recommended_concepts.append(concept.id)
                        break
            
            # Get exercises for AI-recommended concepts
            return self.get_adaptive_exercises(
                learning_graph, recommended_concepts[:target_count], target_count
            )
            
        except Exception as e:
            print(f"AI optimization failed: {e}, falling back to rule-based")
            return self._rule_based_optimize_exercise_selection(
                learning_graph, session_type, target_count, learning_goals
            )
    
    def _rule_based_optimize_exercise_selection(
        self,
        learning_graph: LearningGraph,
        session_type: str,
        target_count: int,
        learning_goals: Optional[List[str]] = None
    ) -> List[ExerciseRecommendation]:
        """Rule-based exercise optimization as fallback."""
        
        if session_type == "assessment":
            # For assessment, test a variety of concepts
            all_concepts = list(learning_graph.concept_mastery.keys())
            selected_concepts = random.sample(all_concepts, min(target_count, len(all_concepts)))
            difficulty_adjustment = 0.0  # Standard difficulty
            
        elif session_type == "review":
            # For review, focus on previously learned concepts
            mastered_concepts = learning_graph.get_mastered_concepts(threshold=0.6)
            selected_concepts = mastered_concepts[:target_count]
            difficulty_adjustment = -0.5  # Easier for review
            
        elif session_type == "challenge":
            # For challenge, focus on harder concepts
            struggling_concepts = learning_graph.get_struggling_concepts(threshold=0.5)
            selected_concepts = struggling_concepts[:target_count]
            difficulty_adjustment = 0.5  # Harder for challenge
            
        else:  # "practice" or default
            # For practice, focus on concepts that need work
            selected_concepts = None  # Let adaptive selection choose
            difficulty_adjustment = 0.0
        
        return self.get_adaptive_exercises(
            learning_graph, selected_concepts, target_count, difficulty_adjustment
        ) 