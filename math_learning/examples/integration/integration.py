"""
Math Learning Platform Integration Module.

This module provides a unified interface that integrates the new exercise system
with the existing math learning platform components including:
- AI Tutoring Agent
- Knowledge Graph
- Learning Graph
- Exercise Bank
- Recommendation System
"""

import asyncio
import logging
import sys
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .knowledge_graph.graph import KnowledgeGraph
    from .learning_graph.user_model import LearningGraph
    from .exercises.exercise_bank import ExerciseBank
    from .exercises.exercise_selector import ExerciseSelector
    from .exercises.exercise_system import ExerciseSystem
    from .recommendation.gap_analyzer import GapAnalyzer
    from .recommendation.recommender import Recommender
    from .ai_agent.math_tutoring_agent import MathTutoringAgent
except ImportError:
    # Fallback for direct execution
    from math_learning.knowledge_graph.graph import KnowledgeGraph
    from math_learning.learning_graph.user_model import LearningGraph
    from math_learning.exercises.exercise_bank import ExerciseBank
    from math_learning.exercises.exercise_selector import ExerciseSelector
    from math_learning.exercises.exercise_system import ExerciseSystem
    from math_learning.recommendation.gap_analyzer import GapAnalyzer
    from math_learning.recommendation.recommender import Recommender
    from math_learning.ai_agent.math_tutoring_agent import MathTutoringAgent

# Import LLM if available
try:
    from llm.llm import LLMBase
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMBase = None

logger = logging.getLogger(__name__)


class IntegratedMathLearningPlatform:
    """
    Comprehensive math learning platform that integrates all components.
    
    This platform provides:
    - Intelligent exercise selection and generation
    - AI-powered tutoring and guidance
    - Adaptive learning path planning
    - Progress tracking and analytics
    - Knowledge gap analysis and remediation
    """
    
    def __init__(self, llm: Optional[LLMBase] = None):
        """
        Initialize the integrated math learning platform.
        
        Args:
            llm: Optional language model for AI-powered features
        """
        self.llm = llm
        
        # Core components
        self.knowledge_graph = KnowledgeGraph(name="Integrated Math Knowledge")
        self.exercise_bank = ExerciseBank()
        
        # Advanced exercise system
        self.exercise_selector = ExerciseSelector(
            exercise_bank=self.exercise_bank,
            knowledge_graph=self.knowledge_graph
        )
        self.exercise_system = ExerciseSystem(
            exercise_bank=self.exercise_bank,
            knowledge_graph=self.knowledge_graph,
            llm=self.llm
        )
        
        # Analytics and recommendation
        self.gap_analyzer = GapAnalyzer(self.knowledge_graph)
        self.recommender = Recommender(self.knowledge_graph, self.exercise_bank)
        
        # Student learning graphs
        self.student_graphs: Dict[str, LearningGraph] = {}
        
        # AI tutoring agents per student
        self.tutoring_agents: Dict[str, MathTutoringAgent] = {}
        
        # Platform analytics
        self.platform_analytics = {
            "total_students": 0,
            "total_sessions": 0,
            "total_exercises_completed": 0,
            "average_session_duration": 0,
            "concept_mastery_rates": {},
            "platform_start_time": datetime.now().isoformat()
        }
        
        logger.info("Integrated Math Learning Platform initialized")
    
    def register_student(self, student_id: str, student_name: str = "") -> Dict[str, Any]:
        """
        Register a new student on the platform.
        
        Args:
            student_id: Unique identifier for the student
            student_name: Optional display name for the student
            
        Returns:
            Registration confirmation with student profile
        """
        if student_id in self.student_graphs:
            return {
                "success": False,
                "message": f"Student {student_id} is already registered",
                "student_id": student_id
            }
        
        # Create learning graph for student
        learning_graph = LearningGraph(
            user_id=student_id, 
            name=student_name or f"Student {student_id}"
        )
        self.student_graphs[student_id] = learning_graph
        
        # Create AI tutoring agent for student
        ai_tutor = MathTutoringAgent(
            llm=self.llm,
            learning_graph=learning_graph,
            user_model=learning_graph,
            recommender=self.recommender,
            gap_analyzer=self.gap_analyzer
        )
        self.tutoring_agents[student_id] = ai_tutor
        
        # Update platform analytics
        self.platform_analytics["total_students"] += 1
        
        logger.info(f"Registered student: {student_id}")
        
        return {
            "success": True,
            "message": f"Successfully registered student {student_id}",
            "student_id": student_id,
            "student_name": student_name,
            "learning_graph_initialized": True,
            "ai_tutor_available": True,
            "registration_time": datetime.now().isoformat()
        }
    
    def get_personalized_exercises(self, student_id: str, concept: str = "adaptive", 
                                 difficulty: float = 3.0, count: int = 5,
                                 session_type: str = "practice") -> Dict[str, Any]:
        """
        Get personalized exercises for a student using the advanced exercise system.
        
        Args:
            student_id: Student identifier
            concept: Target concept or "adaptive" for AI selection
            difficulty: Difficulty score (1.0-5.0)
            count: Number of exercises requested
            session_type: Type of session (practice, assessment, review, challenge)
            
        Returns:
            Personalized exercise selection with analytics
        """
        if student_id not in self.student_graphs:
            return {
                "success": False,
                "error": f"Student {student_id} not registered",
                "exercises": []
            }
        
        learning_graph = self.student_graphs[student_id]
        
        try:
            if concept == "adaptive" or len(learning_graph.concepts) > 3:
                # Use AI-optimized exercise selection
                session = self.exercise_system.get_ai_optimized_exercises(
                    learning_graph=learning_graph,
                    session_type=session_type,
                    duration_minutes=count * 6,  # Estimate 6 minutes per exercise
                    learning_goals=[concept] if concept != "adaptive" else None
                )
                
                if session:
                    exercises = [ex.to_dict() for ex in session.exercises]
                    return {
                        "success": True,
                        "student_id": student_id,
                        "exercises": exercises[:count],  # Limit to requested count
                        "session_analytics": {
                            "session_id": session.session_id,
                            "session_type": session.session_type,
                            "learning_objectives": session.learning_objectives,
                            "estimated_duration": session.estimated_duration,
                            "target_concepts": session.target_concepts
                        },
                        "generation_method": "ai_optimized",
                        "total_exercises": len(exercises[:count])
                    }
            
            # Fallback to adaptive selection
            adaptive_exercises = self.exercise_selector.select_adaptive_exercises(
                learning_graph=learning_graph,
                target_concepts=[concept] if concept != "adaptive" else None,
                count=count
            )
            
            if adaptive_exercises:
                exercises = []
                for ex, reason in adaptive_exercises:
                    exercise_dict = ex.to_dict()
                    exercise_dict["selection_reason"] = reason
                    exercises.append(exercise_dict)
                
                return {
                    "success": True,
                    "student_id": student_id,
                    "exercises": exercises,
                    "session_analytics": {
                        "adaptive_selection": True,
                        "target_concepts": [concept] if concept != "adaptive" else None
                    },
                    "generation_method": "adaptive_selection",
                    "total_exercises": len(exercises)
                }
            
            # Final fallback to basic selection
            basic_exercises = self.exercise_selector.select_exercises_by_concept_and_difficulty(
                concept_id=concept if concept != "adaptive" else "algebra",
                difficulty_score=difficulty,
                count=count
            )
            
            exercises = [ex.to_dict() for ex in basic_exercises] if basic_exercises else []
            
            return {
                "success": True,
                "student_id": student_id,
                "exercises": exercises,
                "session_analytics": {
                    "basic_selection": True,
                    "target_difficulty": difficulty
                },
                "generation_method": "basic_selection",
                "total_exercises": len(exercises)
            }
            
        except Exception as e:
            logger.error(f"Error generating exercises for {student_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to generate exercises: {str(e)}",
                "student_id": student_id,
                "exercises": []
            }
    
    async def start_tutoring_session(self, student_id: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start an AI tutoring session for a student.
        
        Args:
            student_id: Student identifier
            session_context: Session context and goals
            
        Returns:
            Session initialization result
        """
        if student_id not in self.tutoring_agents:
            return {
                "success": False,
                "error": f"No AI tutor available for student {student_id}"
            }
        
        ai_tutor = self.tutoring_agents[student_id]
        
        try:
            await ai_tutor.start_tutoring_session(student_id, session_context)
            
            # Update platform analytics
            self.platform_analytics["total_sessions"] += 1
            
            return {
                "success": True,
                "student_id": student_id,
                "session_started": True,
                "ai_tutor_ready": True,
                "session_context": session_context,
                "start_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error starting tutoring session for {student_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to start tutoring session: {str(e)}"
            }
    
    async def get_ai_tutoring_response(self, student_id: str, student_input: str) -> Dict[str, Any]:
        """
        Get an AI tutoring response for student input.
        
        Args:
            student_id: Student identifier
            student_input: Student's question or input
            
        Returns:
            AI tutor response
        """
        if student_id not in self.tutoring_agents:
            return {
                "success": False,
                "error": f"No AI tutor available for student {student_id}"
            }
        
        ai_tutor = self.tutoring_agents[student_id]
        
        try:
            response = await ai_tutor.tutor(student_input)
            
            return {
                "success": True,
                "student_id": student_id,
                "student_input": student_input,
                "tutor_response": response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting AI tutoring response for {student_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get tutoring response: {str(e)}"
            }
    
    def record_exercise_completion(self, student_id: str, exercise_id: str, 
                                 success: bool, time_spent: int = 0) -> Dict[str, Any]:
        """
        Record exercise completion and update student progress.
        
        Args:
            student_id: Student identifier
            exercise_id: Exercise identifier
            success: Whether the exercise was completed successfully
            time_spent: Time spent on exercise in seconds
            
        Returns:
            Progress update result
        """
        if student_id not in self.student_graphs:
            return {
                "success": False,
                "error": f"Student {student_id} not registered"
            }
        
        learning_graph = self.student_graphs[student_id]
        
        try:
            # Find the exercise to get concept information
            exercise = self.exercise_bank.get_exercise(exercise_id)
            
            if exercise:
                # Record the exercise attempt
                for rel in exercise.concept_relationships:
                    concept_id = rel["concept_id"]
                    weight = rel["weight"]
                    learning_graph.record_exercise_attempt(
                        concept_id=concept_id,
                        exercise_id=exercise_id,
                        success=success,
                        weight=weight
                    )
                
                # Update platform analytics
                self.platform_analytics["total_exercises_completed"] += 1
                
                # Get updated mastery levels
                mastery_updates = {}
                for rel in exercise.concept_relationships:
                    concept_id = rel["concept_id"]
                    mastery_updates[concept_id] = learning_graph.get_mastery(concept_id)
                
                return {
                    "success": True,
                    "student_id": student_id,
                    "exercise_id": exercise_id,
                    "exercise_success": success,
                    "time_spent": time_spent,
                    "mastery_updates": mastery_updates,
                    "concepts_affected": [rel["concept_id"] for rel in exercise.concept_relationships],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Exercise {exercise_id} not found"
                }
                
        except Exception as e:
            logger.error(f"Error recording exercise completion for {student_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to record exercise completion: {str(e)}"
            }
    
    def get_student_analytics(self, student_id: str) -> Dict[str, Any]:
        """
        Get comprehensive analytics for a student.
        
        Args:
            student_id: Student identifier
            
        Returns:
            Student analytics and progress summary
        """
        if student_id not in self.student_graphs:
            return {
                "success": False,
                "error": f"Student {student_id} not registered"
            }
        
        learning_graph = self.student_graphs[student_id]
        
        try:
            # Get basic learning graph analytics
            total_exercises = len(learning_graph.exercise_history)
            successful_exercises = sum(1 for ex in learning_graph.exercise_history.values() if ex.success)
            success_rate = successful_exercises / total_exercises if total_exercises > 0 else 0
            
            # Get mastery levels
            mastery_levels = {}
            for concept_id in learning_graph.concepts:
                mastery_levels[concept_id] = learning_graph.get_mastery(concept_id)
            
            # Calculate average mastery
            avg_mastery = sum(mastery_levels.values()) / len(mastery_levels) if mastery_levels else 0
            
            # Get knowledge gaps
            gaps = self.gap_analyzer.detect_critical_gaps(learning_graph)
            
            return {
                "success": True,
                "student_id": student_id,
                "analytics": {
                    "total_exercises_attempted": total_exercises,
                    "successful_exercises": successful_exercises,
                    "success_rate": success_rate,
                    "average_mastery": avg_mastery,
                    "mastery_levels": mastery_levels,
                    "knowledge_gaps": [gap.to_dict() if hasattr(gap, 'to_dict') else str(gap) for gap in gaps],
                    "concepts_tracked": len(learning_graph.concepts),
                    "learning_graph_name": learning_graph.name
                },
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics for {student_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get student analytics: {str(e)}"
            }
    
    def get_platform_analytics(self) -> Dict[str, Any]:
        """
        Get overall platform analytics.
        
        Returns:
            Platform-wide analytics and statistics
        """
        try:
            # Calculate concept mastery rates across all students
            concept_mastery_totals = {}
            concept_student_counts = {}
            
            for learning_graph in self.student_graphs.values():
                for concept_id in learning_graph.concepts:
                    mastery = learning_graph.get_mastery(concept_id)
                    if concept_id not in concept_mastery_totals:
                        concept_mastery_totals[concept_id] = 0
                        concept_student_counts[concept_id] = 0
                    concept_mastery_totals[concept_id] += mastery
                    concept_student_counts[concept_id] += 1
            
            # Calculate average mastery per concept
            concept_mastery_rates = {}
            for concept_id in concept_mastery_totals:
                concept_mastery_rates[concept_id] = (
                    concept_mastery_totals[concept_id] / concept_student_counts[concept_id]
                )
            
            # Update platform analytics
            self.platform_analytics["concept_mastery_rates"] = concept_mastery_rates
            
            return {
                "success": True,
                "platform_analytics": self.platform_analytics,
                "exercise_bank_size": len(self.exercise_bank.exercises),
                "knowledge_graph_concepts": len(self.knowledge_graph.get_all_concepts()),
                "active_students": len(self.student_graphs),
                "ai_tutors_active": len(self.tutoring_agents),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting platform analytics: {e}")
            return {
                "success": False,
                "error": f"Failed to get platform analytics: {str(e)}"
            }


# Convenience function for easy platform initialization
def create_integrated_platform(llm: Optional[LLMBase] = None) -> IntegratedMathLearningPlatform:
    """
    Create and initialize an integrated math learning platform.
    
    Args:
        llm: Optional language model for AI features
        
    Returns:
        Initialized platform instance
    """
    platform = IntegratedMathLearningPlatform(llm=llm)
    
    # Load default knowledge graph and exercises
    try:
        # This would typically load from configuration or database
        logger.info("Loading default knowledge graph and exercises...")
        # platform.knowledge_graph.load_default_algebra_concepts()
        # platform.exercise_bank.load_default_exercises()
    except Exception as e:
        logger.warning(f"Could not load default content: {e}")
    
    return platform 