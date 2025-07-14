"""
Exercise Generation Tool for the Math Tutoring Agent.

This tool generates personalized exercises using the advanced exercise selection system
with difficulty scoring, learning graph integration, and AI-powered path optimization.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os

# Add the math_learning directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import the new exercise system
from math_learning.exercises.exercise_selector import ExerciseSelector
from math_learning.exercises.exercise_system import ExerciseSystem
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.recommendation.recommender import Recommender


class ExerciseGenerationTool:
    """
    Advanced exercise generation tool using the new exercise selection system.
    
    This tool integrates with:
    - ExerciseSelector for intelligent exercise selection
    - ExerciseSystem for high-level learning path management
    - Learning graph for student progress tracking
    - AI-powered session planning and optimization
    """
    
    def __init__(self, recommender: Recommender, learning_graph: LearningGraph):
        """
        Initialize the Exercise Generation Tool with advanced capabilities.
        
        Args:
            recommender: Exercise recommendation system (legacy compatibility)
            learning_graph: Student's learning progress graph
        """
        self.recommender = recommender
        self.learning_graph = learning_graph
        self.name = "exercise_generation"
        self.description = "Generate personalized math exercises using advanced difficulty scoring and AI optimization"
        
        # Initialize the new exercise system components
        # Get the exercise bank and knowledge graph from the recommender
        self.exercise_bank = getattr(recommender, 'exercise_bank', ExerciseBank())
        self.knowledge_graph = getattr(recommender, 'knowledge_graph', KnowledgeGraph())
        
        self.exercise_selector = ExerciseSelector(
            exercise_bank=self.exercise_bank,
            knowledge_graph=self.knowledge_graph
        )
        self.exercise_system = ExerciseSystem(
            exercise_bank=self.exercise_bank,
            knowledge_graph=self.knowledge_graph,
            llm=None  # No LLM integration for now
        )
        
    async def execute(self, student_id: str, concept: str, difficulty: str = "medium", 
                     count: int = 5, session_type: str = "practice", 
                     duration_minutes: int = 30, **kwargs) -> Dict[str, Any]:
        """
        Execute advanced exercise generation for a student.
        
        Args:
            student_id: Unique identifier for the student
            concept: The mathematical concept to generate exercises for
            difficulty: Difficulty level (easy, medium, hard, adaptive) or numeric (1.0-5.0)
            count: Number of exercises to generate
            session_type: Type of session (practice, assessment, review, challenge)
            duration_minutes: Target session duration in minutes
            **kwargs: Additional parameters
            
        Returns:
            Dict containing generated exercises and comprehensive metadata
        """
        try:
            # Convert difficulty to numeric if needed
            difficulty_score = self._convert_difficulty_to_score(difficulty)
            
            # Determine the best approach based on available data
            if hasattr(self.learning_graph, 'concepts') and len(self.learning_graph.concepts) > 0:
                # Use Level 2: Adaptive selection with learning graph
                return await self._generate_adaptive_exercises(
                    student_id, concept, difficulty_score, count, session_type, duration_minutes
                )
            else:
                # Use Level 1: Basic concept + difficulty selection
                return await self._generate_basic_exercises(
                    student_id, concept, difficulty_score, count, session_type
                )
                
        except Exception as e:
            return {
                "error": f"Failed to generate exercises for {concept}: {str(e)}",
                "student_id": student_id,
                "concept": concept,
                "generated_at": datetime.now().isoformat(),
                "fallback_used": True
            }
    
    async def _generate_adaptive_exercises(self, student_id: str, concept: str, 
                                         difficulty_score: float, count: int,
                                         session_type: str, duration_minutes: int) -> Dict[str, Any]:
        """Generate exercises using adaptive selection with learning graph."""
        
        # Use Level 3: AI-powered session planning
        session_result = self.exercise_system.get_ai_optimized_exercises(
            learning_graph=self.learning_graph,
            session_type=session_type,
            duration_minutes=duration_minutes,
            learning_goals=[concept] if concept != "adaptive" else None
        )
        
        if session_result and hasattr(session_result, 'exercises'):
            exercises = [ex.to_dict() for ex in session_result.exercises]
            
            # If we need more exercises, supplement with Level 2 selection
            if len(exercises) < count:
                additional_needed = count - len(exercises)
                level2_result = self.exercise_selector.select_adaptive_exercises(
                    learning_graph=self.learning_graph,
                    target_concepts=[concept] if concept != "adaptive" else None,
                    count=additional_needed
                )
                
                if level2_result:
                    # Convert to dict format
                    for ex, reason in level2_result:
                        exercises.append({
                            **ex.to_dict(),
                            "selection_reason": reason
                        })
            
            return {
                "student_id": student_id,
                "concept": concept,
                "difficulty": difficulty_score,
                "session_type": session_type,
                "total_exercises": len(exercises),
                "exercises": exercises,
                "session_analytics": {
                    "session_id": session_result.session_id,
                    "learning_objectives": session_result.learning_objectives,
                    "target_concepts": session_result.target_concepts
                },
                "estimated_duration": session_result.estimated_duration,
                "learning_objectives": session_result.learning_objectives,
                "generated_at": datetime.now().isoformat(),
                "generation_method": "adaptive_ai_powered",
                "success": True
            }
        else:
            # Fallback to Level 2 if AI planning fails
            return await self._generate_level2_exercises(student_id, concept, difficulty_score, count)
    
    async def _generate_level2_exercises(self, student_id: str, concept: str, 
                                       difficulty_score: float, count: int) -> Dict[str, Any]:
        """Generate exercises using Level 2: Adaptive selection with learning graph."""
        
        result = self.exercise_selector.select_adaptive_exercises(
            learning_graph=self.learning_graph,
            target_concepts=[concept] if concept != "adaptive" else None,
            count=count
        )
        
        if result:
            exercises = []
            for ex, reason in result:
                exercise_dict = ex.to_dict()
                exercise_dict["selection_reason"] = reason
                exercises.append(exercise_dict)
            
            return {
                "student_id": student_id,
                "concept": concept,
                "difficulty": difficulty_score,
                "total_exercises": len(exercises),
                "exercises": exercises,
                "student_analytics": {
                    "adaptive_selection": True,
                    "target_concepts": [concept] if concept != "adaptive" else None
                },
                "estimated_duration": self._estimate_session_duration(exercises),
                "learning_objectives": [f"Practice {concept}", f"Improve mastery in {concept}"],
                "generated_at": datetime.now().isoformat(),
                "generation_method": "adaptive_learning_graph",
                "success": True
            }
        else:
            # Fallback to Level 1
            return await self._generate_basic_exercises(student_id, concept, difficulty_score, count, "practice")
    
    async def _generate_basic_exercises(self, student_id: str, concept: str, 
                                      difficulty_score: float, count: int, 
                                      session_type: str) -> Dict[str, Any]:
        """Generate exercises using Level 1: Basic concept + difficulty selection."""
        
        result = self.exercise_selector.select_exercises_by_concept_and_difficulty(
            concept_id=concept,
            difficulty_score=difficulty_score,
            count=count
        )
        
        if result:
            exercises = [ex.to_dict() for ex in result]
            
            return {
                "student_id": student_id,
                "concept": concept,
                "difficulty": difficulty_score,
                "session_type": session_type,
                "total_exercises": len(exercises),
                "exercises": exercises,
                "selection_analytics": {
                    "concept_difficulty_selection": True,
                    "target_difficulty": difficulty_score
                },
                "estimated_duration": self._estimate_session_duration(exercises),
                "learning_objectives": [f"Practice {concept}", f"Build foundational skills"],
                "generated_at": datetime.now().isoformat(),
                "generation_method": "basic_concept_difficulty",
                "success": True
            }
        else:
            # Final fallback to legacy system
            return await self._generate_legacy_exercises(student_id, concept, difficulty_score, count)
    
    async def _generate_legacy_exercises(self, student_id: str, concept: str, 
                                       difficulty_score: float, count: int) -> Dict[str, Any]:
        """Generate exercises using the legacy system as final fallback."""
        
        # Convert numeric difficulty back to string for legacy system
        difficulty_str = self._convert_score_to_difficulty(difficulty_score)
        
        exercises = []
        for i in range(count):
            exercise = {
                "id": f"legacy_{concept}_{i+1}",
                "title": f"{concept.replace('_', ' ').title()} Exercise {i+1}",
                "concept": concept,
                "difficulty": difficulty_str,
                "type": self._determine_exercise_type(concept),
                "content": self._generate_exercise_content(concept, difficulty_str, i+1),
                "solution": self._generate_solution(concept, difficulty_str, i+1),
                "hints": self._generate_hints(concept, difficulty_str),
                "estimated_time": self._estimate_time(difficulty_str),
                "learning_objectives": [f"Practice {concept}"]
            }
            exercises.append(exercise)
        
        return {
            "student_id": student_id,
            "concept": concept,
            "difficulty": difficulty_score,
            "total_exercises": len(exercises),
            "exercises": exercises,
            "estimated_duration": sum(ex["estimated_time"] for ex in exercises),
            "generated_at": datetime.now().isoformat(),
            "generation_method": "legacy_fallback",
            "success": True
        }
    
    def _convert_difficulty_to_score(self, difficulty: str) -> float:
        """Convert string difficulty to numeric score."""
        if isinstance(difficulty, (int, float)):
            return float(difficulty)
        
        difficulty_mapping = {
            "easy": 2.0,
            "medium": 3.0,
            "hard": 4.0,
            "adaptive": 3.0  # Default for adaptive
        }
        return difficulty_mapping.get(difficulty.lower(), 3.0)
    
    def _convert_score_to_difficulty(self, score: float) -> str:
        """Convert numeric score to string difficulty."""
        if score <= 2.5:
            return "easy"
        elif score <= 3.5:
            return "medium"
        else:
            return "hard"
    
    def _estimate_session_duration(self, exercises: List[Dict]) -> int:
        """Estimate session duration based on exercises."""
        total_minutes = 0
        for exercise in exercises:
            # Use estimated_time if available, otherwise estimate based on difficulty
            if "estimated_time" in exercise:
                total_minutes += exercise["estimated_time"]
            else:
                difficulty = exercise.get("difficulty", 3.0)
                if isinstance(difficulty, str):
                    difficulty = self._convert_difficulty_to_score(difficulty)
                total_minutes += int(3 + difficulty * 2)  # 5-13 minutes per exercise
        return total_minutes
    
    # Legacy methods for backward compatibility
    def _determine_exercise_type(self, concept: str) -> str:
        """Determine the most appropriate exercise type for a concept."""
        type_mapping = {
            "fractions": "computation",
            "algebra": "equation_solving",
            "geometry": "problem_solving",
            "quadratic_equations": "equation_solving",
            "calculus": "computation",
            "trigonometry": "computation",
            "statistics": "data_analysis"
        }
        return type_mapping.get(concept, "problem_solving")
    
    def _generate_exercise_content(self, concept: str, difficulty: str, number: int) -> str:
        """Generate exercise content based on concept and difficulty."""
        content_templates = {
            "fractions": {
                "easy": f"Simplify the fraction: {number+1}/{(number+1)*2}",
                "medium": f"Add these fractions: {number}/{number+2} + {number+1}/{number+3}",
                "hard": f"Solve: {number}x/{number+1} + {number+2}/{number+3} = {number+4}/{number+5}"
            },
            "algebra": {
                "easy": f"Solve for x: x + {number} = {number+5}",
                "medium": f"Solve for x: {number}x + {number+1} = {number*2+3}",
                "hard": f"Solve for x: {number}x² + {number+1}x + {number+2} = 0"
            }
        }
        
        if concept in content_templates and difficulty in content_templates[concept]:
            return content_templates[concept][difficulty]
        else:
            return f"Practice problem {number} for {concept.replace('_', ' ')}"
    
    def _generate_solution(self, concept: str, difficulty: str, number: int) -> str:
        """Generate solution for the exercise."""
        return f"Solution for {concept} exercise {number} (difficulty: {difficulty})"
    
    def _generate_hints(self, concept: str, difficulty: str) -> List[str]:
        """Generate hints for the exercise."""
        return [
            f"Remember the key principles of {concept.replace('_', ' ')}",
            f"Start with the basics and work step by step",
            f"Check your work by substituting back into the original problem"
        ]
    
    def _estimate_time(self, difficulty: str) -> int:
        """Estimate time needed for an exercise."""
        time_mapping = {
            "easy": 5,
            "medium": 8,
            "hard": 12
        }
        return time_mapping.get(difficulty, 8) 