"""
Exercise Generation Tool for the Math Tutoring Agent.

This tool generates personalized exercises using the recommender system
and learning graph to create targeted practice opportunities.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

# Fix import paths to match actual structure
from math_learning.learning_graph import LearningGraph
from math_learning.recommendation.recommender import Recommender


class ExerciseGenerationTool:
    """
    Tool for generating personalized math exercises.
    
    This tool integrates with the recommender system and learning graph to:
    - Generate exercises targeted to specific concepts
    - Adapt difficulty based on student performance
    - Create practice sets for identified knowledge gaps
    - Provide varied exercise types for comprehensive practice
    """
    
    def __init__(self, recommender: Recommender, learning_graph: LearningGraph):
        """
        Initialize the Exercise Generation Tool.
        
        Args:
            recommender: Exercise recommendation system
            learning_graph: Student's learning progress graph
        """
        self.recommender = recommender
        self.learning_graph = learning_graph
        self.name = "exercise_generation"
        self.description = "Generate personalized math exercises based on student needs and performance"
        
    async def execute(self, student_id: str, concept: str, difficulty: str = "medium", 
                     count: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute exercise generation for a student.
        
        Args:
            student_id: Unique identifier for the student
            concept: The mathematical concept to generate exercises for
            difficulty: Difficulty level (easy, medium, hard, adaptive)
            count: Number of exercises to generate
            **kwargs: Additional parameters
            
        Returns:
            Dict containing generated exercises and metadata
        """
        try:
            # Get student's current mastery level for adaptive difficulty
            mastery_level = self.learning_graph.get_mastery(concept, default=0.0)
            
            # Adjust difficulty based on mastery if adaptive
            if difficulty == "adaptive":
                if mastery_level < 0.3:
                    difficulty = "easy"
                elif mastery_level < 0.7:
                    difficulty = "medium"
                else:
                    difficulty = "hard"
            
            # Generate exercises using the recommender
            exercises = []
            try:
                # Try to get recommendations from the recommender
                recommendations = self.recommender.recommend_exercises(self.learning_graph, max_exercises=count)
                
                for i, rec in enumerate(recommendations[:count]):
                    exercise = {
                        "id": f"ex_{concept}_{i+1}",
                        "title": getattr(rec.exercise, 'title', f"Exercise {i+1}"),
                        "concept": concept,
                        "difficulty": difficulty,
                        "type": self._determine_exercise_type(concept),
                        "content": self._generate_exercise_content(concept, difficulty, i+1),
                        "solution": self._generate_solution(concept, difficulty, i+1),
                        "hints": self._generate_hints(concept, difficulty),
                        "estimated_time": self._estimate_time(difficulty),
                        "learning_objectives": [f"Practice {concept}", f"Apply {concept} concepts"]
                    }
                    exercises.append(exercise)
                    
            except Exception as e:
                # If recommender fails, generate basic exercises
                for i in range(count):
                    exercise = {
                        "id": f"ex_{concept}_{i+1}",
                        "title": f"{concept.replace('_', ' ').title()} Exercise {i+1}",
                        "concept": concept,
                        "difficulty": difficulty,
                        "type": self._determine_exercise_type(concept),
                        "content": self._generate_exercise_content(concept, difficulty, i+1),
                        "solution": self._generate_solution(concept, difficulty, i+1),
                        "hints": self._generate_hints(concept, difficulty),
                        "estimated_time": self._estimate_time(difficulty),
                        "learning_objectives": [f"Practice {concept}", f"Apply {concept} concepts"]
                    }
                    exercises.append(exercise)
            
            # Generate practice set metadata
            practice_set = {
                "student_id": student_id,
                "concept": concept,
                "difficulty": difficulty,
                "total_exercises": len(exercises),
                "estimated_total_time": sum(ex["estimated_time"] for ex in exercises),
                "exercises": exercises,
                "generated_at": datetime.now().isoformat(),
                "mastery_level": mastery_level,
                "recommendations": self._generate_practice_recommendations(concept, mastery_level, difficulty),
                "summary": f"Generated {len(exercises)} {difficulty} exercises for {concept}"
            }
            
            return practice_set
            
        except Exception as e:
            return {
                "error": f"Failed to generate exercises for {concept}: {str(e)}",
                "student_id": student_id,
                "concept": concept,
                "generated_at": datetime.now().isoformat()
            }
    
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
            },
            "quadratic_equations": {
                "easy": f"Factor: x² + {number+1}x + {number}",
                "medium": f"Solve: x² + {number}x - {number+3} = 0",
                "hard": f"Solve using the quadratic formula: {number}x² + {number+1}x + {number-2} = 0"
            }
        }
        
        if concept in content_templates and difficulty in content_templates[concept]:
            return content_templates[concept][difficulty]
        else:
            return f"Practice problem {number} for {concept.replace('_', ' ')}"
    
    def _generate_solution(self, concept: str, difficulty: str, number: int) -> str:
        """Generate solution for the exercise."""
        # This would typically be more sophisticated
        return f"Solution for {concept} exercise {number} (difficulty: {difficulty})"
    
    def _generate_hints(self, concept: str, difficulty: str) -> List[str]:
        """Generate helpful hints for the exercise."""
        hint_templates = {
            "fractions": [
                "Remember to find a common denominator",
                "Simplify your answer to lowest terms",
                "Cross multiply when solving equations"
            ],
            "algebra": [
                "Isolate the variable on one side",
                "Use inverse operations",
                "Check your answer by substituting back"
            ],
            "quadratic_equations": [
                "Look for common factors first",
                "Remember the quadratic formula: x = (-b ± √(b²-4ac))/2a",
                "Check if the equation can be factored"
            ]
        }
        
        return hint_templates.get(concept, ["Think step by step", "Review the concept if needed"])
    
    def _estimate_time(self, difficulty: str) -> int:
        """Estimate time needed for exercise in minutes."""
        time_mapping = {
            "easy": 3,
            "medium": 5,
            "hard": 8
        }
        return time_mapping.get(difficulty, 5)
    
    def _generate_practice_recommendations(self, concept: str, mastery_level: float, difficulty: str) -> List[str]:
        """Generate recommendations for practice approach."""
        recommendations = []
        
        if mastery_level < 0.3:
            recommendations.append("Start with the easier exercises to build confidence")
            recommendations.append("Review concept explanations before attempting")
            recommendations.append("Don't hesitate to use hints")
        elif mastery_level < 0.7:
            recommendations.append("Focus on understanding the process, not just getting answers")
            recommendations.append("Try to solve without hints first")
            recommendations.append("Review mistakes to identify patterns")
        else:
            recommendations.append("Challenge yourself with the harder problems")
            recommendations.append("Try to solve problems using multiple methods")
            recommendations.append("Focus on speed and accuracy")
        
        if difficulty == "adaptive":
            recommendations.append(f"Difficulty automatically adjusted based on your {mastery_level:.1%} mastery")
        
        return recommendations 