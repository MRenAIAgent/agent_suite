"""
Geometry Learning System module.

This module provides a comprehensive geometry learning system that inherits from
and extends the base math learning components.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..exercises.exercise_bank import ExerciseBank
from ..exercises.exercise import Exercise
from .geometry_knowledge_graph import GeometryKnowledgeGraph
from .geometry_learning_graph import GeometryLearningGraph
from .geometry_gap_analyzer import GeometryGapAnalyzer
from .geometry_recommender import GeometryRecommender, GeometryExerciseRecommendation


class GeometryLearningSystem:
    """
    A complete geometry learning system that integrates all geometry-specific components.
    
    This system provides:
    - Geometry knowledge graph with spatial and visual concepts
    - Personal learning graphs with geometry-specific tracking
    - Gap analysis specialized for geometry learning patterns
    - Exercise recommendations considering visual and spatial learning needs
    """
    
    def __init__(self):
        """Initialize the geometry learning system."""
        # Initialize geometry-specific components
        self.knowledge_graph = GeometryKnowledgeGraph()
        self.exercise_bank = self._build_geometry_exercise_bank()
        self.gap_analyzer = GeometryGapAnalyzer(self.knowledge_graph)
        self.recommender = GeometryRecommender(self.knowledge_graph, self.exercise_bank)
        
        # Initialize the knowledge graph with geometry concepts
        self.concept_map = self.knowledge_graph.initialize_geometry_concepts()
        
    def _build_geometry_exercise_bank(self) -> ExerciseBank:
        """
        Build a geometry exercise bank from the enhanced geometry exercises data.
        
        Returns:
            ExerciseBank: The geometry exercise bank
        """
        exercise_bank = ExerciseBank(name="Geometry Exercise Bank")
        
        # Load geometry exercises from data file
        data_path = Path(__file__).parent / "data" / "geometry_exercises_enhanced_complete.json"
        
        if data_path.exists():
            with open(data_path, 'r') as f:
                exercises_data = json.load(f)
            
            # Convert to Exercise objects and add to bank
            for exercise_data in exercises_data:
                exercise = Exercise(
                    exercise_id=exercise_data.get('id', ''),
                    title=exercise_data.get('title', ''),
                    problem=exercise_data.get('problem', ''),
                    solution=exercise_data.get('answer', ''),
                    difficulty=exercise_data.get('difficulty', 3),
                    format_type=exercise_data.get('format_type', 'calculation'),
                    hints=exercise_data.get('hints', []),
                    estimated_time=exercise_data.get('time_estimate', 5)
                )
                
                # Link exercise to geometry concepts
                for concept in exercise_data.get('concepts', []):
                    concept_id = f"geom_{concept.lower()}"
                    exercise.add_concept_relationship(concept_id, 0.8)
                
                exercise_bank.add_exercise(exercise)
        
        return exercise_bank
    
    def create_student_learning_graph(self, student_id: str, student_name: str = "") -> GeometryLearningGraph:
        """
        Create a new geometry learning graph for a student.
        
        Args:
            student_id: Unique identifier for the student
            student_name: Optional display name for the student
            
        Returns:
            GeometryLearningGraph: The student's learning graph
        """
        return GeometryLearningGraph(
            user_id=student_id,
            name=student_name or f"Geometry Student {student_id}"
        )
    
    def analyze_student_geometry_gaps(self, learning_graph: GeometryLearningGraph) -> Dict[str, Any]:
        """
        Perform comprehensive geometry gap analysis for a student.
        
        Args:
            learning_graph: The student's learning graph
            
        Returns:
            Dictionary with comprehensive gap analysis
        """
        # Get geometry-specific gaps
        geometry_gaps = self.gap_analyzer.detect_geometry_gaps(learning_graph)
        
        # Get learning boundary
        boundary_concepts = self.gap_analyzer.find_geometry_learning_boundary(learning_graph)
        
        # Get weakness patterns
        weakness_patterns = self.gap_analyzer.get_geometry_weakness_patterns(learning_graph)
        
        # Get learning strategies
        strategies = self.gap_analyzer.recommend_geometry_learning_strategies(learning_graph)
        
        return {
            'geometry_gaps': [gap.__dict__ for gap in geometry_gaps],
            'learning_boundary': boundary_concepts,
            'weakness_patterns': weakness_patterns,
            'recommended_strategies': strategies,
            'gap_summary': {
                'total_gaps': len(geometry_gaps),
                'visual_gaps': len(weakness_patterns['visual_learning_gaps']),
                'spatial_gaps': len(weakness_patterns['spatial_reasoning_gaps']),
                'construction_gaps': len(weakness_patterns['construction_skill_gaps'])
            }
        }
    
    def get_geometry_exercise_recommendations(self, learning_graph: GeometryLearningGraph,
                                            max_exercises: int = 5,
                                            focus_area: Optional[str] = None,
                                            learning_style: Optional[str] = None) -> List[GeometryExerciseRecommendation]:
        """
        Get personalized geometry exercise recommendations.
        
        Args:
            learning_graph: The student's learning graph
            max_exercises: Maximum number of exercises to recommend
            focus_area: Optional focus area (shapes, measurement, etc.)
            learning_style: Optional learning style preference
            
        Returns:
            List of geometry exercise recommendations
        """
        if learning_style:
            return self.recommender.recommend_by_learning_style(learning_graph, learning_style)
        else:
            return self.recommender.recommend_geometry_exercises(
                learning_graph, max_exercises, focus_area
            )
    
    def create_geometry_learning_path(self, learning_graph: GeometryLearningGraph,
                                    target_concept: str) -> List[GeometryExerciseRecommendation]:
        """
        Create a complete learning path for a geometry concept.
        
        Args:
            learning_graph: The student's learning graph
            target_concept: Target concept name to learn
            
        Returns:
            List of recommendations forming a learning path
        """
        return self.recommender.get_geometry_learning_path(learning_graph, target_concept)
    
    def record_geometry_exercise_attempt(self, learning_graph: GeometryLearningGraph,
                                       exercise_id: str, concept_id: str, result: bool,
                                       exercise_type: str = "calculation",
                                       visual_aids_used: bool = False,
                                       construction_involved: bool = False) -> None:
        """
        Record a geometry exercise attempt with geometry-specific tracking.
        
        Args:
            learning_graph: The student's learning graph
            exercise_id: The ID of the exercise
            concept_id: The ID of the geometry concept
            result: Whether the exercise was completed correctly
            exercise_type: Type of geometry exercise
            visual_aids_used: Whether visual aids were used
            construction_involved: Whether construction was involved
        """
        learning_graph.record_geometry_exercise(
            exercise_id, concept_id, result, exercise_type, visual_aids_used, construction_involved
        )
    
    def get_student_geometry_insights(self, learning_graph: GeometryLearningGraph) -> Dict[str, Any]:
        """
        Generate comprehensive geometry learning insights for a student.
        
        Args:
            learning_graph: The student's learning graph
            
        Returns:
            Dictionary with detailed geometry learning insights
        """
        # Get geometry-specific insights
        geometry_insights = learning_graph.get_geometry_learning_insights()
        
        # Get performance analysis
        performance_by_type = learning_graph.get_geometry_performance_by_type()
        visual_effectiveness = learning_graph.get_visual_learning_effectiveness()
        construction_performance = learning_graph.get_construction_performance()
        
        # Get mastery by grade level
        grade_mastery = learning_graph.get_geometry_mastery_by_grade_level()
        
        # Get knowledge graph statistics
        kg_stats = self.knowledge_graph.get_geometry_statistics()
        
        return {
            'geometry_insights': geometry_insights,
            'performance_analysis': {
                'by_exercise_type': performance_by_type,
                'visual_learning_effectiveness': visual_effectiveness,
                'construction_performance': construction_performance
            },
            'mastery_by_grade_level': grade_mastery,
            'knowledge_graph_coverage': kg_stats,
            'learning_recommendations': self._generate_personalized_recommendations(learning_graph)
        }
    
    def _generate_personalized_recommendations(self, learning_graph: GeometryLearningGraph) -> List[str]:
        """Generate personalized learning recommendations."""
        recommendations = []
        
        # Analyze visual learning effectiveness
        visual_effectiveness = learning_graph.get_visual_learning_effectiveness()
        if visual_effectiveness['visual_aid_effectiveness'] > 0.2:
            recommendations.append("Visual learning is working well - continue using diagrams and visual aids")
        elif visual_effectiveness['visual_aid_effectiveness'] < -0.1:
            recommendations.append("Try different visual approaches - current visual aids may not be optimal")
        
        # Analyze construction performance
        construction_perf = learning_graph.get_construction_performance()
        if construction_perf['construction_success_rate'] > 0.7:
            recommendations.append("Excellent construction skills - try more advanced geometric constructions")
        elif construction_perf['construction_success_rate'] < 0.4:
            recommendations.append("Practice basic constructions with compass and straightedge")
        
        # Analyze spatial reasoning
        spatial_score = learning_graph.geometry_specific_data.get('spatial_reasoning_score', 0.5)
        if spatial_score < 0.4:
            recommendations.append("Focus on developing spatial reasoning with 3D models and rotation exercises")
        elif spatial_score > 0.8:
            recommendations.append("Strong spatial reasoning - ready for advanced 3D geometry concepts")
        
        return recommendations
    
    def get_geometry_concept_details(self, concept_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a geometry concept.
        
        Args:
            concept_name: Name of the geometry concept
            
        Returns:
            Dictionary with concept details
        """
        # Find concept in knowledge graph
        concept = None
        for c in self.knowledge_graph.get_all_concepts():
            if concept_name.lower() in c.name.lower():
                concept = c
                break
        
        if not concept:
            return {'error': f'Concept "{concept_name}" not found'}
        
        # Get related exercises
        related_exercises = []
        for exercise in self.exercise_bank.exercises:
            if concept.id in exercise.get_all_concept_ids():
                related_exercises.append({
                    'id': exercise.id,
                    'title': exercise.title,
                    'difficulty': exercise.difficulty,
                    'format_type': exercise.format_type
                })
        
        # Get prerequisite and dependent concepts
        prerequisites = self.knowledge_graph.get_prerequisites(concept.id)
        dependents = self.knowledge_graph.get_dependent_concepts(concept.id)
        
        return {
            'concept_id': concept.id,
            'name': concept.name,
            'description': concept.description,
            'difficulty': concept.difficulty,
            'time_to_master': concept.time_to_master,
            'category': concept.category,
            'grade_level': getattr(concept, 'grade_level', 'Unknown'),
            'prerequisites': prerequisites,
            'dependent_concepts': dependents,
            'related_exercises': related_exercises,
            'total_exercises': len(related_exercises)
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the geometry learning system.
        
        Returns:
            Dictionary with system statistics
        """
        kg_stats = self.knowledge_graph.get_geometry_statistics()
        
        # Exercise bank statistics
        exercise_stats = {
            'total_exercises': len(self.exercise_bank.exercises),
            'difficulty_distribution': {},
            'format_type_distribution': {},
            'average_difficulty': 0.0
        }
        
        if self.exercise_bank.exercises:
            difficulties = [ex.difficulty for ex in self.exercise_bank.exercises]
            exercise_stats['average_difficulty'] = sum(difficulties) / len(difficulties)
            
            for ex in self.exercise_bank.exercises:
                diff = ex.difficulty
                exercise_stats['difficulty_distribution'][diff] = exercise_stats['difficulty_distribution'].get(diff, 0) + 1
                
                fmt = ex.format_type
                exercise_stats['format_type_distribution'][fmt] = exercise_stats['format_type_distribution'].get(fmt, 0) + 1
        
        return {
            'knowledge_graph_stats': kg_stats,
            'exercise_bank_stats': exercise_stats,
            'system_capabilities': [
                'Geometry knowledge graph with prerequisite relationships',
                'Personal learning graphs with geometry-specific tracking',
                'Visual and spatial learning preference analysis',
                'Construction skill assessment',
                'Geometry-specific gap analysis',
                'Learning style-based exercise recommendations',
                'Personalized learning path generation'
            ]
        } 