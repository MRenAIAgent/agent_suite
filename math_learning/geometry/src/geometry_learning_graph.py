"""
Geometry Learning Graph module.

This module defines a geometry-specific learning graph that inherits from
the base LearningGraph and adds geometry-specific tracking and analysis.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..learning_graph.user_model import LearningGraph


class GeometryLearningGraph(LearningGraph):
    """
    Geometry-specific learning graph that extends the base LearningGraph
    with geometry-specific progress tracking and analysis methods.
    """
    
    def __init__(self, user_id: str = "", name: str = ""):
        """
        Initialize the geometry learning graph.
        
        Args:
            user_id: Identifier for the user
            name: Optional name for the learning graph
        """
        super().__init__(user_id, name)
        self.geometry_specific_data = {
            'visual_learning_preference': 0.5,  # 0.0-1.0 scale
            'spatial_reasoning_score': 0.5,
            'construction_tool_familiarity': {},
            'geometry_learning_style': 'balanced'  # visual, analytical, kinesthetic, balanced
        }
    
    def record_geometry_exercise(self, exercise_id: str, concept_id: str, result: bool,
                               exercise_type: str = "calculation", 
                               visual_aids_used: bool = False,
                               construction_involved: bool = False) -> None:
        """
        Record a geometry exercise attempt with geometry-specific metadata.
        
        Args:
            exercise_id: The ID of the exercise
            concept_id: The ID of the geometry concept being tested
            result: Whether the exercise was completed correctly
            exercise_type: Type of geometry exercise (calculation, construction, proof, etc.)
            visual_aids_used: Whether visual aids were used
            construction_involved: Whether geometric construction was involved
        """
        # Call parent method first
        self.record_exercise_attempt(exercise_id, concept_id, result)
        
        # Add geometry-specific metadata to the latest attempt
        if exercise_id in self.exercise_history and self.exercise_history[exercise_id]:
            latest_attempt = self.exercise_history[exercise_id][-1]
            latest_attempt.update({
                'exercise_type': exercise_type,
                'visual_aids_used': visual_aids_used,
                'construction_involved': construction_involved,
                'geometry_metadata': True
            })
    
    def update_spatial_reasoning_score(self, new_score: float) -> None:
        """
        Update the spatial reasoning score based on performance.
        
        Args:
            new_score: New spatial reasoning score (0.0-1.0)
        """
        self.geometry_specific_data['spatial_reasoning_score'] = max(0.0, min(1.0, new_score))
    
    def update_visual_learning_preference(self, preference: float) -> None:
        """
        Update visual learning preference based on performance with visual aids.
        
        Args:
            preference: Visual learning preference (0.0-1.0)
        """
        self.geometry_specific_data['visual_learning_preference'] = max(0.0, min(1.0, preference))
    
    def get_geometry_performance_by_type(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by geometry exercise type.
        
        Returns:
            Dictionary with performance statistics by exercise type
        """
        type_performance = {}
        
        for exercise_id, attempts in self.exercise_history.items():
            for attempt in attempts:
                if attempt.get('geometry_metadata', False):
                    exercise_type = attempt.get('exercise_type', 'unknown')
                    
                    if exercise_type not in type_performance:
                        type_performance[exercise_type] = {
                            'total_attempts': 0,
                            'successful_attempts': 0,
                            'success_rate': 0.0,
                            'with_visual_aids': 0,
                            'with_construction': 0
                        }
                    
                    stats = type_performance[exercise_type]
                    stats['total_attempts'] += 1
                    
                    if attempt['result']:
                        stats['successful_attempts'] += 1
                    
                    if attempt.get('visual_aids_used', False):
                        stats['with_visual_aids'] += 1
                    
                    if attempt.get('construction_involved', False):
                        stats['with_construction'] += 1
        
        # Calculate success rates
        for exercise_type, stats in type_performance.items():
            if stats['total_attempts'] > 0:
                stats['success_rate'] = stats['successful_attempts'] / stats['total_attempts']
        
        return type_performance
    
    def get_visual_learning_effectiveness(self) -> Dict[str, float]:
        """
        Analyze effectiveness of visual aids in learning.
        
        Returns:
            Dictionary with visual learning statistics
        """
        with_visual_aids = {'attempts': 0, 'successes': 0}
        without_visual_aids = {'attempts': 0, 'successes': 0}
        
        for exercise_id, attempts in self.exercise_history.items():
            for attempt in attempts:
                if attempt.get('geometry_metadata', False):
                    if attempt.get('visual_aids_used', False):
                        with_visual_aids['attempts'] += 1
                        if attempt['result']:
                            with_visual_aids['successes'] += 1
                    else:
                        without_visual_aids['attempts'] += 1
                        if attempt['result']:
                            without_visual_aids['successes'] += 1
        
        # Calculate success rates
        with_visual_rate = (with_visual_aids['successes'] / with_visual_aids['attempts'] 
                           if with_visual_aids['attempts'] > 0 else 0.0)
        without_visual_rate = (without_visual_aids['successes'] / without_visual_aids['attempts'] 
                              if without_visual_aids['attempts'] > 0 else 0.0)
        
        return {
            'success_rate_with_visual_aids': with_visual_rate,
            'success_rate_without_visual_aids': without_visual_rate,
            'visual_aid_effectiveness': with_visual_rate - without_visual_rate,
            'total_attempts_with_visual': with_visual_aids['attempts'],
            'total_attempts_without_visual': without_visual_aids['attempts']
        }
    
    def get_construction_performance(self) -> Dict[str, Any]:
        """
        Analyze performance on construction-based exercises.
        
        Returns:
            Dictionary with construction performance statistics
        """
        construction_stats = {
            'total_construction_exercises': 0,
            'successful_constructions': 0,
            'construction_success_rate': 0.0,
            'construction_vs_calculation_performance': 0.0
        }
        
        calculation_stats = {'total': 0, 'successes': 0}
        
        for exercise_id, attempts in self.exercise_history.items():
            for attempt in attempts:
                if attempt.get('geometry_metadata', False):
                    if attempt.get('construction_involved', False):
                        construction_stats['total_construction_exercises'] += 1
                        if attempt['result']:
                            construction_stats['successful_constructions'] += 1
                    elif attempt.get('exercise_type') == 'calculation':
                        calculation_stats['total'] += 1
                        if attempt['result']:
                            calculation_stats['successes'] += 1
        
        # Calculate rates
        if construction_stats['total_construction_exercises'] > 0:
            construction_stats['construction_success_rate'] = (
                construction_stats['successful_constructions'] / 
                construction_stats['total_construction_exercises']
            )
        
        # Compare construction vs calculation performance
        calc_rate = (calculation_stats['successes'] / calculation_stats['total'] 
                    if calculation_stats['total'] > 0 else 0.0)
        construction_stats['construction_vs_calculation_performance'] = (
            construction_stats['construction_success_rate'] - calc_rate
        )
        
        return construction_stats
    
    def get_geometry_learning_insights(self) -> Dict[str, Any]:
        """
        Generate geometry-specific learning insights.
        
        Returns:
            Dictionary with personalized geometry learning insights
        """
        performance_by_type = self.get_geometry_performance_by_type()
        visual_effectiveness = self.get_visual_learning_effectiveness()
        construction_performance = self.get_construction_performance()
        
        insights = {
            'strongest_geometry_areas': [],
            'geometry_learning_style_recommendation': self.geometry_specific_data['geometry_learning_style'],
            'visual_learner_score': self.geometry_specific_data['visual_learning_preference'],
            'spatial_reasoning_level': self.geometry_specific_data['spatial_reasoning_score'],
            'recommended_learning_approaches': []
        }
        
        # Identify strongest areas
        for exercise_type, stats in performance_by_type.items():
            if stats['success_rate'] > 0.7:
                insights['strongest_geometry_areas'].append(exercise_type)
        
        # Recommend learning approaches
        if visual_effectiveness['visual_aid_effectiveness'] > 0.1:
            insights['recommended_learning_approaches'].append('Use visual aids and diagrams')
        
        if construction_performance['construction_success_rate'] > 0.6:
            insights['recommended_learning_approaches'].append('Hands-on geometric construction')
        
        if self.geometry_specific_data['spatial_reasoning_score'] < 0.5:
            insights['recommended_learning_approaches'].append('Focus on spatial reasoning exercises')
        
        return insights
    
    def get_geometry_mastery_by_grade_level(self) -> Dict[str, float]:
        """
        Get mastery levels organized by typical geometry grade levels.
        
        Returns:
            Dictionary mapping grade levels to average mastery
        """
        # This would need to be integrated with the geometry knowledge graph
        # to properly categorize concepts by grade level
        grade_mastery = {
            'K-5': 0.0,
            '6-8': 0.0, 
            '9-12': 0.0
        }
        
        # For now, return current implementation
        # In a full implementation, this would categorize concepts by grade level
        all_mastery = [data['mastery'] for data in self.concept_mastery.values()]
        avg_mastery = sum(all_mastery) / len(all_mastery) if all_mastery else 0.0
        
        # Distribute evenly for demo purposes
        for grade in grade_mastery:
            grade_mastery[grade] = avg_mastery
        
        return grade_mastery 