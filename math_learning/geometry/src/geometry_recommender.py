"""
Geometry Recommender module.

This module defines a geometry-specific recommender that inherits from
the base Recommender and adds geometry-specific exercise recommendation logic.
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..recommendation.recommender import Recommender, ExerciseRecommendation
from ..exercises.exercise import Exercise
from ..exercises.exercise_bank import ExerciseBank
from .geometry_knowledge_graph import GeometryKnowledgeGraph
from .geometry_learning_graph import GeometryLearningGraph
from .geometry_gap_analyzer import GeometryGapAnalyzer


class GeometryExerciseRecommendation(ExerciseRecommendation):
    """Extended exercise recommendation for geometry with additional metadata."""
    
    def __init__(self, exercise: Exercise, concept_id: str, reason: str,
                 requires_visual_aids: bool = False, requires_construction: bool = False,
                 spatial_reasoning_level: str = "basic", geometry_category: str = ""):
        """
        Initialize geometry exercise recommendation.
        
        Args:
            exercise: The recommended exercise
            concept_id: ID of the concept being targeted
            reason: Explanation for why this exercise was recommended
            requires_visual_aids: Whether visual aids are recommended
            requires_construction: Whether construction tools are needed
            spatial_reasoning_level: Level of spatial reasoning required
            geometry_category: Category of geometry concept
        """
        super().__init__(exercise, concept_id, reason)
        self.requires_visual_aids = requires_visual_aids
        self.requires_construction = requires_construction
        self.spatial_reasoning_level = spatial_reasoning_level
        self.geometry_category = geometry_category
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "requires_visual_aids": self.requires_visual_aids,
            "requires_construction": self.requires_construction,
            "spatial_reasoning_level": self.spatial_reasoning_level,
            "geometry_category": self.geometry_category
        })
        return base_dict


class GeometryRecommender(Recommender):
    """
    Geometry-specific recommender that extends the base Recommender
    with geometry domain knowledge and specialized recommendation logic.
    """
    
    def __init__(self, geometry_knowledge_graph: GeometryKnowledgeGraph, 
                 geometry_exercise_bank: ExerciseBank):
        """
        Initialize the geometry recommender.
        
        Args:
            geometry_knowledge_graph: The geometry knowledge graph
            geometry_exercise_bank: The geometry exercise bank
        """
        super().__init__(geometry_knowledge_graph, geometry_exercise_bank)
        self.geometry_kg = geometry_knowledge_graph
        self.geometry_gap_analyzer = GeometryGapAnalyzer(geometry_knowledge_graph)
        self.geometry_data_path = Path(__file__).parent / "data"
    
    def recommend_geometry_exercises(self, learning_graph: GeometryLearningGraph,
                                   max_exercises: int = 5,
                                   focus_area: Optional[str] = None) -> List[GeometryExerciseRecommendation]:
        """
        Recommend geometry exercises with geometry-specific considerations.
        
        Args:
            learning_graph: The student's geometry learning graph
            max_exercises: Maximum number of exercises to recommend
            focus_area: Optional focus area (shapes, measurement, etc.)
            
        Returns:
            List of geometry exercise recommendations
        """
        # Get geometry-specific gaps
        geometry_gaps = self.geometry_gap_analyzer.detect_geometry_gaps(learning_graph)
        
        # Get learning boundary
        boundary_concepts = self.geometry_gap_analyzer.find_geometry_learning_boundary(learning_graph)
        
        recommendations = []
        
        # Prioritize concepts based on geometry-specific criteria
        prioritized_concepts = self._prioritize_geometry_concepts(
            geometry_gaps, boundary_concepts, learning_graph, focus_area
        )
        
        # Generate recommendations for each prioritized concept
        for concept_id in prioritized_concepts[:max_exercises]:
            concept = self.geometry_kg.get_concept(concept_id)
            if not concept:
                continue
            
            # Find suitable exercises for this concept
            suitable_exercises = self._find_suitable_geometry_exercises(
                concept_id, learning_graph, concept
            )
            
            if suitable_exercises:
                best_exercise = suitable_exercises[0]  # Take the best match
                
                # Create geometry-specific recommendation
                recommendation = self._create_geometry_recommendation(
                    best_exercise, concept_id, concept, learning_graph
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _prioritize_geometry_concepts(self, gaps: List, boundary_concepts: List[str],
                                    learning_graph: GeometryLearningGraph,
                                    focus_area: Optional[str] = None) -> List[str]:
        """Prioritize concepts for geometry learning."""
        concept_scores = {}
        
        # Score gaps
        for gap in gaps:
            score = gap.priority
            
            # Boost visual concepts if student learns well visually
            if gap.visual_component_missing:
                visual_pref = learning_graph.geometry_specific_data.get('visual_learning_preference', 0.5)
                score += visual_pref * 0.3
            
            # Focus area boost
            if focus_area and gap.geometry_category == focus_area:
                score += 0.5
            
            concept_scores[gap.concept_id] = score
        
        # Score boundary concepts
        for concept_id in boundary_concepts:
            if concept_id not in concept_scores:
                concept_scores[concept_id] = 0.5  # Base score for ready concepts
        
        # Sort by score
        return sorted(concept_scores.keys(), key=lambda x: concept_scores[x], reverse=True)
    
    def _find_suitable_geometry_exercises(self, concept_id: str, 
                                        learning_graph: GeometryLearningGraph,
                                        concept) -> List[Exercise]:
        """Find exercises suitable for a geometry concept and student."""
        # Load geometry exercises
        exercises_file = self.geometry_data_path / "geometry_exercises_enhanced_complete.json"
        
        if not exercises_file.exists():
            return []
        
        with open(exercises_file, 'r') as f:
            exercises_data = json.load(f)
        
        suitable_exercises = []
        mastery = learning_graph.get_mastery(concept_id)
        
        # Filter exercises for this concept
        for exercise_data in exercises_data:
            # Check if exercise is for this concept
            exercise_concepts = exercise_data.get('concepts', [])
            if not any(concept_id.endswith(c.lower()) for c in exercise_concepts):
                continue
            
            # Check difficulty appropriateness
            exercise_difficulty = exercise_data.get('difficulty', 3)
            if self._is_appropriate_difficulty(exercise_difficulty, mastery):
                # Create Exercise object
                exercise = Exercise(
                    exercise_id=exercise_data.get('id', ''),
                    title=exercise_data.get('title', ''),
                    problem=exercise_data.get('problem', ''),
                    solution=exercise_data.get('answer', ''),
                    difficulty=exercise_difficulty,
                    format_type=exercise_data.get('format_type', 'calculation'),
                    hints=exercise_data.get('hints', [])
                )
                
                # Link to concepts
                for concept_name in exercise_concepts:
                    exercise.add_concept_relationship(f"geom_{concept_name.lower()}", 0.8)
                
                suitable_exercises.append(exercise)
        
        # Sort by appropriateness
        return sorted(suitable_exercises, key=lambda e: self._calculate_exercise_score(e, learning_graph))
    
    def _is_appropriate_difficulty(self, exercise_difficulty: int, mastery: float) -> bool:
        """Check if exercise difficulty is appropriate for student mastery level."""
        if mastery < 0.3:  # Struggling student
            return exercise_difficulty <= 2
        elif mastery < 0.7:  # Developing student
            return exercise_difficulty <= 3
        else:  # Advanced student
            return exercise_difficulty <= 4
    
    def _calculate_exercise_score(self, exercise: Exercise, learning_graph: GeometryLearningGraph) -> float:
        """Calculate appropriateness score for an exercise."""
        score = 0.0
        
        # Prefer exercises matching student's learning style
        visual_pref = learning_graph.geometry_specific_data.get('visual_learning_preference', 0.5)
        
        # Boost visual exercises for visual learners
        if exercise.format_type in ['visual-identification', 'construction', 'diagram']:
            score += visual_pref * 0.5
        
        # Boost construction exercises if student does well with them
        construction_perf = learning_graph.get_construction_performance()
        if (exercise.format_type == 'construction' and 
            construction_perf['construction_success_rate'] > 0.6):
            score += 0.3
        
        return score
    
    def _create_geometry_recommendation(self, exercise: Exercise, concept_id: str,
                                      concept, learning_graph: GeometryLearningGraph) -> GeometryExerciseRecommendation:
        """Create a geometry-specific exercise recommendation."""
        # Determine geometry-specific attributes
        requires_visual = self._exercise_requires_visual_aids(exercise)
        requires_construction = self._exercise_requires_construction(exercise)
        spatial_level = self._determine_spatial_reasoning_level(exercise)
        geometry_category = self._categorize_exercise(exercise)
        
        # Generate reason
        mastery = learning_graph.get_mastery(concept_id)
        if mastery < 0.3:
            reason = f"Building foundational understanding of {concept.name}"
        elif mastery < 0.7:
            reason = f"Strengthening skills in {concept.name}"
        else:
            reason = f"Advanced practice with {concept.name}"
        
        return GeometryExerciseRecommendation(
            exercise=exercise,
            concept_id=concept_id,
            reason=reason,
            requires_visual_aids=requires_visual,
            requires_construction=requires_construction,
            spatial_reasoning_level=spatial_level,
            geometry_category=geometry_category
        )
    
    def _exercise_requires_visual_aids(self, exercise: Exercise) -> bool:
        """Determine if exercise would benefit from visual aids."""
        visual_indicators = ['diagram', 'shape', 'figure', 'draw', 'construct', 'visualize']
        exercise_text = (exercise.problem + ' ' + exercise.title).lower()
        return any(indicator in exercise_text for indicator in visual_indicators)
    
    def _exercise_requires_construction(self, exercise: Exercise) -> bool:
        """Determine if exercise requires construction tools."""
        construction_indicators = ['construct', 'draw', 'compass', 'straightedge', 'protractor']
        exercise_text = (exercise.problem + ' ' + exercise.title).lower()
        return any(indicator in exercise_text for indicator in construction_indicators)
    
    def _determine_spatial_reasoning_level(self, exercise: Exercise) -> str:
        """Determine spatial reasoning level required."""
        exercise_text = (exercise.problem + ' ' + exercise.title).lower()
        
        if any(word in exercise_text for word in ['3d', 'volume', 'surface', 'rotation', 'perspective']):
            return "advanced"
        elif any(word in exercise_text for word in ['coordinate', 'transform', 'reflection', 'translation']):
            return "intermediate"
        else:
            return "basic"
    
    def _categorize_exercise(self, exercise: Exercise) -> str:
        """Categorize exercise by geometry area."""
        exercise_text = (exercise.problem + ' ' + exercise.title).lower()
        
        if any(word in exercise_text for word in ['triangle', 'square', 'circle', 'polygon']):
            return "shapes_and_figures"
        elif any(word in exercise_text for word in ['area', 'perimeter', 'volume', 'surface']):
            return "measurement"
        elif any(word in exercise_text for word in ['angle', 'degree', 'acute', 'obtuse']):
            return "angles"
        elif any(word in exercise_text for word in ['coordinate', 'graph', 'plot']):
            return "coordinate_geometry"
        else:
            return "general_geometry"
    
    def get_geometry_learning_path(self, learning_graph: GeometryLearningGraph,
                                 target_concept: str) -> List[GeometryExerciseRecommendation]:
        """
        Generate a complete learning path for a geometry concept.
        
        Args:
            learning_graph: The student's learning graph
            target_concept: Target concept name
            
        Returns:
            List of recommendations forming a learning path
        """
        # Get learning path from knowledge graph
        concept_path = self.geometry_kg.get_geometry_learning_path(target_concept)
        
        path_recommendations = []
        
        for concept in concept_path:
            # Get recommendations for this concept
            concept_recs = self.recommend_geometry_exercises(
                learning_graph, max_exercises=2, focus_area=None
            )
            
            # Filter for this specific concept
            for rec in concept_recs:
                if rec.concept_id == concept.id:
                    path_recommendations.append(rec)
                    break
        
        return path_recommendations
    
    def recommend_by_learning_style(self, learning_graph: GeometryLearningGraph,
                                  learning_style: str = "balanced") -> List[GeometryExerciseRecommendation]:
        """
        Recommend exercises based on specific learning style.
        
        Args:
            learning_graph: The student's learning graph
            learning_style: Learning style preference (visual, kinesthetic, analytical, balanced)
            
        Returns:
            List of style-appropriate recommendations
        """
        # Get base recommendations
        base_recs = self.recommend_geometry_exercises(learning_graph, max_exercises=10)
        
        # Filter and score by learning style
        style_appropriate = []
        
        for rec in base_recs:
            style_score = self._calculate_style_appropriateness(rec, learning_style)
            if style_score > 0.5:  # Only include appropriate exercises
                style_appropriate.append(rec)
        
        # Sort by style appropriateness
        style_appropriate.sort(
            key=lambda r: self._calculate_style_appropriateness(r, learning_style),
            reverse=True
        )
        
        return style_appropriate[:5]
    
    def _calculate_style_appropriateness(self, recommendation: GeometryExerciseRecommendation,
                                       learning_style: str) -> float:
        """Calculate how well a recommendation matches a learning style."""
        score = 0.5  # Base score
        
        if learning_style == "visual":
            if recommendation.requires_visual_aids:
                score += 0.4
            if recommendation.exercise.format_type in ['visual-identification', 'diagram']:
                score += 0.3
        
        elif learning_style == "kinesthetic":
            if recommendation.requires_construction:
                score += 0.4
            if recommendation.exercise.format_type == 'construction':
                score += 0.3
        
        elif learning_style == "analytical":
            if recommendation.exercise.format_type in ['calculation', 'proof']:
                score += 0.4
            if recommendation.spatial_reasoning_level == "advanced":
                score += 0.2
        
        elif learning_style == "balanced":
            # Balanced learners get moderate boost for all types
            score += 0.2
        
        return min(1.0, score) 