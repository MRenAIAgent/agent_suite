"""
Geometry Gap Analyzer module.

This module defines a geometry-specific gap analyzer that inherits from
the base GapAnalyzer and adds geometry-specific gap detection methods.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

from ..recommendation.gap_analyzer import GapAnalyzer, GapInfo
from .geometry_knowledge_graph import GeometryKnowledgeGraph
from .geometry_learning_graph import GeometryLearningGraph


class GeometryGapInfo(GapInfo):
    """Extended gap information for geometry concepts."""
    
    def __init__(self, concept_id: str, mastery: float, impact_score: float,
                 centrality: float, priority: float, visual_component_missing: bool = False,
                 spatial_reasoning_required: bool = False, construction_skills_needed: bool = False,
                 grade_level: str = "", geometry_category: str = ""):
        """
        Initialize geometry gap information.
        
        Args:
            concept_id: ID of the concept
            mastery: Current mastery level
            impact_score: How many other concepts this unlocks
            centrality: Centrality in the knowledge graph
            priority: Overall priority score
            visual_component_missing: Whether visual component is missing
            spatial_reasoning_required: Whether spatial reasoning is required
            construction_skills_needed: Whether construction skills are needed
            grade_level: Grade level of the concept
            geometry_category: Category of geometry concept
        """
        super().__init__(concept_id, mastery, impact_score, centrality, priority)
        self.visual_component_missing = visual_component_missing
        self.spatial_reasoning_required = spatial_reasoning_required
        self.construction_skills_needed = construction_skills_needed
        self.grade_level = grade_level
        self.geometry_category = geometry_category


class GeometryGapAnalyzer(GapAnalyzer):
    """
    Geometry-specific gap analyzer that extends the base GapAnalyzer
    with geometry domain knowledge and specialized gap detection.
    """
    
    def __init__(self, geometry_knowledge_graph: GeometryKnowledgeGraph):
        """
        Initialize the geometry gap analyzer.
        
        Args:
            geometry_knowledge_graph: The geometry knowledge graph
        """
        super().__init__(geometry_knowledge_graph)
        self.geometry_kg = geometry_knowledge_graph
    
    def detect_geometry_gaps(self, learning_graph: GeometryLearningGraph, 
                           mastery_threshold: float = 0.7) -> List[GeometryGapInfo]:
        """
        Detect geometry-specific knowledge gaps.
        
        Args:
            learning_graph: The student's geometry learning graph
            mastery_threshold: Threshold below which a concept is considered not mastered
            
        Returns:
            List of geometry-specific gap information objects
        """
        # Get basic gaps first
        basic_gaps = super().detect_critical_gaps(learning_graph, mastery_threshold)
        
        geometry_gaps = []
        
        for gap in basic_gaps:
            # Get concept details
            concept = self.geometry_kg.get_concept(gap.concept_id)
            if not concept:
                continue
            
            # Analyze geometry-specific aspects
            visual_component = self._requires_visual_component(concept.name)
            spatial_reasoning = self._requires_spatial_reasoning(concept.name)
            construction_skills = self._requires_construction_skills(concept.name)
            grade_level = getattr(concept, 'grade_level', 'Unknown')
            geometry_category = self._categorize_geometry_concept(concept.name)
            
            # Create enhanced gap info
            geometry_gap = GeometryGapInfo(
                concept_id=gap.concept_id,
                mastery=gap.mastery,
                impact_score=gap.impact_score,
                centrality=gap.centrality,
                priority=gap.priority,
                visual_component_missing=visual_component,
                spatial_reasoning_required=spatial_reasoning,
                construction_skills_needed=construction_skills,
                grade_level=grade_level,
                geometry_category=geometry_category
            )
            
            geometry_gaps.append(geometry_gap)
        
        # Sort by geometry-specific priority
        return sorted(geometry_gaps, key=self._calculate_geometry_priority, reverse=True)
    
    def _requires_visual_component(self, concept_name: str) -> bool:
        """Determine if a concept requires strong visual understanding."""
        visual_concepts = [
            'shapes', 'polygons', 'circles', 'symmetry', 'transformations',
            'coordinate', 'construction', 'diagrams', 'spatial'
        ]
        return any(visual in concept_name.lower() for visual in visual_concepts)
    
    def _requires_spatial_reasoning(self, concept_name: str) -> bool:
        """Determine if a concept requires spatial reasoning skills."""
        spatial_concepts = [
            'volume', 'surface area', '3d', 'rotation', 'reflection',
            'coordinate geometry', 'transformations', 'perspective', 'orientation'
        ]
        return any(spatial in concept_name.lower() for spatial in spatial_concepts)
    
    def _requires_construction_skills(self, concept_name: str) -> bool:
        """Determine if a concept involves geometric construction."""
        construction_concepts = [
            'construction', 'compass', 'straightedge', 'bisector',
            'perpendicular', 'parallel', 'inscribed', 'circumscribed'
        ]
        return any(construction in concept_name.lower() for construction in construction_concepts)
    
    def _categorize_geometry_concept(self, concept_name: str) -> str:
        """Categorize geometry concept into main areas."""
        concept_lower = concept_name.lower()
        
        if any(word in concept_lower for word in ['shape', 'polygon', 'triangle', 'circle', 'square']):
            return 'shapes_and_figures'
        elif any(word in concept_lower for word in ['area', 'perimeter', 'volume', 'surface']):
            return 'measurement'
        elif any(word in concept_lower for word in ['angle', 'degree', 'acute', 'obtuse']):
            return 'angles'
        elif any(word in concept_lower for word in ['coordinate', 'graph', 'plot']):
            return 'coordinate_geometry'
        elif any(word in concept_lower for word in ['transform', 'rotation', 'reflection', 'translation']):
            return 'transformations'
        elif any(word in concept_lower for word in ['proof', 'theorem', 'postulate']):
            return 'proofs_and_reasoning'
        elif any(word in concept_lower for word in ['construction', 'compass', 'straightedge']):
            return 'constructions'
        else:
            return 'general_geometry'
    
    def _calculate_geometry_priority(self, gap: GeometryGapInfo) -> float:
        """Calculate geometry-specific priority for a gap."""
        base_priority = gap.priority
        
        # Boost priority for foundational visual concepts
        if gap.visual_component_missing and gap.grade_level in ['K-5', '6-8']:
            base_priority += 0.3
        
        # Boost priority for spatial reasoning gaps
        if gap.spatial_reasoning_required and gap.mastery < 0.3:
            base_priority += 0.2
        
        # Category-based priority adjustments
        category_boosts = {
            'shapes_and_figures': 0.4,  # Foundational
            'angles': 0.3,              # Essential for many concepts
            'measurement': 0.2,         # Practical importance
            'coordinate_geometry': 0.1,
            'transformations': 0.1,
            'constructions': 0.05,
            'proofs_and_reasoning': 0.0  # Advanced, lower priority for gaps
        }
        
        base_priority += category_boosts.get(gap.geometry_category, 0.0)
        
        return base_priority
    
    def find_geometry_learning_boundary(self, learning_graph: GeometryLearningGraph) -> List[str]:
        """
        Find concepts at the geometry learning boundary, considering visual and spatial prerequisites.
        
        Args:
            learning_graph: The student's geometry learning graph
            
        Returns:
            List of concept IDs ready to learn
        """
        # Get basic learning boundary
        basic_boundary = super().find_learning_boundary(learning_graph)
        
        # Filter for geometry-appropriate concepts
        geometry_boundary = []
        
        for concept_id in basic_boundary:
            concept = self.geometry_kg.get_concept(concept_id)
            if not concept:
                continue
            
            # Check if student has sufficient visual/spatial foundation
            if self._has_sufficient_foundation(concept, learning_graph):
                geometry_boundary.append(concept_id)
        
        return geometry_boundary
    
    def _has_sufficient_foundation(self, concept, learning_graph: GeometryLearningGraph) -> bool:
        """Check if student has sufficient foundation for a geometry concept."""
        # Check visual learning readiness
        if self._requires_visual_component(concept.name):
            visual_preference = learning_graph.geometry_specific_data.get('visual_learning_preference', 0.5)
            if visual_preference < 0.3:
                return False
        
        # Check spatial reasoning readiness
        if self._requires_spatial_reasoning(concept.name):
            spatial_score = learning_graph.geometry_specific_data.get('spatial_reasoning_score', 0.5)
            if spatial_score < 0.4:
                return False
        
        return True
    
    def get_geometry_weakness_patterns(self, learning_graph: GeometryLearningGraph) -> Dict[str, Any]:
        """
        Identify patterns in geometry learning weaknesses.
        
        Args:
            learning_graph: The student's geometry learning graph
            
        Returns:
            Dictionary with weakness pattern analysis
        """
        patterns = {
            'visual_learning_gaps': [],
            'spatial_reasoning_gaps': [],
            'construction_skill_gaps': [],
            'category_weaknesses': {},
            'grade_level_gaps': {}
        }
        
        gaps = self.detect_geometry_gaps(learning_graph)
        
        for gap in gaps:
            # Categorize gaps by type
            if gap.visual_component_missing:
                patterns['visual_learning_gaps'].append(gap.concept_id)
            
            if gap.spatial_reasoning_required:
                patterns['spatial_reasoning_gaps'].append(gap.concept_id)
            
            if gap.construction_skills_needed:
                patterns['construction_skill_gaps'].append(gap.concept_id)
            
            # Track by category
            category = gap.geometry_category
            if category not in patterns['category_weaknesses']:
                patterns['category_weaknesses'][category] = []
            patterns['category_weaknesses'][category].append(gap.concept_id)
            
            # Track by grade level
            grade = gap.grade_level
            if grade not in patterns['grade_level_gaps']:
                patterns['grade_level_gaps'][grade] = []
            patterns['grade_level_gaps'][grade].append(gap.concept_id)
        
        return patterns
    
    def recommend_geometry_learning_strategies(self, learning_graph: GeometryLearningGraph) -> List[str]:
        """
        Recommend geometry-specific learning strategies based on gaps and performance.
        
        Args:
            learning_graph: The student's geometry learning graph
            
        Returns:
            List of recommended learning strategies
        """
        strategies = []
        
        # Analyze performance patterns
        visual_effectiveness = learning_graph.get_visual_learning_effectiveness()
        construction_performance = learning_graph.get_construction_performance()
        weakness_patterns = self.get_geometry_weakness_patterns(learning_graph)
        
        # Visual learning strategies
        if visual_effectiveness['visual_aid_effectiveness'] > 0.1:
            strategies.append("Continue using visual aids and diagrams - they're helping!")
        elif len(weakness_patterns['visual_learning_gaps']) > 2:
            strategies.append("Try more visual approaches: use geometric software, physical models, or drawing")
        
        # Spatial reasoning strategies
        if len(weakness_patterns['spatial_reasoning_gaps']) > 1:
            strategies.append("Practice spatial reasoning with 3D models and rotation exercises")
        
        # Construction strategies
        if construction_performance['construction_success_rate'] < 0.5:
            strategies.append("Practice geometric constructions with compass and straightedge")
        
        # Category-specific strategies
        category_counts = {cat: len(gaps) for cat, gaps in weakness_patterns['category_weaknesses'].items()}
        
        if category_counts.get('shapes_and_figures', 0) > 2:
            strategies.append("Focus on basic shape identification and properties")
        
        if category_counts.get('measurement', 0) > 2:
            strategies.append("Practice area, perimeter, and volume calculations with real objects")
        
        if category_counts.get('angles', 0) > 1:
            strategies.append("Use protractors and angle manipulatives to understand angle relationships")
        
        return strategies[:5]  # Limit to top 5 strategies 