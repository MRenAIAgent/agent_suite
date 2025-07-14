"""
Geometry Knowledge Graph module.

This module defines a geometry-specific knowledge graph that inherits from
the base KnowledgeGraph and adds geometry-specific concepts and relationships.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..knowledge_graph.graph import KnowledgeGraph
from ..knowledge_graph.concept import Concept


class GeometryKnowledgeGraph(KnowledgeGraph):
    """
    Geometry-specific knowledge graph that extends the base KnowledgeGraph
    with geometry concepts, relationships, and domain-specific methods.
    """
    
    def __init__(self, name: str = "Geometry Knowledge Graph", config=None):
        """
        Initialize the geometry knowledge graph.
        
        Args:
            name: Name of the knowledge graph
            config: Optional storage configuration
        """
        super().__init__(name, config)
        self.geometry_data_path = Path(__file__).parent / "data"
        
    def initialize_geometry_concepts(self) -> Dict[str, str]:
        """
        Initialize the knowledge graph with geometry concepts.
        Loads concepts from the enhanced geometry exercises data.
        
        Returns:
            Dictionary mapping concept names to concept IDs
        """
        # Load geometry exercises to extract concepts
        exercises_file = self.geometry_data_path / "geometry_exercises_enhanced_complete.json"
        
        if not exercises_file.exists():
            raise FileNotFoundError(f"Geometry exercises file not found: {exercises_file}")
            
        with open(exercises_file, 'r') as f:
            exercises_data = json.load(f)
        
        concept_map = {}
        
        # Extract unique concepts from exercises
        concepts_seen = set()
        for exercise in exercises_data:
            for concept in exercise.get('concepts', []):
                if concept not in concepts_seen:
                    concepts_seen.add(concept)
                    
                    # Create concept object
                    geometry_concept = Concept(
                        name=concept.replace('_', ' ').title(),
                        description=f"Geometry concept: {concept}",
                        difficulty=self._determine_concept_difficulty(concept),
                        time_to_master=self._estimate_time_to_master(concept),
                        category="Geometry",
                        concept_id=f"geom_{concept.lower()}",
                        grade_level=self._determine_grade_level(concept)
                    )
                    
                    # Add to knowledge graph
                    self.add_concept(geometry_concept)
                    concept_map[concept] = geometry_concept.id
        
        # Add geometry-specific prerequisite relationships
        self._add_geometry_prerequisites(concept_map)
        
        return concept_map
    
    def _determine_concept_difficulty(self, concept: str) -> int:
        """Determine difficulty level for a geometry concept."""
        elementary_concepts = [
            'basic_shapes', 'shape_identification', 'counting_sides', 
            'shape_comparison', 'symmetry_basic'
        ]
        
        middle_concepts = [
            'angles', 'triangles', 'quadrilaterals', 'area_perimeter',
            'coordinate_geometry', 'transformations'
        ]
        
        advanced_concepts = [
            'trigonometry', 'advanced_triangles', 'circle_theorems',
            'volume_surface_area', 'proofs'
        ]
        
        if any(elem in concept.lower() for elem in elementary_concepts):
            return 2
        elif any(mid in concept.lower() for mid in middle_concepts):
            return 3
        elif any(adv in concept.lower() for adv in advanced_concepts):
            return 4
        else:
            return 3  # Default middle difficulty
    
    def _estimate_time_to_master(self, concept: str) -> int:
        """Estimate time to master a geometry concept in minutes."""
        difficulty = self._determine_concept_difficulty(concept)
        base_time = {2: 45, 3: 60, 4: 90, 5: 120}
        return base_time.get(difficulty, 60)
    
    def _determine_grade_level(self, concept: str) -> str:
        """Determine appropriate grade level for concept."""
        elementary_concepts = ['basic_shapes', 'shape_identification', 'counting']
        middle_concepts = ['angles', 'triangles', 'area', 'perimeter']
        high_school_concepts = ['trigonometry', 'proofs', 'advanced']
        
        if any(elem in concept.lower() for elem in elementary_concepts):
            return "K-5"
        elif any(mid in concept.lower() for mid in middle_concepts):
            return "6-8"
        elif any(high in concept.lower() for high in high_school_concepts):
            return "9-12"
        else:
            return "6-8"  # Default
    
    def _add_geometry_prerequisites(self, concept_map: Dict[str, str]) -> None:
        """Add geometry-specific prerequisite relationships."""
        # Define prerequisite relationships for geometry
        prerequisites = [
            # Basic progression
            ('basic_shapes', 'shape_identification'),
            ('shape_identification', 'counting_sides'),
            ('counting_sides', 'angles'),
            
            # Angle relationships
            ('angles', 'triangles'),
            ('angles', 'quadrilaterals'),
            
            # Area and perimeter
            ('basic_shapes', 'area_perimeter'),
            ('triangles', 'area_triangles'),
            ('quadrilaterals', 'area_quadrilaterals'),
            
            # Advanced concepts
            ('triangles', 'trigonometry'),
            ('area_perimeter', 'volume_surface_area'),
            ('coordinate_geometry', 'transformations'),
        ]
        
        # Add prerequisite relationships
        for prereq_concept, dependent_concept in prerequisites:
            prereq_id = concept_map.get(prereq_concept)
            dependent_id = concept_map.get(dependent_concept)
            
            if prereq_id and dependent_id:
                self.add_prerequisite(prereq_id, dependent_id)
    
    def get_geometry_learning_path(self, target_concept: str) -> List[Concept]:
        """
        Get a learning path for a specific geometry concept.
        
        Args:
            target_concept: Name of the target geometry concept
            
        Returns:
            List of concepts in learning order
        """
        # Find concept by name
        target_concept_obj = None
        for concept in self.get_all_concepts():
            if target_concept.lower() in concept.name.lower():
                target_concept_obj = concept
                break
        
        if not target_concept_obj:
            return []
        
        # Use parent class method to get learning path
        return self.get_learning_path(target_concept_obj.id)
    
    def get_concepts_by_grade_level(self, grade_level: str) -> List[Concept]:
        """
        Get concepts appropriate for a specific grade level.
        
        Args:
            grade_level: Grade level (e.g., "K-5", "6-8", "9-12")
            
        Returns:
            List of concepts for that grade level
        """
        matching_concepts = []
        for concept in self.get_all_concepts():
            if hasattr(concept, 'grade_level') and concept.grade_level == grade_level:
                matching_concepts.append(concept)
        
        return matching_concepts
    
    def get_geometry_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the geometry knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        all_concepts = self.get_all_concepts()
        
        # Count by difficulty
        difficulty_counts = {}
        grade_level_counts = {}
        
        for concept in all_concepts:
            diff = concept.difficulty
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            
            if hasattr(concept, 'grade_level'):
                grade = concept.grade_level
                grade_level_counts[grade] = grade_level_counts.get(grade, 0) + 1
        
        return {
            'total_concepts': len(all_concepts),
            'difficulty_distribution': difficulty_counts,
            'grade_level_distribution': grade_level_counts,
            'average_difficulty': sum(c.difficulty for c in all_concepts) / len(all_concepts) if all_concepts else 0,
            'total_relationships': len(self.get_all_relationships()) if hasattr(self, 'get_all_relationships') else 0
        } 