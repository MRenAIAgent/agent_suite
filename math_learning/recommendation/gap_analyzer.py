"""
Gap Analyzer module.

This module defines the gap analyzer that identifies knowledge gaps.
"""

import sys
import os
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from knowledge_graph.graph import KnowledgeGraph
from math_learning.learning_graph.user_model import LearningGraph


class GapInfo:
    """Information about a knowledge gap."""
    
    def __init__(self, concept_id: str, mastery: float, impact_score: float,
                 centrality: float, priority: float):
        """
        Initialize gap information.
        
        Args:
            concept_id: ID of the concept
            mastery: Current mastery level
            impact_score: How many other concepts this unlocks
            centrality: Centrality in the knowledge graph
            priority: Overall priority score
        """
        self.concept_id = concept_id
        self.mastery = mastery
        self.impact_score = impact_score
        self.centrality = centrality
        self.priority = priority
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "concept_id": self.concept_id,
            "mastery": self.mastery,
            "impact_score": self.impact_score,
            "centrality": self.centrality,
            "priority": self.priority
        }


class GapAnalyzer:
    """Analyzes gaps between knowledge graph and learning graph."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Initialize the gap analyzer.
        
        Args:
            knowledge_graph: The reference knowledge graph
        """
        self.knowledge_graph = knowledge_graph
        
    def detect_critical_gaps(self, learning_graph: LearningGraph, 
                           mastery_threshold: float = 0.7) -> List[GapInfo]:
        """
        Detect critical knowledge gaps for a user.
        
        Args:
            learning_graph: The user's learning graph
            mastery_threshold: Threshold below which a concept is considered not mastered
            
        Returns:
            List of GapInfo objects, sorted by priority
        """
        gaps = []
        
        for concept in self.knowledge_graph.get_all_concepts():
            mastery = learning_graph.get_mastery(concept.id, default=0.0)
            
            if mastery < mastery_threshold:
                # Get dependent concepts (concepts this unlocks)
                dependent_concepts = self.knowledge_graph.get_dependent_concepts(concept.id)
                
                # Calculate impact score based on how many concepts this unlocks
                impact_score = len(dependent_concepts) * 0.5
                
                # Calculate centrality in knowledge graph
                centrality = self.knowledge_graph.calculate_centrality(concept.id)
                
                # Get prerequisites for this concept
                prerequisites = self.knowledge_graph.get_prerequisites(concept.id)
                
                # If this is a root concept (no prerequisites), give it higher priority
                prerequisite_bonus = 1.0 if len(prerequisites) == 0 else 0.0
                
                # Calculate priority score with prerequisite bonus
                priority = (impact_score * 0.4 + 
                           centrality * 0.3 + 
                           (mastery_threshold - mastery) * 0.2 +
                           prerequisite_bonus * 0.5)
                
                gaps.append(GapInfo(
                    concept_id=concept.id,
                    mastery=mastery,
                    impact_score=impact_score,
                    centrality=centrality,
                    priority=priority
                ))
        
        # Sort by priority (descending)
        return sorted(gaps, key=lambda x: x.priority, reverse=True)
        
    def find_learning_boundary(self, learning_graph: LearningGraph,
                             mastery_threshold: float = 0.7) -> List[str]:
        """
        Find the learning boundary - concepts ready to be learned next.
        
        The learning boundary consists of concepts that haven't been mastered
        but have all prerequisites mastered.
        
        Args:
            learning_graph: The user's learning graph
            mastery_threshold: Threshold for considering a concept mastered
            
        Returns:
            List of concept IDs at the learning boundary
        """
        # Get mastered concepts
        mastered = set(learning_graph.get_mastered_concepts(threshold=mastery_threshold))
        boundary = []
        
        for concept in self.knowledge_graph.get_all_concepts():
            # Skip already mastered concepts
            if concept.id in mastered:
                continue
                
            # Get prerequisites
            prerequisites = self.knowledge_graph.get_prerequisites(concept.id)
            prerequisite_ids = [p.id for p in prerequisites]
            
            # If all prerequisites are mastered, this concept is at the learning boundary
            if all(pre_id in mastered for pre_id in prerequisite_ids):
                boundary.append(concept.id)
                
        return boundary
        
    def get_next_concepts(self, learning_graph: LearningGraph, 
                        count: int = 3) -> List[str]:
        """
        Get recommended next concepts to learn.
        
        Args:
            learning_graph: The user's learning graph
            count: Number of concepts to recommend
            
        Returns:
            List of recommended concept IDs
        """
        # Find concepts at the learning boundary
        boundary = self.find_learning_boundary(learning_graph)
        
        # Get gap information for these concepts
        gaps = []
        for concept_id in boundary:
            mastery = learning_graph.get_mastery(concept_id)
            
            # Get dependent concepts (concepts this unlocks)
            dependent_concepts = self.knowledge_graph.get_dependent_concepts(concept_id)
            
            # Calculate impact score based on how many concepts this unlocks
            impact_score = len(dependent_concepts) * 0.5
            
            # Calculate centrality in knowledge graph
            centrality = self.knowledge_graph.calculate_centrality(concept_id)
            
            # Calculate priority score with higher weight on impact
            priority = impact_score * 0.6 + centrality * 0.4
            
            gaps.append(GapInfo(
                concept_id=concept_id,
                mastery=mastery,
                impact_score=impact_score,
                centrality=centrality,
                priority=priority
            ))
            
        # Sort by priority and return top count
        sorted_gaps = sorted(gaps, key=lambda x: x.priority, reverse=True)
        return [gap.concept_id for gap in sorted_gaps[:count]] 