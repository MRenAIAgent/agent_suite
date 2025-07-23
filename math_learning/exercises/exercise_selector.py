"""
Exercise Selector module.

This module provides intelligent exercise selection based on concepts, difficulty scores,
and learning progression. It implements a sophisticated scoring system for matching
exercises to student needs.
"""

import random
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum

from .exercise import Exercise
from .exercise_bank import ExerciseBank
from ..knowledge_graph.graph import KnowledgeGraph
from ..learning_graph.user_model import LearningGraph


class DifficultyLevel(Enum):
    """Standardized difficulty levels."""
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


@dataclass
class ExerciseScore:
    """Scoring information for an exercise selection."""
    exercise: Exercise
    total_score: float
    difficulty_match_score: float
    concept_relevance_score: float
    freshness_score: float
    prerequisite_score: float
    reason: str


class ExerciseSelector:
    """
    Intelligent exercise selector that picks exercises based on:
    1. Concept relevance
    2. Difficulty matching
    3. Learning progression
    4. Exercise freshness (avoiding repetition)
    5. Prerequisite readiness
    """
    
    def __init__(self, exercise_bank: ExerciseBank, knowledge_graph: Optional[KnowledgeGraph] = None):
        """
        Initialize the exercise selector.
        
        Args:
            exercise_bank: The exercise bank to select from
            knowledge_graph: Optional knowledge graph for prerequisite analysis
        """
        self.exercise_bank = exercise_bank
        self.knowledge_graph = knowledge_graph
        
    def select_exercises_by_concept_and_difficulty(
        self, 
        concept_id: str, 
        difficulty_score: float, 
        count: int = 5,
        avoid_exercises: Optional[Set[str]] = None
    ) -> List[Exercise]:
        """
        Select exercises for a specific concept and difficulty level.
        
        Args:
            concept_id: The target concept ID
            difficulty_score: Target difficulty (1.0-5.0)
            count: Number of exercises to select
            avoid_exercises: Set of exercise IDs to avoid
            
        Returns:
            List of selected exercises, ordered by relevance
        """
        avoid_exercises = avoid_exercises or set()
        
        # Get all exercises for the concept
        concept_exercises = self.exercise_bank.get_exercises_for_concept(concept_id)
        
        if not concept_exercises:
            return []
            
        # Score each exercise
        scored_exercises = []
        for exercise in concept_exercises:
            if exercise.id in avoid_exercises:
                continue
                
            score = self._score_exercise_for_concept_difficulty(
                exercise, concept_id, difficulty_score
            )
            scored_exercises.append(score)
            
        # Sort by total score (highest first)
        scored_exercises.sort(key=lambda x: x.total_score, reverse=True)
        
        # Return top exercises
        return [score.exercise for score in scored_exercises[:count]]
    
    def select_adaptive_exercises(
        self,
        learning_graph: LearningGraph,
        target_concepts: Optional[List[str]] = None,
        count: int = 5,
        difficulty_adjustment: float = 0.0
    ) -> List[Tuple[Exercise, str]]:
        """
        Select exercises adaptively based on learning graph.
        
        Args:
            learning_graph: Student's learning progress
            target_concepts: Optional list of concepts to focus on
            count: Number of exercises to select
            difficulty_adjustment: Adjustment to difficulty (-1.0 to 1.0)
            
        Returns:
            List of (exercise, reason) tuples
        """
        if not target_concepts:
            # Find concepts that need practice
            target_concepts = self._identify_practice_concepts(learning_graph)
            
        selected_exercises = []
        exercises_used = set()
        
        for concept_id in target_concepts:
            if len(selected_exercises) >= count:
                break
                
            # Determine target difficulty based on mastery
            mastery = learning_graph.get_mastery(concept_id)
            confidence = learning_graph.get_confidence(concept_id)
            
            target_difficulty = self._calculate_adaptive_difficulty(
                mastery, confidence, difficulty_adjustment
            )
            
            # Get exercises for this concept
            concept_exercises = self.select_exercises_by_concept_and_difficulty(
                concept_id, 
                target_difficulty, 
                count=2,  # Get 2 per concept
                avoid_exercises=exercises_used
            )
            
            for exercise in concept_exercises:
                if len(selected_exercises) >= count:
                    break
                    
                reason = self._generate_selection_reason(
                    concept_id, mastery, confidence, target_difficulty
                )
                selected_exercises.append((exercise, reason))
                exercises_used.add(exercise.id)
                
        return selected_exercises
    
    def select_learning_path_exercises(
        self,
        learning_graph: LearningGraph,
        knowledge_graph: KnowledgeGraph,
        count: int = 10
    ) -> List[Tuple[Exercise, str, str]]:
        """
        Select exercises based on optimal learning path.
        
        Args:
            learning_graph: Student's learning progress
            knowledge_graph: Knowledge graph for prerequisite analysis
            count: Number of exercises to select
            
        Returns:
            List of (exercise, concept_id, reason) tuples
        """
        # Find next concepts in learning path
        next_concepts = self._find_next_learning_concepts(learning_graph, knowledge_graph)
        
        selected_exercises = []
        exercises_used = set()
        
        for concept_info in next_concepts:
            if len(selected_exercises) >= count:
                break
                
            concept_id = concept_info['concept_id']
            priority = concept_info['priority']
            readiness = concept_info['readiness']
            
            # Calculate difficulty based on readiness
            base_difficulty = 2.0 if readiness > 0.8 else 1.5
            target_difficulty = min(5.0, base_difficulty + priority)
            
            # Select exercises
            exercises = self.select_exercises_by_concept_and_difficulty(
                concept_id,
                target_difficulty,
                count=3,
                avoid_exercises=exercises_used
            )
            
            for exercise in exercises:
                if len(selected_exercises) >= count:
                    break
                    
                reason = f"Learning path: {concept_info['reason']} (readiness: {readiness:.1f})"
                selected_exercises.append((exercise, concept_id, reason))
                exercises_used.add(exercise.id)
                
        return selected_exercises
    
    def _score_exercise_for_concept_difficulty(
        self, 
        exercise: Exercise, 
        concept_id: str, 
        target_difficulty: float
    ) -> ExerciseScore:
        """Score an exercise for concept and difficulty matching."""
        
        # 1. Concept relevance score (0.0-1.0)
        concept_weight = exercise.get_concept_weight(concept_id)
        concept_relevance_score = concept_weight
        
        # 2. Difficulty match score (0.0-1.0)
        difficulty_diff = abs(exercise.difficulty - target_difficulty)
        difficulty_match_score = max(0.0, 1.0 - (difficulty_diff / 4.0))
        
        # 3. Freshness score (always 1.0 for now, can be enhanced with history)
        freshness_score = 1.0
        
        # 4. Prerequisite score (1.0 for now, can be enhanced with knowledge graph)
        prerequisite_score = 1.0
        
        # Calculate weighted total score
        total_score = (
            concept_relevance_score * 0.4 +
            difficulty_match_score * 0.3 +
            freshness_score * 0.2 +
            prerequisite_score * 0.1
        )
        
        reason = f"Concept match: {concept_relevance_score:.2f}, Difficulty match: {difficulty_match_score:.2f}"
        
        return ExerciseScore(
            exercise=exercise,
            total_score=total_score,
            difficulty_match_score=difficulty_match_score,
            concept_relevance_score=concept_relevance_score,
            freshness_score=freshness_score,
            prerequisite_score=prerequisite_score,
            reason=reason
        )
    
    def _identify_practice_concepts(self, learning_graph: LearningGraph) -> List[str]:
        """Identify concepts that need practice based on learning graph."""
        concepts_needing_practice = []
        
        # Find struggling concepts (low mastery)
        struggling = learning_graph.get_struggling_concepts(threshold=0.4)
        concepts_needing_practice.extend(struggling)
        
        # Find concepts with low confidence
        for concept_id, mastery_info in learning_graph.concept_mastery.items():
            confidence = mastery_info.get('confidence', 0.5)
            mastery = mastery_info.get('mastery', 0.0)
            
            if confidence < 0.6 and mastery > 0.2:  # Some knowledge but low confidence
                concepts_needing_practice.append(concept_id)
                
        # Remove duplicates and limit
        return list(set(concepts_needing_practice))[:10]
    
    def _calculate_adaptive_difficulty(
        self, 
        mastery: float, 
        confidence: float, 
        adjustment: float = 0.0
    ) -> float:
        """Calculate adaptive difficulty based on mastery and confidence."""
        
        # Base difficulty calculation
        if mastery < 0.3:
            base_difficulty = 1.5  # Very easy
        elif mastery < 0.5:
            base_difficulty = 2.0  # Easy
        elif mastery < 0.7:
            base_difficulty = 2.5  # Medium-easy
        elif mastery < 0.85:
            base_difficulty = 3.0  # Medium
        else:
            base_difficulty = 3.5  # Medium-hard
            
        # Adjust for confidence
        confidence_adjustment = (confidence - 0.5) * 0.5  # -0.25 to +0.25
        
        # Apply user adjustment
        final_difficulty = base_difficulty + confidence_adjustment + adjustment
        
        # Clamp to valid range
        return max(1.0, min(5.0, final_difficulty))
    
    def _generate_selection_reason(
        self, 
        concept_id: str, 
        mastery: float, 
        confidence: float, 
        target_difficulty: float
    ) -> str:
        """Generate explanation for exercise selection."""
        
        mastery_desc = "low" if mastery < 0.5 else "medium" if mastery < 0.8 else "high"
        confidence_desc = "low" if confidence < 0.6 else "medium" if confidence < 0.8 else "high"
        difficulty_desc = DifficultyLevel(round(target_difficulty)).name.lower().replace('_', ' ')
        
        return f"Targeting {concept_id}: {mastery_desc} mastery, {confidence_desc} confidence → {difficulty_desc} exercises"
    
    def _find_next_learning_concepts(
        self, 
        learning_graph: LearningGraph, 
        knowledge_graph: KnowledgeGraph
    ) -> List[Dict[str, Any]]:
        """Find the next concepts to learn based on prerequisites."""
        
        next_concepts = []
        all_concepts = knowledge_graph.get_all_concepts()
        
        for concept in all_concepts:
            # Check if concept is already mastered
            mastery = learning_graph.get_mastery(concept.id)
            if mastery > 0.8:
                continue
                
            # Check prerequisite readiness
            prerequisites = knowledge_graph.get_prerequisites(concept.id)
            if not prerequisites:
                # No prerequisites - can start immediately
                readiness = 1.0
                reason = "No prerequisites required"
            else:
                # Calculate readiness based on prerequisite mastery
                prereq_masteries = [learning_graph.get_mastery(prereq) for prereq in prerequisites]
                readiness = sum(prereq_masteries) / len(prereq_masteries) if prereq_masteries else 0.0
                
                if readiness < 0.6:
                    continue  # Not ready yet
                    
                reason = f"Prerequisites {readiness:.1f} ready"
                
            # Calculate priority (inverse of current mastery + readiness bonus)
            priority = (1.0 - mastery) + (readiness * 0.5)
            
            next_concepts.append({
                'concept_id': concept.id,
                'priority': priority,
                'readiness': readiness,
                'reason': reason
            })
            
        # Sort by priority (highest first)
        next_concepts.sort(key=lambda x: x['priority'], reverse=True)
        
        return next_concepts[:10]  # Return top 10 