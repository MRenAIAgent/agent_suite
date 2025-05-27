"""
Personalized Learning System module.

This module provides advanced personalization features for the math learning system.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter
import statistics

from .user_model import LearningGraph
from ..knowledge_graph.graph import KnowledgeGraph
from ..recommendation.gap_analyzer import GapAnalyzer


@dataclass
class LearningPattern:
    """Represents a user's learning pattern."""
    concept_id: str
    attempts: int
    success_rate: float
    avg_time_per_attempt: float
    difficulty_progression: List[float]
    mistake_patterns: List[str]
    learning_velocity: float  # How quickly mastery improves
    retention_rate: float  # How well knowledge is retained over time
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeaknessProfile:
    """Detailed profile of a user's weaknesses."""
    concept_id: str
    weakness_type: str  # 'conceptual', 'procedural', 'computational'
    severity: float  # 0.0 to 1.0
    related_concepts: List[str]
    common_mistakes: List[str]
    suggested_interventions: List[str]
    confidence_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearningInsight:
    """Actionable insight about user's learning."""
    insight_type: str  # 'strength', 'weakness', 'pattern', 'recommendation'
    title: str
    description: str
    confidence: float
    actionable_steps: List[str]
    related_concepts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersonalizedLearningSystem:
    """Advanced personalized learning system with pattern analysis and adaptive recommendations."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Initialize the personalized learning system.
        
        Args:
            knowledge_graph: The knowledge graph containing concept relationships
        """
        self.knowledge_graph = knowledge_graph
        self.gap_analyzer = GapAnalyzer(knowledge_graph)
        
    def analyze_learning_patterns(self, learning_graph: LearningGraph) -> Dict[str, LearningPattern]:
        """
        Analyze learning patterns for each concept the user has interacted with.
        
        Args:
            learning_graph: The user's learning graph
            
        Returns:
            Dictionary mapping concept IDs to learning patterns
        """
        patterns = {}
        
        for concept_id, mastery_data in learning_graph.concept_mastery.items():
            # Calculate attempts from exercise history
            attempts = sum(
                len(attempts) for exercise_id, attempts in learning_graph.exercise_history.items()
                if any(attempt['concept_id'] == concept_id for attempt in attempts)
            )
            
            if attempts == 0:
                continue
                
            # Calculate success rate
            correct_attempts = sum(
                sum(1 for attempt in attempts if attempt['concept_id'] == concept_id and attempt['result'])
                for attempts in learning_graph.exercise_history.values()
            )
            success_rate = correct_attempts / attempts if attempts > 0 else 0.0
            
            # Analyze difficulty progression from history
            history = mastery_data.get('history', [])
            difficulty_progression = [h['mastery'] for h in history] + [mastery_data['mastery']]
            
            # Calculate learning velocity (rate of improvement)
            learning_velocity = self._calculate_learning_velocity(difficulty_progression)
            
            # Calculate retention rate
            retention_rate = self._calculate_retention_rate(history)
            
            # Identify mistake patterns
            mistake_patterns = self._identify_mistake_patterns(learning_graph, concept_id)
            
            patterns[concept_id] = LearningPattern(
                concept_id=concept_id,
                attempts=attempts,
                success_rate=success_rate,
                avg_time_per_attempt=60.0,  # Default, could be tracked in future
                difficulty_progression=difficulty_progression,
                mistake_patterns=mistake_patterns,
                learning_velocity=learning_velocity,
                retention_rate=retention_rate
            )
            
        return patterns
    
    def detect_weaknesses(self, learning_graph: LearningGraph) -> List[WeaknessProfile]:
        """
        Detect and analyze user weaknesses with detailed profiles.
        
        Args:
            learning_graph: The user's learning graph
            
        Returns:
            List of weakness profiles sorted by severity
        """
        weaknesses = []
        patterns = self.analyze_learning_patterns(learning_graph)
        
        for concept_id, pattern in patterns.items():
            concept = self.knowledge_graph.get_concept(concept_id)
            if not concept:
                continue
                
            # Determine weakness type and severity
            weakness_type, severity = self._classify_weakness(pattern, learning_graph, concept_id)
            
            if severity > 0.3:  # Only include significant weaknesses
                # Find related struggling concepts
                related_concepts = self._find_related_struggling_concepts(
                    learning_graph, concept_id, patterns
                )
                
                # Generate intervention suggestions
                interventions = self._suggest_interventions(weakness_type, concept, pattern)
                
                weaknesses.append(WeaknessProfile(
                    concept_id=concept_id,
                    weakness_type=weakness_type,
                    severity=severity,
                    related_concepts=related_concepts,
                    common_mistakes=pattern.mistake_patterns,
                    suggested_interventions=interventions,
                    confidence_level=learning_graph.get_confidence(concept_id)
                ))
        
        return sorted(weaknesses, key=lambda w: w.severity, reverse=True)
    
    def generate_personalized_path(self, learning_graph: LearningGraph, 
                                 target_concept_id: str,
                                 max_concepts: int = 10) -> List[Tuple[str, float]]:
        """
        Generate a personalized learning path considering user's strengths and weaknesses.
        
        Args:
            learning_graph: The user's learning graph
            target_concept_id: The target concept to reach
            max_concepts: Maximum number of concepts in the path
            
        Returns:
            List of (concept_id, priority_score) tuples in recommended order
        """
        # Get basic learning path from knowledge graph
        basic_path = self._get_basic_learning_path(target_concept_id)
        
        # Analyze user patterns
        patterns = self.analyze_learning_patterns(learning_graph)
        weaknesses = self.detect_weaknesses(learning_graph)
        
        # Score each concept in the path
        scored_path = []
        for concept_id in basic_path:
            priority_score = self._calculate_concept_priority(
                concept_id, learning_graph, patterns, weaknesses
            )
            scored_path.append((concept_id, priority_score))
        
        # Sort by priority and limit to max_concepts
        scored_path.sort(key=lambda x: x[1], reverse=True)
        return scored_path[:max_concepts]
    
    def get_adaptive_recommendations(self, learning_graph: LearningGraph,
                                   num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get adaptive recommendations based on user's current state and learning patterns.
        
        Args:
            learning_graph: The user's learning graph
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Get learning boundary concepts
        boundary_concepts = self.gap_analyzer.find_learning_boundary(learning_graph)
        
        # Analyze patterns and weaknesses
        patterns = self.analyze_learning_patterns(learning_graph)
        weaknesses = self.detect_weaknesses(learning_graph)
        
        # Generate different types of recommendations
        
        # 1. Address critical weaknesses
        for weakness in weaknesses[:2]:
            recommendations.append({
                'type': 'weakness_remediation',
                'concept_id': weakness.concept_id,
                'title': f"Strengthen {self.knowledge_graph.get_concept(weakness.concept_id).name}",
                'description': f"Focus on {weakness.weakness_type} understanding",
                'reason': f"Detected {weakness.weakness_type} weakness with severity {weakness.severity:.2f}",
                'priority': weakness.severity,
                'interventions': weakness.suggested_interventions
            })
        
        # 2. Ready-to-learn concepts
        for concept_id in boundary_concepts[:3]:
            concept = self.knowledge_graph.get_concept(concept_id)
            if concept:
                recommendations.append({
                    'type': 'ready_to_learn',
                    'concept_id': concept_id,
                    'title': f"Learn {concept.name}",
                    'description': "You're ready to tackle this new concept",
                    'reason': "All prerequisites have been mastered",
                    'priority': 0.8,
                    'difficulty': concept.difficulty
                })
        
        # 3. Review recommendations based on retention
        review_concepts = self._identify_review_concepts(learning_graph, patterns)
        for concept_id in review_concepts[:2]:
            concept = self.knowledge_graph.get_concept(concept_id)
            if concept:
                recommendations.append({
                    'type': 'review',
                    'concept_id': concept_id,
                    'title': f"Review {concept.name}",
                    'description': "Reinforce this concept to improve retention",
                    'reason': "Good initial mastery but declining retention detected",
                    'priority': 0.6
                })
        
        # Sort by priority and return top recommendations
        recommendations.sort(key=lambda r: r['priority'], reverse=True)
        return recommendations[:num_recommendations]
    
    def generate_learning_insights(self, learning_graph: LearningGraph) -> List[LearningInsight]:
        """
        Generate actionable insights about the user's learning.
        
        Args:
            learning_graph: The user's learning graph
            
        Returns:
            List of learning insights
        """
        insights = []
        patterns = self.analyze_learning_patterns(learning_graph)
        weaknesses = self.detect_weaknesses(learning_graph)
        
        # Identify strengths
        strengths = self._identify_strengths(learning_graph, patterns)
        for strength in strengths:
            insights.append(LearningInsight(
                insight_type='strength',
                title=f"Strong in {strength['area']}",
                description=strength['description'],
                confidence=strength['confidence'],
                actionable_steps=[f"Leverage this strength to learn {c}" for c in strength['related_concepts']],
                related_concepts=strength['related_concepts']
            ))
        
        # Learning velocity insights
        fast_learners = [p for p in patterns.values() if p.learning_velocity > 0.7]
        if fast_learners:
            insights.append(LearningInsight(
                insight_type='pattern',
                title="Fast Learning Pattern Detected",
                description=f"You learn quickly in {len(fast_learners)} concept areas",
                confidence=0.8,
                actionable_steps=["Consider tackling more challenging concepts", "Explore advanced topics"],
                related_concepts=[p.concept_id for p in fast_learners]
            ))
        
        # Retention insights
        low_retention = [p for p in patterns.values() if p.retention_rate < 0.5]
        if low_retention:
            insights.append(LearningInsight(
                insight_type='pattern',
                title="Review Strategy Needed",
                description=f"Knowledge retention could be improved in {len(low_retention)} areas",
                confidence=0.7,
                actionable_steps=["Schedule regular review sessions", "Use spaced repetition"],
                related_concepts=[p.concept_id for p in low_retention]
            ))
        
        return insights
    
    def _calculate_learning_velocity(self, progression: List[float]) -> float:
        """Calculate how quickly mastery improves over time."""
        if len(progression) < 2:
            return 0.5
        
        # Calculate average improvement rate
        improvements = [progression[i] - progression[i-1] for i in range(1, len(progression))]
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, avg_improvement * 2 + 0.5))
    
    def _calculate_retention_rate(self, history: List[Dict]) -> float:
        """Calculate how well knowledge is retained over time."""
        if len(history) < 2:
            return 0.7  # Default assumption
        
        # Look for patterns of mastery decline
        recent_changes = []
        for i in range(1, len(history)):
            change = history[i]['mastery'] - history[i-1]['mastery']
            recent_changes.append(change)
        
        # If mastery generally increases or stays stable, retention is good
        avg_change = np.mean(recent_changes) if recent_changes else 0.0
        return max(0.0, min(1.0, 0.7 + avg_change))
    
    def _identify_mistake_patterns(self, learning_graph: LearningGraph, concept_id: str) -> List[str]:
        """Identify common mistake patterns for a concept."""
        # This is a simplified version - in practice, you'd analyze actual mistake data
        patterns = []
        
        mastery = learning_graph.get_mastery(concept_id)
        if mastery < 0.3:
            patterns.append("fundamental_misunderstanding")
        elif mastery < 0.6:
            patterns.append("procedural_errors")
        
        return patterns
    
    def _classify_weakness(self, pattern: LearningPattern, learning_graph: LearningGraph, 
                          concept_id: str) -> Tuple[str, float]:
        """Classify the type and severity of a weakness."""
        mastery = learning_graph.get_mastery(concept_id)
        
        # Determine weakness type
        if pattern.success_rate < 0.3:
            weakness_type = "conceptual"
        elif pattern.learning_velocity < 0.3:
            weakness_type = "procedural"
        else:
            weakness_type = "computational"
        
        # Calculate severity based on multiple factors
        severity = (
            (1.0 - mastery) * 0.4 +
            (1.0 - pattern.success_rate) * 0.3 +
            (1.0 - pattern.learning_velocity) * 0.2 +
            (1.0 - pattern.retention_rate) * 0.1
        )
        
        return weakness_type, min(1.0, severity)
    
    def _find_related_struggling_concepts(self, learning_graph: LearningGraph, 
                                        concept_id: str, patterns: Dict[str, LearningPattern]) -> List[str]:
        """Find related concepts where the user is also struggling."""
        concept = self.knowledge_graph.get_concept(concept_id)
        if not concept:
            return []
        
        related = []
        
        # Check prerequisites and dependents
        for related_concept in (concept.prerequisites | concept.dependents):
            if related_concept in patterns:
                pattern = patterns[related_concept]
                if pattern.success_rate < 0.5 or learning_graph.get_mastery(related_concept) < 0.5:
                    related.append(related_concept)
        
        return related
    
    def _suggest_interventions(self, weakness_type: str, concept: Any, pattern: LearningPattern) -> List[str]:
        """Suggest specific interventions based on weakness type."""
        interventions = []
        
        if weakness_type == "conceptual":
            interventions.extend([
                "Review fundamental concepts and definitions",
                "Work through visual examples and analogies",
                "Practice with concrete manipulatives"
            ])
        elif weakness_type == "procedural":
            interventions.extend([
                "Break down procedures into smaller steps",
                "Practice step-by-step guided examples",
                "Use procedural checklists"
            ])
        else:  # computational
            interventions.extend([
                "Practice basic computational skills",
                "Use computational aids when appropriate",
                "Focus on accuracy before speed"
            ])
        
        return interventions
    
    def _get_basic_learning_path(self, target_concept_id: str) -> List[str]:
        """Get basic learning path from knowledge graph."""
        if target_concept_id not in self.knowledge_graph.concepts:
            return []
        
        # Simple topological sort of prerequisites
        visited = set()
        path = []
        
        def dfs(concept_id):
            if concept_id in visited:
                return
            visited.add(concept_id)
            
            concept = self.knowledge_graph.get_concept(concept_id)
            if concept:
                for prereq in concept.prerequisites:
                    dfs(prereq)
                path.append(concept_id)
        
        dfs(target_concept_id)
        return path
    
    def _calculate_concept_priority(self, concept_id: str, learning_graph: LearningGraph,
                                  patterns: Dict[str, LearningPattern], 
                                  weaknesses: List[WeaknessProfile]) -> float:
        """Calculate priority score for a concept in the learning path."""
        base_priority = 0.5
        
        # Boost priority if it addresses a weakness
        for weakness in weaknesses:
            if weakness.concept_id == concept_id:
                base_priority += weakness.severity * 0.4
        
        # Boost priority if user has shown aptitude in related areas
        if concept_id in patterns:
            pattern = patterns[concept_id]
            base_priority += pattern.learning_velocity * 0.2
        
        # Consider current mastery level
        mastery = learning_graph.get_mastery(concept_id)
        if mastery < 0.3:  # Low mastery = high priority
            base_priority += 0.3
        
        return min(1.0, base_priority)
    
    def _identify_review_concepts(self, learning_graph: LearningGraph,
                                patterns: Dict[str, LearningPattern]) -> List[str]:
        """Identify concepts that need review based on retention patterns."""
        review_concepts = []
        
        for concept_id, pattern in patterns.items():
            mastery = learning_graph.get_mastery(concept_id)
            
            # Concepts with good initial mastery but poor retention
            if mastery > 0.6 and pattern.retention_rate < 0.5:
                review_concepts.append(concept_id)
        
        return review_concepts
    
    def _identify_strengths(self, learning_graph: LearningGraph,
                          patterns: Dict[str, LearningPattern]) -> List[Dict[str, Any]]:
        """Identify user's learning strengths."""
        strengths = []
        
        # Group concepts by category
        category_performance = defaultdict(list)
        for concept_id, pattern in patterns.items():
            concept = self.knowledge_graph.get_concept(concept_id)
            if concept:
                category_performance[concept.category].append({
                    'concept_id': concept_id,
                    'mastery': learning_graph.get_mastery(concept_id),
                    'success_rate': pattern.success_rate,
                    'learning_velocity': pattern.learning_velocity
                })
        
        # Identify strong categories
        for category, concepts in category_performance.items():
            if len(concepts) >= 2:  # Need at least 2 concepts to identify a pattern
                avg_mastery = np.mean([c['mastery'] for c in concepts])
                avg_success = np.mean([c['success_rate'] for c in concepts])
                avg_velocity = np.mean([c['learning_velocity'] for c in concepts])
                
                if avg_mastery > 0.7 and avg_success > 0.7:
                    strengths.append({
                        'area': category,
                        'description': f"Consistently strong performance in {category}",
                        'confidence': min(avg_mastery, avg_success),
                        'related_concepts': [c['concept_id'] for c in concepts]
                    })
        
        return strengths 