"""
Knowledge Gap Identifier

This module identifies knowledge gaps by mapping errors to concepts in the
knowledge graph and analyzing prerequisite relationships.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from math_learning.math_recognization.error_analysis_engine import ErrorAnalysisResult, ErrorType
from math_learning.math_recognization.root_cause_analyzer import RootCauseAnalysis
from math_learning.knowledge_graph.algebra_graph import build_algebra_knowledge_graph

class GapSeverity(Enum):
    """Severity levels for knowledge gaps."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"

class GapType(Enum):
    """Types of knowledge gaps."""
    PREREQUISITE = "prerequisite"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    APPLICATION = "application"

@dataclass
class KnowledgeGap:
    """Represents a specific knowledge gap."""
    concept_id: str
    concept_name: str
    gap_type: GapType
    severity: GapSeverity
    confidence: float
    evidence: List[str]
    prerequisite_gaps: List[str] = field(default_factory=list)
    blocking_concepts: List[str] = field(default_factory=list)
    remediation_estimate_hours: float = 0.0

@dataclass
class GapAnalysisResult:
    """Result of knowledge gap analysis."""
    identified_gaps: List[KnowledgeGap]
    gap_hierarchy: Dict[str, List[str]]
    learning_sequence: List[str]
    total_remediation_time: float
    priority_gaps: List[str]
    confidence_score: float

class ConceptMapper:
    """Maps errors and missing concepts to knowledge graph concepts."""
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.concept_keywords = self._build_concept_keyword_map()
    
    def _build_concept_keyword_map(self) -> Dict[str, List[str]]:
        """Build a map of keywords to concept IDs for better matching."""
        keyword_map = {}
        
        for concept_id, concept in self.knowledge_graph.concepts.items():
            keywords = []
            
            # Add concept name words
            name_words = concept.name.lower().split()
            keywords.extend(name_words)
            
            # Add description words
            if concept.description:
                desc_words = [word.strip('.,!?') for word in concept.description.lower().split()]
                keywords.extend(desc_words)
            
            # Add examples keywords
            if hasattr(concept, 'examples') and concept.examples:
                for example in concept.examples:
                    example_words = [word.strip('.,!?') for word in example.lower().split()]
                    keywords.extend(example_words)
            
            keyword_map[concept_id] = list(set(keywords))
        
        return keyword_map
    
    def map_error_to_concepts(self, error_analysis: ErrorAnalysisResult) -> List[str]:
        """Map an error analysis result to knowledge graph concepts."""
        mapped_concepts = []
        
        # Direct mapping from missing concepts
        for missing_concept in error_analysis.missing_concepts:
            concept_ids = self._find_matching_concepts(missing_concept)
            mapped_concepts.extend(concept_ids)
        
        # Mapping from error description
        description_concepts = self._find_concepts_from_text(error_analysis.description)
        mapped_concepts.extend(description_concepts)
        
        # Mapping from error subtype
        subtype_concepts = self._find_concepts_from_text(error_analysis.error_subtype)
        mapped_concepts.extend(subtype_concepts)
        
        return list(set(mapped_concepts))  # Remove duplicates
    
    def _find_matching_concepts(self, concept_name: str) -> List[str]:
        """Find knowledge graph concepts that match a given concept name."""
        matches = []
        concept_lower = concept_name.lower()
        
        # Direct name matching
        for concept_id, concept in self.knowledge_graph.concepts.items():
            if concept_lower in concept.name.lower() or concept.name.lower() in concept_lower:
                matches.append(concept_id)
        
        # Keyword matching
        for concept_id, keywords in self.concept_keywords.items():
            if any(keyword in concept_lower for keyword in keywords if len(keyword) > 3):
                matches.append(concept_id)
        
        return list(set(matches))
    
    def _find_concepts_from_text(self, text: str) -> List[str]:
        """Find concepts mentioned in a text description."""
        matches = []
        text_lower = text.lower()
        
        for concept_id, keywords in self.concept_keywords.items():
            # Look for significant keywords (length > 3 to avoid false positives)
            significant_keywords = [kw for kw in keywords if len(kw) > 3]
            if any(keyword in text_lower for keyword in significant_keywords):
                matches.append(concept_id)
        
        return matches

class GapSeverityAssessor:
    """Assesses the severity of knowledge gaps."""
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
    
    def assess_gap_severity(self, concept_id: str, error_analysis: ErrorAnalysisResult,
                          student_history: Dict[str, Any] = None) -> GapSeverity:
        """Assess the severity of a knowledge gap."""
        severity_factors = {
            'error_type_weight': self._get_error_type_weight(error_analysis.error_type),
            'concept_fundamentalness': self._assess_concept_fundamentalness(concept_id),
            'blocking_factor': self._assess_blocking_factor(concept_id),
            'frequency_factor': self._assess_frequency_factor(concept_id, student_history)
        }
        
        # Calculate weighted severity score
        weights = {
            'error_type_weight': 0.3,
            'concept_fundamentalness': 0.3,
            'blocking_factor': 0.25,
            'frequency_factor': 0.15
        }
        
        severity_score = sum(
            severity_factors[factor] * weights[factor] 
            for factor in severity_factors
        )
        
        # Map score to severity level
        if severity_score >= 0.8:
            return GapSeverity.CRITICAL
        elif severity_score >= 0.6:
            return GapSeverity.MAJOR
        elif severity_score >= 0.4:
            return GapSeverity.MODERATE
        else:
            return GapSeverity.MINOR
    
    def _get_error_type_weight(self, error_type: ErrorType) -> float:
        """Get severity weight based on error type."""
        weights = {
            ErrorType.CONCEPTUAL: 1.0,
            ErrorType.PROCEDURAL: 0.8,
            ErrorType.ARITHMETIC: 0.6,
            ErrorType.NOTATION: 0.4,
            ErrorType.CARELESS: 0.2
        }
        return weights.get(error_type, 0.5)
    
    def _assess_concept_fundamentalness(self, concept_id: str) -> float:
        """Assess how fundamental a concept is (based on difficulty and prerequisites)."""
        if concept_id not in self.knowledge_graph.concepts:
            return 0.5
        
        concept = self.knowledge_graph.concepts[concept_id]
        
        # Lower difficulty concepts are more fundamental
        difficulty_score = 1.0 - (concept.difficulty - 1) / 4.0  # Normalize to 0-1
        
        # Concepts with fewer prerequisites are more fundamental
        prerequisites = self.knowledge_graph.get_prerequisites(concept_id)
        prereq_score = 1.0 - min(len(prerequisites) / 5.0, 1.0)  # Normalize
        
        return (difficulty_score + prereq_score) / 2.0
    
    def _assess_blocking_factor(self, concept_id: str) -> float:
        """Assess how much this gap blocks learning other concepts."""
        # Count how many other concepts depend on this one
        dependents = []
        for other_concept_id in self.knowledge_graph.concepts:
            prerequisites = self.knowledge_graph.get_prerequisites(other_concept_id)
            if concept_id in prerequisites:
                dependents.append(other_concept_id)
        
        # Normalize blocking factor
        return min(len(dependents) / 10.0, 1.0)
    
    def _assess_frequency_factor(self, concept_id: str, student_history: Dict[str, Any]) -> float:
        """Assess how frequently this gap appears in student's history."""
        if not student_history or 'error_history' not in student_history:
            return 0.5  # Default if no history
        
        error_history = student_history['error_history']
        concept_errors = sum(1 for error in error_history if concept_id in error.get('related_concepts', []))
        total_errors = len(error_history)
        
        if total_errors == 0:
            return 0.5
        
        return min(concept_errors / total_errors, 1.0)

class LearningPathBuilder:
    """Builds optimal learning paths to address knowledge gaps."""
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
    
    def build_learning_sequence(self, gaps: List[KnowledgeGap]) -> List[str]:
        """Build an optimal learning sequence to address gaps."""
        # Create dependency graph for the gaps
        gap_concepts = [gap.concept_id for gap in gaps]
        dependency_graph = self._build_dependency_graph(gap_concepts)
        
        # Topological sort to get learning order
        learning_sequence = self._topological_sort(dependency_graph)
        
        return learning_sequence
    
    def _build_dependency_graph(self, concept_ids: List[str]) -> Dict[str, List[str]]:
        """Build a dependency graph for the given concepts."""
        dependency_graph = {concept_id: [] for concept_id in concept_ids}
        
        for concept_id in concept_ids:
            prerequisites = self.knowledge_graph.get_prerequisites(concept_id)
            # Only include prerequisites that are also in our gap list
            relevant_prereqs = [prereq for prereq in prerequisites if prereq in concept_ids]
            dependency_graph[concept_id] = relevant_prereqs
        
        return dependency_graph
    
    def _topological_sort(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort to determine learning order."""
        # Kahn's algorithm for topological sorting
        in_degree = {node: 0 for node in dependency_graph}
        
        # Calculate in-degrees
        for node in dependency_graph:
            for prereq in dependency_graph[node]:
                in_degree[node] += 1
        
        # Find nodes with no incoming edges
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            # Sort queue by priority (concepts with higher severity first)
            queue.sort()
            current = queue.pop(0)
            result.append(current)
            
            # Remove edges from current node
            for node in dependency_graph:
                if current in dependency_graph[node]:
                    dependency_graph[node].remove(current)
                    in_degree[node] -= 1
                    if in_degree[node] == 0:
                        queue.append(node)
        
        return result
    
    def estimate_remediation_time(self, gap: KnowledgeGap) -> float:
        """Estimate time needed to remediate a knowledge gap."""
        if gap.concept_id not in self.knowledge_graph.concepts:
            return 2.0  # Default estimate
        
        concept = self.knowledge_graph.concepts[gap.concept_id]
        base_time = concept.time_to_master / 60.0  # Convert to hours
        
        # Adjust based on gap severity
        severity_multipliers = {
            GapSeverity.MINOR: 0.5,
            GapSeverity.MODERATE: 1.0,
            GapSeverity.MAJOR: 1.5,
            GapSeverity.CRITICAL: 2.0
        }
        
        return base_time * severity_multipliers.get(gap.severity, 1.0)

class KnowledgeGapIdentifier:
    """Main class for identifying and analyzing knowledge gaps."""
    
    def __init__(self):
        self.knowledge_graph = build_algebra_knowledge_graph()
        self.concept_mapper = ConceptMapper(self.knowledge_graph)
        self.severity_assessor = GapSeverityAssessor(self.knowledge_graph)
        self.path_builder = LearningPathBuilder(self.knowledge_graph)
    
    def identify_gaps(self, error_analysis: ErrorAnalysisResult, 
                     root_cause_analysis: RootCauseAnalysis,
                     student_history: Dict[str, Any] = None) -> GapAnalysisResult:
        """
        Identify knowledge gaps from error and root cause analysis.
        
        Args:
            error_analysis: Result from error analysis engine
            root_cause_analysis: Result from root cause analyzer
            student_history: Historical performance data
            
        Returns:
            GapAnalysisResult with identified gaps and learning plan
        """
        # Map errors to knowledge graph concepts
        related_concepts = self.concept_mapper.map_error_to_concepts(error_analysis)
        
        # Add concepts from root cause analysis
        all_missing_concepts = (
            error_analysis.missing_concepts + 
            root_cause_analysis.prerequisite_gaps +
            related_concepts
        )
        
        # Create knowledge gap objects
        identified_gaps = []
        for concept_name in set(all_missing_concepts):
            # Find matching concept in knowledge graph
            matching_concepts = self.concept_mapper._find_matching_concepts(concept_name)
            
            for concept_id in matching_concepts:
                if concept_id in self.knowledge_graph.concepts:
                    gap = self._create_knowledge_gap(
                        concept_id, error_analysis, root_cause_analysis, student_history
                    )
                    identified_gaps.append(gap)
        
        # Remove duplicates and sort by severity
        unique_gaps = self._deduplicate_gaps(identified_gaps)
        unique_gaps.sort(key=lambda g: (g.severity.value, -g.confidence), reverse=True)
        
        # Build gap hierarchy
        gap_hierarchy = self._build_gap_hierarchy(unique_gaps)
        
        # Create learning sequence
        learning_sequence = self.path_builder.build_learning_sequence(unique_gaps)
        
        # Calculate total remediation time
        total_time = sum(
            self.path_builder.estimate_remediation_time(gap) 
            for gap in unique_gaps
        )
        
        # Identify priority gaps
        priority_gaps = [
            gap.concept_id for gap in unique_gaps 
            if gap.severity in [GapSeverity.CRITICAL, GapSeverity.MAJOR]
        ]
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(
            unique_gaps, error_analysis, root_cause_analysis
        )
        
        return GapAnalysisResult(
            identified_gaps=unique_gaps,
            gap_hierarchy=gap_hierarchy,
            learning_sequence=learning_sequence,
            total_remediation_time=total_time,
            priority_gaps=priority_gaps,
            confidence_score=confidence_score
        )
    
    def _create_knowledge_gap(self, concept_id: str, error_analysis: ErrorAnalysisResult,
                            root_cause_analysis: RootCauseAnalysis,
                            student_history: Dict[str, Any]) -> KnowledgeGap:
        """Create a KnowledgeGap object for a concept."""
        concept = self.knowledge_graph.concepts[concept_id]
        
        # Determine gap type
        gap_type = self._determine_gap_type(concept_id, error_analysis, root_cause_analysis)
        
        # Assess severity
        severity = self.severity_assessor.assess_gap_severity(
            concept_id, error_analysis, student_history
        )
        
        # Find prerequisite gaps
        prerequisites = self.knowledge_graph.get_prerequisites(concept_id)
        prerequisite_gaps = [
            prereq for prereq in prerequisites 
            if prereq in root_cause_analysis.prerequisite_gaps
        ]
        
        # Find blocking concepts
        blocking_concepts = [
            other_id for other_id in self.knowledge_graph.concepts
            if concept_id in self.knowledge_graph.get_prerequisites(other_id)
        ]
        
        # Estimate remediation time
        remediation_time = self.path_builder.estimate_remediation_time(
            KnowledgeGap(
                concept_id=concept_id,
                concept_name=concept.name,
                gap_type=gap_type,
                severity=severity,
                confidence=0.0,  # Placeholder
                evidence=[]  # Placeholder
            )
        )
        
        return KnowledgeGap(
            concept_id=concept_id,
            concept_name=concept.name,
            gap_type=gap_type,
            severity=severity,
            confidence=min(error_analysis.confidence + root_cause_analysis.confidence, 1.0),
            evidence=[error_analysis.description, root_cause_analysis.primary_cause],
            prerequisite_gaps=prerequisite_gaps,
            blocking_concepts=blocking_concepts[:5],  # Limit to top 5
            remediation_estimate_hours=remediation_time
        )
    
    def _determine_gap_type(self, concept_id: str, error_analysis: ErrorAnalysisResult,
                          root_cause_analysis: RootCauseAnalysis) -> GapType:
        """Determine the type of knowledge gap."""
        # Check if it's a prerequisite gap
        if concept_id in root_cause_analysis.prerequisite_gaps:
            return GapType.PREREQUISITE
        
        # Check error type
        if error_analysis.error_type == ErrorType.CONCEPTUAL:
            return GapType.CONCEPTUAL
        elif error_analysis.error_type == ErrorType.PROCEDURAL:
            return GapType.PROCEDURAL
        else:
            return GapType.APPLICATION
    
    def _deduplicate_gaps(self, gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """Remove duplicate gaps, keeping the one with highest confidence."""
        gap_dict = {}
        
        for gap in gaps:
            if gap.concept_id not in gap_dict or gap.confidence > gap_dict[gap.concept_id].confidence:
                gap_dict[gap.concept_id] = gap
        
        return list(gap_dict.values())
    
    def _build_gap_hierarchy(self, gaps: List[KnowledgeGap]) -> Dict[str, List[str]]:
        """Build a hierarchy showing prerequisite relationships among gaps."""
        hierarchy = {}
        gap_ids = [gap.concept_id for gap in gaps]
        
        for gap in gaps:
            # Find prerequisites that are also gaps
            prereq_gaps = [
                prereq for prereq in gap.prerequisite_gaps 
                if prereq in gap_ids
            ]
            hierarchy[gap.concept_id] = prereq_gaps
        
        return hierarchy
    
    def _calculate_overall_confidence(self, gaps: List[KnowledgeGap],
                                    error_analysis: ErrorAnalysisResult,
                                    root_cause_analysis: RootCauseAnalysis) -> float:
        """Calculate overall confidence in the gap analysis."""
        if not gaps:
            return 0.0
        
        # Base confidence from input analyses
        base_confidence = (error_analysis.confidence + root_cause_analysis.confidence) / 2
        
        # Adjust based on number of gaps identified
        if len(gaps) > 10:
            base_confidence *= 0.8  # Less confident with too many gaps
        elif len(gaps) < 3:
            base_confidence *= 0.9  # Slightly less confident with too few gaps
        
        # Adjust based on gap confidence scores
        avg_gap_confidence = sum(gap.confidence for gap in gaps) / len(gaps)
        combined_confidence = (base_confidence + avg_gap_confidence) / 2
        
        return min(combined_confidence, 1.0) 