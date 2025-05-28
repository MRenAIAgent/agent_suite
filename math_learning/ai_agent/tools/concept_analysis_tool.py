"""
Concept Analysis Tool for the Math Tutoring Agent.

This tool analyzes student understanding of mathematical concepts using the learning graph
and gap analyzer to provide detailed insights into knowledge gaps and strengths.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

# Fix import paths to match actual structure
from math_learning.learning_graph import LearningGraph
from math_learning.recommendation import GapAnalyzer


class ConceptAnalysisTool:
    """
    Tool for analyzing student understanding of mathematical concepts.
    
    This tool integrates with the learning graph and gap analyzer to provide
    comprehensive analysis of student knowledge, including:
    - Current mastery levels
    - Knowledge gaps and prerequisites
    - Learning progress trends
    - Recommendations for improvement
    """
    
    def __init__(self, learning_graph: LearningGraph, gap_analyzer: GapAnalyzer):
        """
        Initialize the Concept Analysis Tool.
        
        Args:
            learning_graph: Student's learning progress graph
            gap_analyzer: Knowledge gap analysis system
        """
        self.learning_graph = learning_graph
        self.gap_analyzer = gap_analyzer
        self.name = "concept_analysis"
        self.description = "Analyze student understanding of mathematical concepts and identify knowledge gaps"
        
    async def execute(self, student_id: str, concept: str, include_prerequisites: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute concept analysis for a student.
        
        Args:
            student_id: Unique identifier for the student
            concept: The mathematical concept to analyze
            include_prerequisites: Whether to include prerequisite analysis
            **kwargs: Additional parameters
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Get current mastery level for the concept
            mastery_level = self.learning_graph.get_mastery(concept, default=0.0)
            confidence = self.learning_graph.get_confidence(concept, default=0.5)
            
            # Analyze knowledge gaps
            gaps = []
            try:
                # Try to detect gaps using the gap analyzer
                detected_gaps = self.gap_analyzer.detect_critical_gaps(self.learning_graph)
                gaps = [{"concept_id": gap.concept_id, "severity": gap.severity, "reason": gap.reason} 
                       for gap in detected_gaps if hasattr(gap, 'concept_id')]
            except Exception as e:
                # If gap analysis fails, create a basic gap analysis
                gaps = [{"concept_id": concept, "severity": "medium", "reason": f"Analysis needed for {concept}"}]
            
            # Get mastered and struggling concepts
            mastered_concepts = self.learning_graph.get_mastered_concepts(threshold=0.7)
            struggling_concepts = self.learning_graph.get_struggling_concepts(threshold=0.3)
            
            # Analyze prerequisites if requested
            prerequisites_analysis = {}
            if include_prerequisites:
                # Basic prerequisite analysis based on concept naming patterns
                prerequisites_analysis = self._analyze_prerequisites(concept, mastered_concepts, struggling_concepts)
            
            # Generate learning recommendations
            recommendations = self._generate_recommendations(concept, mastery_level, gaps, struggling_concepts)
            
            # Create comprehensive analysis
            analysis = {
                "student_id": student_id,
                "concept": concept,
                "mastery_level": mastery_level,
                "confidence": confidence,
                "status": self._determine_status(mastery_level),
                "knowledge_gaps": gaps,
                "mastered_concepts": mastered_concepts,
                "struggling_concepts": struggling_concepts,
                "prerequisites_analysis": prerequisites_analysis,
                "recommendations": recommendations,
                "summary": self._generate_summary(concept, mastery_level, gaps, recommendations),
                "analyzed_at": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            return {
                "error": f"Failed to analyze concept {concept}: {str(e)}",
                "student_id": student_id,
                "concept": concept,
                "analyzed_at": datetime.now().isoformat()
            }
    
    def _analyze_prerequisites(self, concept: str, mastered: List[str], struggling: List[str]) -> Dict[str, Any]:
        """
        Analyze prerequisite concepts for the given concept.
        
        Args:
            concept: The concept to analyze prerequisites for
            mastered: List of mastered concept IDs
            struggling: List of struggling concept IDs
            
        Returns:
            Prerequisites analysis
        """
        # Basic prerequisite mapping based on common math concept relationships
        prerequisite_map = {
            "quadratic_equations": ["linear_equations", "factoring", "polynomials"],
            "calculus": ["algebra", "functions", "trigonometry"],
            "trigonometry": ["geometry", "algebra"],
            "fractions": ["basic_arithmetic", "division"],
            "algebra": ["basic_arithmetic", "fractions"],
            "geometry": ["basic_arithmetic", "measurement"]
        }
        
        prerequisites = prerequisite_map.get(concept, [])
        
        analysis = {
            "required_prerequisites": prerequisites,
            "mastered_prerequisites": [p for p in prerequisites if p in mastered],
            "missing_prerequisites": [p for p in prerequisites if p in struggling],
            "readiness_score": len([p for p in prerequisites if p in mastered]) / max(len(prerequisites), 1)
        }
        
        return analysis
    
    def _determine_status(self, mastery_level: float) -> str:
        """Determine learning status based on mastery level."""
        if mastery_level >= 0.8:
            return "mastered"
        elif mastery_level >= 0.6:
            return "proficient"
        elif mastery_level >= 0.4:
            return "developing"
        elif mastery_level >= 0.2:
            return "beginning"
        else:
            return "not_started"
    
    def _generate_recommendations(self, concept: str, mastery_level: float, 
                                gaps: List[Dict], struggling_concepts: List[str]) -> List[str]:
        """Generate learning recommendations based on analysis."""
        recommendations = []
        
        if mastery_level < 0.3:
            recommendations.append(f"Start with foundational practice in {concept}")
            recommendations.append("Focus on basic concepts and definitions")
        elif mastery_level < 0.6:
            recommendations.append(f"Continue practicing {concept} with guided exercises")
            recommendations.append("Work on connecting concepts to prior knowledge")
        elif mastery_level < 0.8:
            recommendations.append(f"Practice more challenging {concept} problems")
            recommendations.append("Focus on application and problem-solving")
        else:
            recommendations.append(f"Excellent mastery of {concept}!")
            recommendations.append("Ready to move on to advanced topics")
        
        # Add gap-specific recommendations
        if gaps:
            recommendations.append("Address identified knowledge gaps before proceeding")
            
        # Add recommendations for struggling concepts
        if struggling_concepts:
            recommendations.append(f"Review struggling areas: {', '.join(struggling_concepts[:3])}")
        
        return recommendations
    
    def _generate_summary(self, concept: str, mastery_level: float, 
                         gaps: List[Dict], recommendations: List[str]) -> str:
        """Generate a human-readable summary of the analysis."""
        status = self._determine_status(mastery_level)
        
        summary = f"Analysis of {concept}: Student shows {status} level understanding "
        summary += f"with {mastery_level:.1%} mastery. "
        
        if gaps:
            summary += f"Identified {len(gaps)} knowledge gaps that need attention. "
        
        if recommendations:
            summary += f"Key recommendation: {recommendations[0]}"
        
        return summary 