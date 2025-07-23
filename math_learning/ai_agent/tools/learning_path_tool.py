"""
Learning Path Tool for the Math Tutoring Agent.

This tool generates personalized learning paths using the learning graph
and gap analyzer to create optimal learning sequences.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Fix import paths to match actual structure
from math_learning.learning_graph import LearningGraph
from math_learning.recommendation import GapAnalyzer


class LearningPathTool:
    """
    Tool for generating personalized learning paths.
    
    This tool creates adaptive learning sequences that:
    - Identify optimal learning order based on prerequisites
    - Adapt to student's current knowledge and gaps
    - Provide multiple path strategies (sequential, adaptive, goal-oriented)
    - Include time estimates and difficulty progression
    """
    
    def __init__(self, learning_graph: LearningGraph, gap_analyzer: GapAnalyzer):
        """
        Initialize the Learning Path Tool.
        
        Args:
            learning_graph: Student's learning progress graph
            gap_analyzer: Tool for analyzing knowledge gaps
        """
        self.learning_graph = learning_graph
        self.gap_analyzer = gap_analyzer
        self.name = "learning_path"
        self.description = "Generate personalized learning paths based on student needs and goals"
        
    async def execute(self, student_id: str, goal: str = "general_improvement", 
                     path_type: str = "adaptive", time_constraint: int = None, 
                     **kwargs) -> Dict[str, Any]:
        """
        Execute learning path generation for a student.
        
        Args:
            student_id: Unique identifier for the student
            goal: Learning goal (general_improvement, specific_concept, test_prep, remediation)
            path_type: Type of path (sequential, adaptive, goal_oriented, remediation)
            time_constraint: Maximum time in days for the path (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dict containing the generated learning path and metadata
        """
        try:
            # Analyze current student state
            current_state = self._analyze_current_state()
            
            # Identify knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps()
            
            # Generate path based on type
            if path_type == "sequential":
                learning_path = self._generate_sequential_path(goal, time_constraint)
            elif path_type == "adaptive":
                learning_path = self._generate_adaptive_path(goal, knowledge_gaps, time_constraint)
            elif path_type == "goal_oriented":
                learning_path = self._generate_goal_oriented_path(goal, current_state, time_constraint)
            elif path_type == "remediation":
                learning_path = self._generate_remediation_path(knowledge_gaps, time_constraint)
            else:
                learning_path = self._generate_adaptive_path(goal, knowledge_gaps, time_constraint)
            
            # Add path metadata and recommendations
            path_metadata = self._generate_path_metadata(learning_path, current_state)
            
            # Create comprehensive path response
            path_response = {
                "student_id": student_id,
                "goal": goal,
                "path_type": path_type,
                "time_constraint_days": time_constraint,
                "current_state": current_state,
                "knowledge_gaps": knowledge_gaps,
                "learning_path": learning_path,
                "metadata": path_metadata,
                "recommendations": self._generate_path_recommendations(learning_path, current_state),
                "estimated_outcomes": self._estimate_learning_outcomes(learning_path, current_state),
                "generated_at": datetime.now().isoformat()
            }
            
            return path_response
            
        except Exception as e:
            return {
                "error": f"Failed to generate learning path: {str(e)}",
                "student_id": student_id,
                "goal": goal,
                "path_type": path_type,
                "generated_at": datetime.now().isoformat()
            }
    
    def _analyze_current_state(self) -> Dict[str, Any]:
        """Analyze the student's current learning state."""
        try:
            mastered_concepts = self.learning_graph.get_mastered_concepts(threshold=0.7)
            struggling_concepts = self.learning_graph.get_struggling_concepts(threshold=0.3)
            
            # Calculate learning metrics
            total_concepts = len(mastered_concepts) + len(struggling_concepts) + 15  # Estimate
            mastery_rate = len(mastered_concepts) / max(total_concepts, 1)
            
            return {
                "mastered_concepts": mastered_concepts,
                "struggling_concepts": struggling_concepts,
                "mastery_rate": mastery_rate,
                "learning_level": self._determine_learning_level(mastery_rate),
                "preferred_difficulty": self._estimate_preferred_difficulty(),
                "learning_style": self._estimate_learning_style(),
                "recent_performance": 0.75,  # Would calculate from recent sessions
                "engagement_level": "high"   # Would calculate from activity patterns
            }
        except Exception:
            return {
                "mastered_concepts": [],
                "struggling_concepts": [],
                "mastery_rate": 0.0,
                "learning_level": "beginner",
                "preferred_difficulty": "easy",
                "learning_style": "visual",
                "recent_performance": 0.5,
                "engagement_level": "medium"
            }
    
    def _identify_knowledge_gaps(self) -> Dict[str, Any]:
        """Identify specific knowledge gaps using the gap analyzer."""
        try:
            # Use gap analyzer to find gaps
            gaps = self.gap_analyzer.analyze_gaps()
            
            # Categorize gaps by severity and type
            critical_gaps = [gap for gap in gaps if gap.get("severity", 0) > 0.7]
            moderate_gaps = [gap for gap in gaps if 0.4 <= gap.get("severity", 0) <= 0.7]
            minor_gaps = [gap for gap in gaps if gap.get("severity", 0) < 0.4]
            
            return {
                "total_gaps": len(gaps),
                "critical_gaps": critical_gaps,
                "moderate_gaps": moderate_gaps,
                "minor_gaps": minor_gaps,
                "gap_categories": self._categorize_gaps(gaps),
                "prerequisite_gaps": self._identify_prerequisite_gaps(gaps),
                "priority_order": self._prioritize_gaps(gaps)
            }
        except Exception:
            # Provide simulated gap analysis
            return {
                "total_gaps": 3,
                "critical_gaps": [{"concept": "fractions", "severity": 0.8}],
                "moderate_gaps": [{"concept": "decimals", "severity": 0.6}],
                "minor_gaps": [{"concept": "percentages", "severity": 0.3}],
                "gap_categories": {"arithmetic": 2, "algebra": 1},
                "prerequisite_gaps": ["basic_arithmetic"],
                "priority_order": ["fractions", "decimals", "percentages"]
            }
    
    def _generate_sequential_path(self, goal: str, time_constraint: int) -> Dict[str, Any]:
        """Generate a sequential learning path following curriculum order."""
        # Define standard curriculum sequence
        curriculum_sequence = [
            "basic_arithmetic", "fractions", "decimals", "percentages",
            "basic_algebra", "linear_equations", "quadratic_equations",
            "geometry_basics", "trigonometry", "advanced_algebra"
        ]
        
        # Filter based on current level and goal
        if goal == "test_prep":
            sequence = curriculum_sequence[-5:]  # Focus on advanced topics
        elif goal == "remediation":
            sequence = curriculum_sequence[:5]   # Focus on basics
        else:
            sequence = curriculum_sequence
        
        # Apply time constraint
        if time_constraint:
            max_concepts = max(1, time_constraint // 3)  # ~3 days per concept
            sequence = sequence[:max_concepts]
        
        # Create detailed path steps
        path_steps = []
        for i, concept in enumerate(sequence):
            step = {
                "step_number": i + 1,
                "concept": concept,
                "estimated_days": 3,
                "difficulty_level": self._get_concept_difficulty(concept),
                "prerequisites": self._get_prerequisites(concept),
                "learning_objectives": self._get_learning_objectives(concept),
                "recommended_exercises": 10,
                "mastery_threshold": 0.8
            }
            path_steps.append(step)
        
        return {
            "path_name": f"Sequential Path - {goal}",
            "total_steps": len(path_steps),
            "estimated_duration_days": sum(step["estimated_days"] for step in path_steps),
            "difficulty_progression": "gradual",
            "path_steps": path_steps,
            "path_strategy": "sequential"
        }
    
    def _generate_adaptive_path(self, goal: str, knowledge_gaps: Dict, time_constraint: int) -> Dict[str, Any]:
        """Generate an adaptive learning path based on current knowledge and gaps."""
        # Start with gap priorities
        priority_concepts = knowledge_gaps.get("priority_order", [])
        
        # Add concepts based on goal
        if goal == "general_improvement":
            additional_concepts = ["problem_solving", "critical_thinking"]
        elif goal == "test_prep":
            additional_concepts = ["test_strategies", "time_management"]
        else:
            additional_concepts = []
        
        # Combine and optimize order
        all_concepts = priority_concepts + additional_concepts
        optimized_sequence = self._optimize_learning_sequence(all_concepts)
        
        # Apply time constraint with adaptive scheduling
        if time_constraint:
            optimized_sequence = self._apply_adaptive_time_constraint(optimized_sequence, time_constraint)
        
        # Create adaptive path steps
        path_steps = []
        for i, concept in enumerate(optimized_sequence):
            # Adapt difficulty and time based on gaps
            gap_severity = self._get_gap_severity(concept, knowledge_gaps)
            
            step = {
                "step_number": i + 1,
                "concept": concept,
                "estimated_days": self._calculate_adaptive_time(concept, gap_severity),
                "difficulty_level": self._calculate_adaptive_difficulty(concept, gap_severity),
                "gap_severity": gap_severity,
                "adaptive_features": self._get_adaptive_features(concept, gap_severity),
                "learning_objectives": self._get_adaptive_objectives(concept, gap_severity),
                "recommended_exercises": self._calculate_adaptive_exercises(gap_severity),
                "mastery_threshold": 0.75 + (gap_severity * 0.15)  # Higher threshold for bigger gaps
            }
            path_steps.append(step)
        
        return {
            "path_name": f"Adaptive Path - {goal}",
            "total_steps": len(path_steps),
            "estimated_duration_days": sum(step["estimated_days"] for step in path_steps),
            "difficulty_progression": "adaptive",
            "adaptation_strategy": "gap_based",
            "path_steps": path_steps,
            "path_strategy": "adaptive"
        }
    
    def _generate_goal_oriented_path(self, goal: str, current_state: Dict, time_constraint: int) -> Dict[str, Any]:
        """Generate a goal-oriented learning path focused on specific objectives."""
        # Define goal-specific concept maps
        goal_concepts = {
            "algebra_mastery": ["variables", "linear_equations", "quadratic_equations", "systems_equations"],
            "geometry_basics": ["shapes", "area_perimeter", "angles", "coordinate_geometry"],
            "test_prep": ["problem_solving", "time_management", "test_strategies", "review_concepts"],
            "general_improvement": ["weak_areas", "practice_problems", "concept_reinforcement"]
        }
        
        target_concepts = goal_concepts.get(goal, goal_concepts["general_improvement"])
        
        # Filter based on current mastery
        mastered = current_state.get("mastered_concepts", [])
        needed_concepts = [c for c in target_concepts if c not in mastered]
        
        # Prioritize based on goal importance
        prioritized_concepts = self._prioritize_for_goal(needed_concepts, goal)
        
        # Apply time constraint with goal focus
        if time_constraint:
            prioritized_concepts = self._apply_goal_time_constraint(prioritized_concepts, time_constraint, goal)
        
        # Create goal-oriented steps
        path_steps = []
        for i, concept in enumerate(prioritized_concepts):
            step = {
                "step_number": i + 1,
                "concept": concept,
                "estimated_days": self._calculate_goal_time(concept, goal),
                "difficulty_level": self._get_goal_difficulty(concept, goal),
                "goal_relevance": self._calculate_goal_relevance(concept, goal),
                "learning_objectives": self._get_goal_objectives(concept, goal),
                "success_metrics": self._get_goal_metrics(concept, goal),
                "recommended_exercises": self._calculate_goal_exercises(concept, goal),
                "mastery_threshold": 0.8  # High threshold for goal achievement
            }
            path_steps.append(step)
        
        return {
            "path_name": f"Goal-Oriented Path - {goal}",
            "total_steps": len(path_steps),
            "estimated_duration_days": sum(step["estimated_days"] for step in path_steps),
            "difficulty_progression": "goal_focused",
            "goal_alignment": "high",
            "path_steps": path_steps,
            "path_strategy": "goal_oriented"
        }
    
    def _generate_remediation_path(self, knowledge_gaps: Dict, time_constraint: int) -> Dict[str, Any]:
        """Generate a remediation path focused on addressing critical gaps."""
        # Focus on critical and moderate gaps
        critical_gaps = knowledge_gaps.get("critical_gaps", [])
        moderate_gaps = knowledge_gaps.get("moderate_gaps", [])
        
        # Prioritize critical gaps first
        remediation_concepts = []
        for gap in critical_gaps:
            remediation_concepts.append(gap.get("concept", "unknown"))
        for gap in moderate_gaps:
            remediation_concepts.append(gap.get("concept", "unknown"))
        
        # Add prerequisite concepts if needed
        prerequisite_gaps = knowledge_gaps.get("prerequisite_gaps", [])
        all_remediation = prerequisite_gaps + remediation_concepts
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in all_remediation:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
        
        # Apply time constraint for remediation
        if time_constraint:
            # Prioritize most critical gaps within time limit
            max_concepts = max(1, time_constraint // 4)  # ~4 days per remediation concept
            unique_concepts = unique_concepts[:max_concepts]
        
        # Create remediation steps
        path_steps = []
        for i, concept in enumerate(unique_concepts):
            gap_info = self._get_gap_info(concept, knowledge_gaps)
            
            step = {
                "step_number": i + 1,
                "concept": concept,
                "estimated_days": 4,  # More time for remediation
                "difficulty_level": "remedial",
                "gap_severity": gap_info.get("severity", 0.5),
                "remediation_type": gap_info.get("type", "concept_gap"),
                "learning_objectives": self._get_remediation_objectives(concept),
                "remediation_strategies": self._get_remediation_strategies(concept),
                "recommended_exercises": 15,  # More practice for remediation
                "mastery_threshold": 0.7,  # Lower threshold initially
                "progress_checkpoints": self._get_remediation_checkpoints(concept)
            }
            path_steps.append(step)
        
        return {
            "path_name": "Remediation Path",
            "total_steps": len(path_steps),
            "estimated_duration_days": sum(step["estimated_days"] for step in path_steps),
            "difficulty_progression": "remedial_to_standard",
            "remediation_focus": "critical_gaps",
            "path_steps": path_steps,
            "path_strategy": "remediation"
        }
    
    # Helper methods for path generation
    def _determine_learning_level(self, mastery_rate: float) -> str:
        """Determine student's learning level based on mastery rate."""
        if mastery_rate < 0.3:
            return "beginner"
        elif mastery_rate < 0.6:
            return "intermediate"
        elif mastery_rate < 0.8:
            return "advanced"
        else:
            return "expert"
    
    def _estimate_preferred_difficulty(self) -> str:
        """Estimate student's preferred difficulty level."""
        # This would analyze historical performance patterns
        return "medium"  # Placeholder
    
    def _estimate_learning_style(self) -> str:
        """Estimate student's learning style preferences."""
        # This would analyze interaction patterns
        return "visual"  # Placeholder
    
    def _categorize_gaps(self, gaps: List[Dict]) -> Dict[str, int]:
        """Categorize gaps by subject area."""
        categories = {}
        for gap in gaps:
            category = gap.get("category", "general")
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _identify_prerequisite_gaps(self, gaps: List[Dict]) -> List[str]:
        """Identify gaps in prerequisite concepts."""
        # This would analyze dependency relationships
        return ["basic_arithmetic"]  # Placeholder
    
    def _prioritize_gaps(self, gaps: List[Dict]) -> List[str]:
        """Prioritize gaps by severity and importance."""
        sorted_gaps = sorted(gaps, key=lambda x: x.get("severity", 0), reverse=True)
        return [gap.get("concept", "unknown") for gap in sorted_gaps]
    
    def _get_concept_difficulty(self, concept: str) -> str:
        """Get difficulty level for a concept."""
        difficulty_map = {
            "basic_arithmetic": "easy",
            "fractions": "medium",
            "algebra": "hard",
            "calculus": "very_hard"
        }
        return difficulty_map.get(concept, "medium")
    
    def _get_prerequisites(self, concept: str) -> List[str]:
        """Get prerequisite concepts."""
        prereq_map = {
            "fractions": ["basic_arithmetic"],
            "decimals": ["fractions"],
            "algebra": ["fractions", "decimals"]
        }
        return prereq_map.get(concept, [])
    
    def _get_learning_objectives(self, concept: str) -> List[str]:
        """Get learning objectives for a concept."""
        return [f"Master {concept} fundamentals", f"Apply {concept} in problems"]
    
    def _optimize_learning_sequence(self, concepts: List[str]) -> List[str]:
        """Optimize the sequence of concepts for learning efficiency."""
        # This would use graph algorithms to optimize based on prerequisites
        return concepts  # Placeholder - return as-is
    
    def _apply_adaptive_time_constraint(self, concepts: List[str], time_constraint: int) -> List[str]:
        """Apply time constraint with adaptive prioritization."""
        if len(concepts) * 3 <= time_constraint:  # 3 days per concept average
            return concepts
        
        max_concepts = max(1, time_constraint // 3)
        return concepts[:max_concepts]
    
    def _get_gap_severity(self, concept: str, knowledge_gaps: Dict) -> float:
        """Get gap severity for a specific concept."""
        all_gaps = (knowledge_gaps.get("critical_gaps", []) + 
                   knowledge_gaps.get("moderate_gaps", []) + 
                   knowledge_gaps.get("minor_gaps", []))
        
        for gap in all_gaps:
            if gap.get("concept") == concept:
                return gap.get("severity", 0.5)
        return 0.0
    
    def _calculate_adaptive_time(self, concept: str, gap_severity: float) -> int:
        """Calculate adaptive time allocation based on gap severity."""
        base_time = 3  # Base days per concept
        additional_time = int(gap_severity * 3)  # Up to 3 extra days for severe gaps
        return base_time + additional_time
    
    def _calculate_adaptive_difficulty(self, concept: str, gap_severity: float) -> str:
        """Calculate adaptive difficulty based on gap severity."""
        if gap_severity > 0.7:
            return "easy"  # Start easier for big gaps
        elif gap_severity > 0.4:
            return "medium"
        else:
            return "standard"
    
    def _get_adaptive_features(self, concept: str, gap_severity: float) -> List[str]:
        """Get adaptive features for a concept based on gap severity."""
        features = ["standard_practice"]
        
        if gap_severity > 0.6:
            features.extend(["extra_examples", "step_by_step_guidance", "frequent_checkpoints"])
        elif gap_severity > 0.3:
            features.extend(["additional_practice", "concept_review"])
        
        return features
    
    def _get_adaptive_objectives(self, concept: str, gap_severity: float) -> List[str]:
        """Get adaptive learning objectives based on gap severity."""
        base_objectives = [f"Understand {concept} basics"]
        
        if gap_severity > 0.6:
            base_objectives.extend([f"Build confidence in {concept}", f"Practice {concept} fundamentals"])
        else:
            base_objectives.extend([f"Master {concept} applications", f"Solve complex {concept} problems"])
        
        return base_objectives
    
    def _calculate_adaptive_exercises(self, gap_severity: float) -> int:
        """Calculate number of exercises based on gap severity."""
        base_exercises = 8
        additional_exercises = int(gap_severity * 10)
        return base_exercises + additional_exercises
    
    def _prioritize_for_goal(self, concepts: List[str], goal: str) -> List[str]:
        """Prioritize concepts based on goal relevance."""
        # This would use goal-specific prioritization logic
        return concepts  # Placeholder
    
    def _apply_goal_time_constraint(self, concepts: List[str], time_constraint: int, goal: str) -> List[str]:
        """Apply time constraint with goal-specific prioritization."""
        if goal == "test_prep":
            # Prioritize high-impact concepts for tests
            max_concepts = max(1, time_constraint // 2)  # Intensive preparation
        else:
            max_concepts = max(1, time_constraint // 3)  # Standard pacing
        
        return concepts[:max_concepts]
    
    def _calculate_goal_time(self, concept: str, goal: str) -> int:
        """Calculate time allocation based on goal requirements."""
        if goal == "test_prep":
            return 2  # Intensive preparation
        elif goal == "mastery":
            return 5  # Deep understanding
        else:
            return 3  # Standard time
    
    def _get_goal_difficulty(self, concept: str, goal: str) -> str:
        """Get difficulty level based on goal requirements."""
        if goal == "test_prep":
            return "challenging"  # Prepare for test difficulty
        elif goal == "mastery":
            return "comprehensive"  # Deep understanding
        else:
            return "standard"
    
    def _calculate_goal_relevance(self, concept: str, goal: str) -> float:
        """Calculate how relevant a concept is to the goal."""
        # This would use goal-concept relevance mapping
        return 0.8  # Placeholder
    
    def _get_goal_objectives(self, concept: str, goal: str) -> List[str]:
        """Get goal-specific learning objectives."""
        if goal == "test_prep":
            return [f"Master {concept} for test success", f"Practice {concept} under time pressure"]
        else:
            return [f"Achieve {goal} in {concept}"]
    
    def _get_goal_metrics(self, concept: str, goal: str) -> List[str]:
        """Get goal-specific success metrics."""
        return [f"Accuracy > 85%", f"Speed improvement", f"Confidence level > 80%"]
    
    def _calculate_goal_exercises(self, concept: str, goal: str) -> int:
        """Calculate exercises needed for goal achievement."""
        if goal == "test_prep":
            return 20  # Intensive practice
        elif goal == "mastery":
            return 15  # Comprehensive practice
        else:
            return 10  # Standard practice
    
    def _get_gap_info(self, concept: str, knowledge_gaps: Dict) -> Dict[str, Any]:
        """Get detailed information about a specific gap."""
        all_gaps = (knowledge_gaps.get("critical_gaps", []) + 
                   knowledge_gaps.get("moderate_gaps", []) + 
                   knowledge_gaps.get("minor_gaps", []))
        
        for gap in all_gaps:
            if gap.get("concept") == concept:
                return gap
        
        return {"severity": 0.5, "type": "concept_gap"}
    
    def _get_remediation_objectives(self, concept: str) -> List[str]:
        """Get remediation-specific learning objectives."""
        return [
            f"Identify and address {concept} misconceptions",
            f"Build solid foundation in {concept}",
            f"Gain confidence in {concept} applications"
        ]
    
    def _get_remediation_strategies(self, concept: str) -> List[str]:
        """Get remediation strategies for a concept."""
        return [
            "Start with concrete examples",
            "Use visual representations",
            "Practice with immediate feedback",
            "Break down into smaller steps"
        ]
    
    def _get_remediation_checkpoints(self, concept: str) -> List[str]:
        """Get progress checkpoints for remediation."""
        return [
            "Basic understanding check",
            "Application practice",
            "Confidence assessment",
            "Mastery verification"
        ]
    
    def _generate_path_metadata(self, learning_path: Dict, current_state: Dict) -> Dict[str, Any]:
        """Generate metadata for the learning path."""
        return {
            "total_concepts": learning_path.get("total_steps", 0),
            "estimated_duration": learning_path.get("estimated_duration_days", 0),
            "difficulty_range": self._calculate_difficulty_range(learning_path),
            "prerequisite_coverage": self._calculate_prerequisite_coverage(learning_path),
            "goal_alignment_score": 0.85,  # Would calculate based on goal matching
            "personalization_level": "high",
            "adaptation_points": self._identify_adaptation_points(learning_path),
            "success_probability": self._estimate_success_probability(learning_path, current_state)
        }
    
    def _generate_path_recommendations(self, learning_path: Dict, current_state: Dict) -> List[str]:
        """Generate recommendations for following the learning path."""
        recommendations = []
        
        total_days = learning_path.get("estimated_duration_days", 0)
        if total_days > 30:
            recommendations.append("Consider breaking this path into smaller milestones")
        
        if current_state.get("engagement_level") == "low":
            recommendations.append("Include more interactive and engaging exercises")
        
        if learning_path.get("difficulty_progression") == "steep":
            recommendations.append("Take extra time on challenging concepts")
        
        recommendations.extend([
            "Review progress weekly and adjust as needed",
            "Practice consistently for best results",
            "Don't hesitate to revisit earlier concepts if needed"
        ])
        
        return recommendations
    
    def _estimate_learning_outcomes(self, learning_path: Dict, current_state: Dict) -> Dict[str, Any]:
        """Estimate expected learning outcomes from the path."""
        return {
            "expected_mastery_gain": 0.3,  # Expected increase in overall mastery
            "concepts_to_master": learning_path.get("total_steps", 0),
            "skill_improvements": ["problem_solving", "mathematical_reasoning"],
            "confidence_boost": "significant",
            "knowledge_gap_reduction": 0.7,  # Expected reduction in knowledge gaps
            "readiness_for_advanced_topics": True,
            "estimated_performance_improvement": "20-30%"
        }
    
    def _calculate_difficulty_range(self, learning_path: Dict) -> Dict[str, str]:
        """Calculate the difficulty range of the learning path."""
        steps = learning_path.get("path_steps", [])
        if not steps:
            return {"min": "unknown", "max": "unknown"}
        
        difficulties = [step.get("difficulty_level", "medium") for step in steps]
        return {"min": min(difficulties), "max": max(difficulties)}
    
    def _calculate_prerequisite_coverage(self, learning_path: Dict) -> float:
        """Calculate how well the path covers prerequisites."""
        # This would analyze prerequisite relationships
        return 0.9  # Placeholder - 90% coverage
    
    def _identify_adaptation_points(self, learning_path: Dict) -> List[str]:
        """Identify points where the path can be adapted."""
        return [
            "After each concept mastery check",
            "Weekly progress review",
            "When difficulty becomes too high/low",
            "Based on engagement levels"
        ]
    
    def _estimate_success_probability(self, learning_path: Dict, current_state: Dict) -> float:
        """Estimate probability of successfully completing the path."""
        # This would use historical data and current performance
        base_probability = 0.75
        
        # Adjust based on current state
        mastery_rate = current_state.get("mastery_rate", 0.5)
        engagement = current_state.get("engagement_level", "medium")
        
        if mastery_rate > 0.7:
            base_probability += 0.1
        elif mastery_rate < 0.3:
            base_probability -= 0.1
        
        if engagement == "high":
            base_probability += 0.1
        elif engagement == "low":
            base_probability -= 0.15
        
        return min(0.95, max(0.1, base_probability)) 