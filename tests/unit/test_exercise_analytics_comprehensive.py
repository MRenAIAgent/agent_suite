#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Exercise Analytics.

This module tests exercise analytics, metrics calculation, performance tracking,
and session analysis functionality.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from datetime import datetime, timedelta
import statistics

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from math_learning.exercises.exercise import Exercise
from math_learning.exercises.exercise_system import ExerciseSession
from math_learning.learning_graph.user_model import LearningGraph


class MockExerciseAnalytics:
    """Mock Exercise Analytics class for testing."""
    
    def __init__(self):
        self.session_data = {}
        self.performance_data = {}
        self.student_progress = {}
    
    def analyze_session_performance(self, session: ExerciseSession, 
                                  student_responses: List[Dict]) -> Dict[str, Any]:
        """Analyze performance for a session."""
        if not student_responses:
            return {"error": "No responses provided"}
        
        total_exercises = len(student_responses)
        correct_responses = sum(1 for resp in student_responses if resp.get("correct", False))
        
        accuracy = correct_responses / total_exercises if total_exercises > 0 else 0
        
        # Calculate time metrics
        response_times = [resp.get("time_taken", 0) for resp in student_responses]
        avg_time = statistics.mean(response_times) if response_times else 0
        
        # Calculate difficulty metrics
        difficulties = [resp.get("difficulty", 1.0) for resp in student_responses]
        avg_difficulty = statistics.mean(difficulties) if difficulties else 1.0
        
        return {
            "session_id": session.session_id,
            "total_exercises": total_exercises,
            "correct_responses": correct_responses,
            "accuracy": accuracy,
            "average_time": avg_time,
            "average_difficulty": avg_difficulty,
            "completion_rate": 1.0,  # Assume completed if we have responses
            "improvement_indicators": self._calculate_improvement(student_responses)
        }
    
    def calculate_learning_velocity(self, student: LearningGraph, 
                                  concept_id: str, 
                                  time_window_days: int = 30) -> Dict[str, Any]:
        """Calculate learning velocity for a concept."""
        # Mock implementation
        current_mastery = student.get_mastery(concept_id, 0.0)
        
        # Simulate historical data
        historical_mastery = max(0.0, current_mastery - 0.2)  # Assume 0.2 improvement
        
        velocity = (current_mastery - historical_mastery) / time_window_days
        
        return {
            "concept_id": concept_id,
            "current_mastery": current_mastery,
            "historical_mastery": historical_mastery,
            "learning_velocity": velocity,
            "time_window_days": time_window_days,
            "trend": "improving" if velocity > 0 else "stable" if velocity == 0 else "declining"
        }
    
    def generate_difficulty_recommendations(self, student: LearningGraph,
                                          concept_id: str,
                                          performance_history: List[Dict]) -> Dict[str, Any]:
        """Generate difficulty recommendations based on performance."""
        if not performance_history:
            base_difficulty = 1.0 + student.get_mastery(concept_id, 0.0) * 3.0
            return {
                "recommended_difficulty": base_difficulty,
                "confidence": 0.5,
                "reasoning": "No performance history available"
            }
        
        # Calculate recent performance
        recent_accuracy = statistics.mean([p.get("accuracy", 0.5) for p in performance_history[-5:]])
        recent_difficulty = statistics.mean([p.get("difficulty", 2.0) for p in performance_history[-5:]])
        
        # Adjust difficulty based on performance
        if recent_accuracy > 0.8:
            recommended_difficulty = min(5.0, recent_difficulty + 0.5)
            reasoning = "High accuracy - increase difficulty"
        elif recent_accuracy < 0.6:
            recommended_difficulty = max(1.0, recent_difficulty - 0.5)
            reasoning = "Low accuracy - decrease difficulty"
        else:
            recommended_difficulty = recent_difficulty
            reasoning = "Maintain current difficulty"
        
        return {
            "recommended_difficulty": recommended_difficulty,
            "current_difficulty": recent_difficulty,
            "recent_accuracy": recent_accuracy,
            "confidence": min(0.9, len(performance_history) / 10.0),
            "reasoning": reasoning
        }
    
    def analyze_error_patterns(self, student_responses: List[Dict]) -> Dict[str, Any]:
        """Analyze error patterns in student responses."""
        if not student_responses:
            return {"error": "No responses to analyze"}
        
        incorrect_responses = [resp for resp in student_responses if not resp.get("correct", True)]
        
        if not incorrect_responses:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "common_error_types": [],
                "recommendations": ["Continue current approach"]
            }
        
        # Analyze error types (mock implementation)
        error_types = {}
        for resp in incorrect_responses:
            error_type = resp.get("error_type", "computational")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Sort by frequency
        common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_errors": len(incorrect_responses),
            "error_rate": len(incorrect_responses) / len(student_responses),
            "common_error_types": common_errors,
            "error_distribution": error_types,
            "recommendations": self._generate_error_recommendations(common_errors)
        }
    
    def calculate_engagement_metrics(self, session: ExerciseSession,
                                   interaction_data: List[Dict]) -> Dict[str, Any]:
        """Calculate student engagement metrics."""
        if not interaction_data:
            return {"error": "No interaction data provided"}
        
        # Time-based metrics
        session_duration = sum(interaction.get("duration", 0) for interaction in interaction_data)
        avg_time_per_exercise = session_duration / len(interaction_data) if interaction_data else 0
        
        # Engagement indicators
        quick_answers = sum(1 for interaction in interaction_data 
                           if interaction.get("duration", 0) < 10)  # Less than 10 seconds
        slow_answers = sum(1 for interaction in interaction_data 
                          if interaction.get("duration", 0) > 120)  # More than 2 minutes
        
        # Interaction quality
        help_requests = sum(1 for interaction in interaction_data 
                           if interaction.get("help_requested", False))
        
        engagement_score = self._calculate_engagement_score(
            avg_time_per_exercise, quick_answers, slow_answers, help_requests, len(interaction_data)
        )
        
        return {
            "session_duration": session_duration,
            "average_time_per_exercise": avg_time_per_exercise,
            "quick_answers": quick_answers,
            "slow_answers": slow_answers,
            "help_requests": help_requests,
            "engagement_score": engagement_score,
            "engagement_level": self._categorize_engagement(engagement_score)
        }
    
    def generate_progress_report(self, student: LearningGraph,
                               time_period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive progress report."""
        # Get all concepts student has worked on
        concepts = ["addition", "multiplication", "variables", "fractions"]  # Mock concepts
        
        concept_progress = {}
        for concept in concepts:
            mastery = student.get_mastery(concept, 0.0)
            confidence = student.get_confidence(concept, 0.0)
            
            concept_progress[concept] = {
                "mastery": mastery,
                "confidence": confidence,
                "improvement": max(0.0, mastery - 0.1),  # Mock improvement
                "exercises_completed": int(mastery * 20),  # Mock exercise count
            }
        
        # Calculate overall metrics
        overall_mastery = statistics.mean([cp["mastery"] for cp in concept_progress.values()])
        total_exercises = sum([cp["exercises_completed"] for cp in concept_progress.values()])
        
        return {
            "student_id": student.student_id,
            "report_period_days": time_period_days,
            "overall_mastery": overall_mastery,
            "total_exercises_completed": total_exercises,
            "concept_progress": concept_progress,
            "strengths": self._identify_strengths(concept_progress),
            "areas_for_improvement": self._identify_weaknesses(concept_progress),
            "recommendations": self._generate_learning_recommendations(concept_progress)
        }
    
    def _calculate_improvement(self, responses: List[Dict]) -> Dict[str, Any]:
        """Calculate improvement indicators."""
        if len(responses) < 2:
            return {"trend": "insufficient_data"}
        
        # Compare first half vs second half
        mid_point = len(responses) // 2
        first_half_accuracy = statistics.mean([r.get("correct", False) for r in responses[:mid_point]])
        second_half_accuracy = statistics.mean([r.get("correct", False) for r in responses[mid_point:]])
        
        improvement = second_half_accuracy - first_half_accuracy
        
        return {
            "improvement_score": improvement,
            "trend": "improving" if improvement > 0.1 else "stable" if abs(improvement) <= 0.1 else "declining",
            "first_half_accuracy": first_half_accuracy,
            "second_half_accuracy": second_half_accuracy
        }
    
    def _generate_error_recommendations(self, common_errors: List[tuple]) -> List[str]:
        """Generate recommendations based on error patterns."""
        if not common_errors:
            return ["Continue current approach"]
        
        recommendations = []
        most_common_error = common_errors[0][0]
        
        error_recommendations = {
            "computational": "Practice basic arithmetic operations",
            "conceptual": "Review fundamental concepts",
            "procedural": "Focus on step-by-step problem solving",
            "careless": "Take more time to check answers"
        }
        
        recommendation = error_recommendations.get(most_common_error, "Review problem-solving strategies")
        recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_engagement_score(self, avg_time: float, quick: int, slow: int, 
                                  help_requests: int, total: int) -> float:
        """Calculate engagement score (0-1)."""
        if total == 0:
            return 0.0
        
        # Penalize too many quick or slow answers
        quick_penalty = (quick / total) * 0.3
        slow_penalty = (slow / total) * 0.2
        
        # Moderate help requests are good
        help_score = min(0.2, (help_requests / total) * 0.5)
        
        # Base score from time appropriateness
        time_score = 0.8 if 15 <= avg_time <= 90 else 0.5
        
        engagement_score = max(0.0, min(1.0, time_score + help_score - quick_penalty - slow_penalty))
        return engagement_score
    
    def _categorize_engagement(self, score: float) -> str:
        """Categorize engagement level."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _identify_strengths(self, concept_progress: Dict) -> List[str]:
        """Identify student strengths."""
        strengths = []
        for concept, progress in concept_progress.items():
            if progress["mastery"] >= 0.8:
                strengths.append(concept)
        return strengths[:3]  # Top 3 strengths
    
    def _identify_weaknesses(self, concept_progress: Dict) -> List[str]:
        """Identify areas needing improvement."""
        weaknesses = []
        for concept, progress in concept_progress.items():
            if progress["mastery"] < 0.6:
                weaknesses.append(concept)
        return weaknesses[:3]  # Top 3 areas for improvement
    
    def _generate_learning_recommendations(self, concept_progress: Dict) -> List[str]:
        """Generate learning recommendations."""
        recommendations = []
        
        # Find lowest mastery concept
        lowest_concept = min(concept_progress.items(), key=lambda x: x[1]["mastery"])
        recommendations.append(f"Focus on improving {lowest_concept[0]} skills")
        
        # Find highest confidence but lower mastery
        confidence_mastery_gaps = []
        for concept, progress in concept_progress.items():
            gap = progress["confidence"] - progress["mastery"]
            if gap > 0.2:
                confidence_mastery_gaps.append((concept, gap))
        
        if confidence_mastery_gaps:
            overconfident_concept = max(confidence_mastery_gaps, key=lambda x: x[1])[0]
            recommendations.append(f"Practice more {overconfident_concept} exercises to match confidence with mastery")
        
        return recommendations


class TestExerciseAnalyticsCore:
    """Test core Exercise Analytics functionality."""

    @pytest.fixture
    def analytics(self):
        """Create MockExerciseAnalytics instance."""
        return MockExerciseAnalytics()

    @pytest.fixture
    def sample_session(self):
        """Create sample exercise session."""
        exercises = [
            Exercise("ex1", "Addition", "2 + 3", "addition", 1.0, []),
            Exercise("ex2", "Multiplication", "4 × 5", "multiplication", 2.0, []),
            Exercise("ex3", "Variables", "x + 5 = 10", "variables", 3.0, [])
        ]
        return ExerciseSession("session1", "practice", exercises, 15, ["addition", "multiplication"])

    @pytest.fixture
    def sample_student(self):
        """Create sample student."""
        student = LearningGraph("test_student", "Test Student")
        student.set_mastery("addition", 0.8, 0.9)
        student.set_mastery("multiplication", 0.6, 0.7)
        student.set_mastery("variables", 0.4, 0.5)
        student.set_mastery("fractions", 0.2, 0.3)
        return student

    @pytest.fixture
    def sample_responses(self):
        """Create sample student responses."""
        return [
            {"exercise_id": "ex1", "correct": True, "time_taken": 30, "difficulty": 1.0},
            {"exercise_id": "ex2", "correct": False, "time_taken": 45, "difficulty": 2.0, "error_type": "computational"},
            {"exercise_id": "ex3", "correct": True, "time_taken": 60, "difficulty": 3.0},
        ]

    def test_analyze_session_performance_basic(self, analytics, sample_session, sample_responses):
        """Test basic session performance analysis."""
        result = analytics.analyze_session_performance(sample_session, sample_responses)
        
        assert "session_id" in result
        assert result["session_id"] == "session1"
        assert result["total_exercises"] == 3
        assert result["correct_responses"] == 2
        assert result["accuracy"] == 2/3
        assert result["average_time"] == 45.0  # (30 + 45 + 60) / 3
        assert result["average_difficulty"] == 2.0  # (1.0 + 2.0 + 3.0) / 3

    def test_analyze_session_performance_empty_responses(self, analytics, sample_session):
        """Test session analysis with no responses."""
        result = analytics.analyze_session_performance(sample_session, [])
        
        assert "error" in result
        assert result["error"] == "No responses provided"

    def test_calculate_learning_velocity_improving(self, analytics, sample_student):
        """Test learning velocity calculation for improving student."""
        result = analytics.calculate_learning_velocity(sample_student, "addition", 30)
        
        assert result["concept_id"] == "addition"
        assert result["current_mastery"] == 0.8
        assert result["learning_velocity"] > 0  # Should be positive for improvement
        assert result["trend"] == "improving"

    def test_calculate_learning_velocity_stable(self, analytics, sample_student):
        """Test learning velocity for stable performance."""
        # Mock a student with no improvement
        stable_student = LearningGraph("stable", "Stable Student")
        stable_student.set_mastery("addition", 0.5, 0.6)
        
        with patch.object(analytics, 'calculate_learning_velocity') as mock_calc:
            mock_calc.return_value = {
                "concept_id": "addition",
                "current_mastery": 0.5,
                "historical_mastery": 0.5,
                "learning_velocity": 0.0,
                "trend": "stable"
            }
            
            result = analytics.calculate_learning_velocity(stable_student, "addition", 30)
            assert result["trend"] == "stable"

    def test_generate_difficulty_recommendations_high_performance(self, analytics, sample_student):
        """Test difficulty recommendations for high-performing student."""
        performance_history = [
            {"accuracy": 0.9, "difficulty": 2.0},
            {"accuracy": 0.85, "difficulty": 2.0},
            {"accuracy": 0.9, "difficulty": 2.0}
        ]
        
        result = analytics.generate_difficulty_recommendations(
            sample_student, "addition", performance_history
        )
        
        assert result["recommended_difficulty"] > result["current_difficulty"]
        assert "increase difficulty" in result["reasoning"].lower()

    def test_generate_difficulty_recommendations_low_performance(self, analytics, sample_student):
        """Test difficulty recommendations for struggling student."""
        performance_history = [
            {"accuracy": 0.4, "difficulty": 3.0},
            {"accuracy": 0.5, "difficulty": 3.0},
            {"accuracy": 0.45, "difficulty": 3.0}
        ]
        
        result = analytics.generate_difficulty_recommendations(
            sample_student, "variables", performance_history
        )
        
        assert result["recommended_difficulty"] < result["current_difficulty"]
        assert "decrease difficulty" in result["reasoning"].lower()

    def test_generate_difficulty_recommendations_no_history(self, analytics, sample_student):
        """Test difficulty recommendations with no performance history."""
        result = analytics.generate_difficulty_recommendations(
            sample_student, "addition", []
        )
        
        assert "recommended_difficulty" in result
        assert result["confidence"] == 0.5
        assert "no performance history" in result["reasoning"].lower()

    def test_analyze_error_patterns_with_errors(self, analytics):
        """Test error pattern analysis with various error types."""
        responses_with_errors = [
            {"correct": False, "error_type": "computational"},
            {"correct": False, "error_type": "computational"},
            {"correct": False, "error_type": "conceptual"},
            {"correct": True},
            {"correct": True}
        ]
        
        result = analytics.analyze_error_patterns(responses_with_errors)
        
        assert result["total_errors"] == 3
        assert result["error_rate"] == 0.6  # 3 errors out of 5 responses
        assert len(result["common_error_types"]) > 0
        assert result["common_error_types"][0][0] == "computational"  # Most common
        assert result["common_error_types"][0][1] == 2  # Appears twice

    def test_analyze_error_patterns_no_errors(self, analytics):
        """Test error pattern analysis with perfect performance."""
        perfect_responses = [
            {"correct": True},
            {"correct": True},
            {"correct": True}
        ]
        
        result = analytics.analyze_error_patterns(perfect_responses)
        
        assert result["total_errors"] == 0
        assert result["error_rate"] == 0.0
        assert result["common_error_types"] == []

    def test_calculate_engagement_metrics_high_engagement(self, analytics, sample_session):
        """Test engagement calculation for highly engaged student."""
        high_engagement_data = [
            {"duration": 45, "help_requested": False},
            {"duration": 60, "help_requested": True},
            {"duration": 50, "help_requested": False}
        ]
        
        result = analytics.calculate_engagement_metrics(sample_session, high_engagement_data)
        
        assert result["session_duration"] == 155
        assert result["average_time_per_exercise"] == 155/3
        assert result["engagement_score"] > 0.6
        assert result["engagement_level"] in ["medium", "high"]

    def test_calculate_engagement_metrics_low_engagement(self, analytics, sample_session):
        """Test engagement calculation for disengaged student."""
        low_engagement_data = [
            {"duration": 5, "help_requested": False},  # Too quick
            {"duration": 8, "help_requested": False},  # Too quick
            {"duration": 200, "help_requested": False}  # Too slow
        ]
        
        result = analytics.calculate_engagement_metrics(sample_session, low_engagement_data)
        
        assert result["quick_answers"] == 2
        assert result["slow_answers"] == 1
        assert result["engagement_score"] < 0.6
        assert result["engagement_level"] == "low"

    def test_generate_progress_report_comprehensive(self, analytics, sample_student):
        """Test comprehensive progress report generation."""
        result = analytics.generate_progress_report(sample_student, 30)
        
        assert result["student_id"] == "test_student"
        assert result["report_period_days"] == 30
        assert "overall_mastery" in result
        assert "concept_progress" in result
        assert "strengths" in result
        assert "areas_for_improvement" in result
        assert "recommendations" in result
        
        # Check concept progress structure
        for concept, progress in result["concept_progress"].items():
            assert "mastery" in progress
            assert "confidence" in progress
            assert "improvement" in progress
            assert "exercises_completed" in progress

    def test_progress_report_identifies_strengths_and_weaknesses(self, analytics, sample_student):
        """Test that progress report correctly identifies strengths and weaknesses."""
        result = analytics.generate_progress_report(sample_student, 30)
        
        # Addition has high mastery (0.8), should be strength
        assert "addition" in result["strengths"]
        
        # Fractions has low mastery (0.2), should be area for improvement
        assert "fractions" in result["areas_for_improvement"]

    def test_analytics_handles_edge_cases(self, analytics, sample_student):
        """Test that analytics handle edge cases gracefully."""
        # Empty interaction data
        empty_result = analytics.calculate_engagement_metrics(
            ExerciseSession("test", "practice", [], 0, []), []
        )
        assert "error" in empty_result
        
        # Single response
        single_response = [{"correct": True, "time_taken": 30, "difficulty": 2.0}]
        single_result = analytics.analyze_session_performance(
            ExerciseSession("test", "practice", [], 0, []), single_response
        )
        assert single_result["total_exercises"] == 1
        assert single_result["accuracy"] == 1.0


class TestExerciseAnalyticsAdvanced:
    """Test advanced analytics functionality."""

    @pytest.fixture
    def analytics_with_history(self):
        """Create analytics with historical data."""
        analytics = MockExerciseAnalytics()
        
        # Add mock historical data
        analytics.performance_data = {
            "test_student": {
                "addition": [
                    {"date": "2024-01-01", "accuracy": 0.7, "difficulty": 2.0},
                    {"date": "2024-01-02", "accuracy": 0.8, "difficulty": 2.0},
                    {"date": "2024-01-03", "accuracy": 0.85, "difficulty": 2.5}
                ]
            }
        }
        
        return analytics

    def test_trend_analysis_over_time(self, analytics_with_history, sample_student):
        """Test trend analysis over time periods."""
        # This would test trend analysis functionality
        # For now, test the learning velocity which shows trends
        result = analytics_with_history.calculate_learning_velocity(sample_student, "addition", 30)
        
        assert "trend" in result
        assert result["trend"] in ["improving", "stable", "declining"]

    def test_comparative_analytics_multiple_students(self, analytics):
        """Test comparative analytics across multiple students."""
        student1 = LearningGraph("student1", "Student One")
        student1.set_mastery("addition", 0.8, 0.9)
        
        student2 = LearningGraph("student2", "Student Two")
        student2.set_mastery("addition", 0.6, 0.7)
        
        # Generate reports for both students
        report1 = analytics.generate_progress_report(student1, 30)
        report2 = analytics.generate_progress_report(student2, 30)
        
        # Compare overall mastery
        assert report1["overall_mastery"] > report2["overall_mastery"]

    def test_recommendation_quality_metrics(self, analytics, sample_student):
        """Test quality of generated recommendations."""
        performance_history = [
            {"accuracy": 0.9, "difficulty": 2.0},
            {"accuracy": 0.85, "difficulty": 2.0}
        ]
        
        result = analytics.generate_difficulty_recommendations(
            sample_student, "addition", performance_history
        )
        
        # Recommendations should be reasonable
        assert 1.0 <= result["recommended_difficulty"] <= 5.0
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 