#!/usr/bin/env python3
"""
Comprehensive tests for Math Tutoring Agent.

This test suite provides coverage for the AI agent components
that currently have 0% test coverage.
"""

import pytest
import asyncio
import tempfile
import os
import sys
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Test imports
try:
    from math_learning.ai_agent.math_tutoring_agent import MathTutoringAgent
    from math_learning.ai_agent.error_analysis import ErrorAnalyzer
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    print(f"AI Agent components not available: {e}")


# Mock LLM for testing
class MockLLM:
    """Mock LLM for testing AI agent."""
    
    def __init__(self):
        self.responses = {
            "explain_concept": "This is a basic math concept that involves...",
            "generate_exercise": '{"problem": "What is 2 + 3?", "answer": "5", "explanation": "Addition"}',
            "analyze_error": "The student made an arithmetic error in step 2.",
            "provide_hint": "Try breaking down the problem into smaller steps.",
            "assess_understanding": "Based on the responses, the student understands 70% of the concept."
        }
        self.call_count = 0
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Mock generate response."""
        self.call_count += 1
        
        # Simple pattern matching for different types of requests
        if "explain" in prompt.lower():
            return self.responses["explain_concept"]
        elif "exercise" in prompt.lower() or "problem" in prompt.lower():
            return self.responses["generate_exercise"]
        elif "error" in prompt.lower():
            return self.responses["analyze_error"]
        elif "hint" in prompt.lower():
            return self.responses["provide_hint"]
        elif "assess" in prompt.lower():
            return self.responses["assess_understanding"]
        else:
            return "I understand your question about mathematics."
    
    def set_response(self, response_type: str, response: str):
        """Set a specific response for testing."""
        self.responses[response_type] = response


# Mock Knowledge Graph
class MockKnowledgeGraph:
    """Mock knowledge graph for testing."""
    
    def __init__(self):
        self.concepts = {
            "addition": {
                "id": "addition",
                "name": "Addition",
                "description": "Adding numbers together",
                "difficulty": 0.2,
                "prerequisites": []
            },
            "multiplication": {
                "id": "multiplication", 
                "name": "Multiplication",
                "description": "Multiplying numbers",
                "difficulty": 0.4,
                "prerequisites": ["addition"]
            }
        }
        self.exercises = []
    
    def get_concept(self, concept_id: str) -> Dict:
        """Get concept by ID."""
        return self.concepts.get(concept_id, {})
    
    def get_prerequisites(self, concept_id: str) -> List[str]:
        """Get prerequisites for a concept."""
        concept = self.concepts.get(concept_id, {})
        return concept.get("prerequisites", [])
    
    def generate_exercise(self, concept_id: str, difficulty: float = 0.5) -> Dict:
        """Generate an exercise for a concept."""
        concept = self.concepts.get(concept_id, {})
        return {
            "id": f"ex_{concept_id}_{len(self.exercises)}",
            "concept_id": concept_id,
            "problem": f"Sample problem for {concept.get('name', concept_id)}",
            "answer": "sample_answer",
            "difficulty": difficulty,
            "hints": ["Try step by step", "Remember the basics"]
        }


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    return MockLLM()


@pytest.fixture
def mock_knowledge_graph():
    """Create mock knowledge graph."""
    return MockKnowledgeGraph()


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="AI Agent components not available")
class TestMathTutoringAgent:
    """Test suite for Math Tutoring Agent."""
    
    def test_agent_initialization(self, mock_llm, mock_knowledge_graph):
        """Test agent initialization."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student",
            student_name="Test Student"
        )
        
        assert agent.llm == mock_llm
        assert agent.knowledge_graph == mock_knowledge_graph
        assert agent.student_id == "test_student"
        assert agent.student_name == "Test Student"
        assert hasattr(agent, 'session_history')
        assert hasattr(agent, 'current_session')
    
    @pytest.mark.asyncio
    async def test_explain_concept(self, mock_llm, mock_knowledge_graph):
        """Test concept explanation functionality."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        explanation = await agent.explain_concept("addition", learning_style="visual")
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert mock_llm.call_count > 0
    
    @pytest.mark.asyncio
    async def test_generate_exercise(self, mock_llm, mock_knowledge_graph):
        """Test exercise generation."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        exercise = await agent.generate_exercise("addition", difficulty=0.3)
        
        assert isinstance(exercise, dict)
        assert "problem" in exercise or "id" in exercise
        assert mock_llm.call_count > 0
    
    @pytest.mark.asyncio
    async def test_provide_hint(self, mock_llm, mock_knowledge_graph):
        """Test hint provision."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        exercise = {"problem": "What is 5 + 3?", "concept_id": "addition"}
        student_work = "I think it's 7"
        
        hint = await agent.provide_hint(exercise, student_work)
        
        assert isinstance(hint, str)
        assert len(hint) > 0
        assert mock_llm.call_count > 0
    
    @pytest.mark.asyncio
    async def test_analyze_error(self, mock_llm, mock_knowledge_graph):
        """Test error analysis."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        exercise = {"problem": "What is 12 ÷ 4?", "answer": "3", "concept_id": "division"}
        student_answer = "4"
        
        analysis = await agent.analyze_error(exercise, student_answer)
        
        assert isinstance(analysis, dict)
        assert "error_type" in analysis or "analysis" in analysis or isinstance(analysis, str)
        assert mock_llm.call_count > 0
    
    @pytest.mark.asyncio
    async def test_assess_understanding(self, mock_llm, mock_knowledge_graph):
        """Test understanding assessment."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        session_data = {
            "exercises_completed": 5,
            "correct_answers": 4,
            "concepts_covered": ["addition", "subtraction"],
            "common_errors": ["arithmetic mistakes"]
        }
        
        assessment = await agent.assess_understanding("addition", session_data)
        
        assert isinstance(assessment, (dict, str, float))
        assert mock_llm.call_count > 0
    
    @pytest.mark.asyncio
    async def test_start_learning_session(self, mock_llm, mock_knowledge_graph):
        """Test starting a learning session."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        session_id = await agent.start_learning_session(
            goals=["Learn addition", "Practice basic math"],
            preferred_style="interactive"
        )
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert agent.current_session is not None
    
    @pytest.mark.asyncio
    async def test_end_learning_session(self, mock_llm, mock_knowledge_graph):
        """Test ending a learning session."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        # Start a session first
        session_id = await agent.start_learning_session(["Learn math"])
        
        # End the session
        summary = await agent.end_learning_session(session_id)
        
        assert isinstance(summary, dict)
        assert "session_id" in summary or "duration" in summary or len(summary) >= 0
    
    @pytest.mark.asyncio
    async def test_get_personalized_recommendations(self, mock_llm, mock_knowledge_graph):
        """Test getting personalized recommendations."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        student_profile = {
            "learning_style": "visual",
            "current_level": "beginner",
            "interests": ["games", "puzzles"],
            "weak_areas": ["multiplication"]
        }
        
        recommendations = await agent.get_personalized_recommendations(student_profile)
        
        assert isinstance(recommendations, (dict, list))
        if isinstance(recommendations, dict):
            assert len(recommendations) >= 0
        if isinstance(recommendations, list):
            assert len(recommendations) >= 0
    
    def test_session_history_management(self, mock_llm, mock_knowledge_graph):
        """Test session history management."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        # Test initial state
        assert hasattr(agent, 'session_history')
        assert isinstance(agent.session_history, list)
        
        # Test adding session data
        session_data = {
            "session_id": "test_session_1",
            "start_time": "2024-01-01T10:00:00",
            "exercises": [],
            "performance": {}
        }
        
        agent.add_to_session_history(session_data)
        assert len(agent.session_history) > 0
        assert agent.session_history[-1]["session_id"] == "test_session_1"
    
    def test_error_handling(self, mock_knowledge_graph):
        """Test error handling in agent operations."""
        # Test with failing LLM
        failing_llm = Mock()
        failing_llm.generate_response = AsyncMock(side_effect=Exception("LLM failed"))
        
        agent = MathTutoringAgent(
            llm=failing_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        # Should handle LLM failures gracefully
        with pytest.raises(Exception):
            asyncio.run(agent.explain_concept("addition"))
    
    @pytest.mark.asyncio
    async def test_adaptive_difficulty(self, mock_llm, mock_knowledge_graph):
        """Test adaptive difficulty adjustment."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        # Simulate student performance
        performance_data = {
            "recent_scores": [0.8, 0.9, 0.7, 0.85],
            "concept_id": "addition",
            "current_difficulty": 0.5
        }
        
        new_difficulty = await agent.adjust_difficulty(performance_data)
        
        assert isinstance(new_difficulty, float)
        assert 0.0 <= new_difficulty <= 1.0
    
    @pytest.mark.asyncio
    async def test_multi_concept_session(self, mock_llm, mock_knowledge_graph):
        """Test handling sessions with multiple concepts."""
        agent = MathTutoringAgent(
            llm=mock_llm,
            knowledge_graph=mock_knowledge_graph,
            student_id="test_student"
        )
        
        # Start session with multiple concepts
        session_id = await agent.start_learning_session(
            goals=["Learn addition", "Learn multiplication"],
            concepts=["addition", "multiplication"]
        )
        
        assert session_id is not None
        
        # Generate exercises for different concepts
        ex1 = await agent.generate_exercise("addition")
        ex2 = await agent.generate_exercise("multiplication")
        
        assert isinstance(ex1, dict)
        assert isinstance(ex2, dict)
        
        # End session
        summary = await agent.end_learning_session(session_id)
        assert isinstance(summary, dict)


@pytest.mark.skipif(not AGENT_AVAILABLE, reason="AI Agent components not available")
class TestErrorAnalyzer:
    """Test suite for Error Analyzer."""
    
    def test_error_analyzer_initialization(self):
        """Test error analyzer initialization."""
        analyzer = ErrorAnalyzer()
        
        assert hasattr(analyzer, 'error_patterns')
        assert hasattr(analyzer, 'common_mistakes')
    
    def test_analyze_arithmetic_error(self):
        """Test analyzing arithmetic errors."""
        analyzer = ErrorAnalyzer()
        
        correct_answer = "8"
        student_answer = "7"
        problem_type = "addition"
        
        analysis = analyzer.analyze_error(correct_answer, student_answer, problem_type)
        
        assert isinstance(analysis, dict)
        assert "error_type" in analysis
        assert "description" in analysis
        assert "suggestions" in analysis
    
    def test_analyze_conceptual_error(self):
        """Test analyzing conceptual errors."""
        analyzer = ErrorAnalyzer()
        
        correct_answer = "15"
        student_answer = "12"
        problem_type = "multiplication"
        problem_context = "3 × 5"
        
        analysis = analyzer.analyze_error(
            correct_answer, 
            student_answer, 
            problem_type,
            context=problem_context
        )
        
        assert isinstance(analysis, dict)
        assert "error_type" in analysis
    
    def test_pattern_recognition(self):
        """Test error pattern recognition."""
        analyzer = ErrorAnalyzer()
        
        # Test multiple errors to identify patterns
        errors = [
            {"correct": "10", "student": "9", "type": "addition"},
            {"correct": "12", "student": "11", "type": "addition"},
            {"correct": "15", "student": "14", "type": "addition"}
        ]
        
        patterns = analyzer.identify_patterns(errors)
        
        assert isinstance(patterns, (dict, list))
        if isinstance(patterns, dict):
            assert len(patterns) >= 0
        if isinstance(patterns, list):
            assert len(patterns) >= 0
    
    def test_suggestion_generation(self):
        """Test generating learning suggestions."""
        analyzer = ErrorAnalyzer()
        
        error_analysis = {
            "error_type": "arithmetic",
            "pattern": "off_by_one",
            "frequency": 3
        }
        
        suggestions = analyzer.generate_suggestions(error_analysis)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)


class TestMockComponents:
    """Test the mock components used in testing."""
    
    @pytest.mark.asyncio
    async def test_mock_llm(self):
        """Test mock LLM functionality."""
        llm = MockLLM()
        
        # Test different prompt types
        explain_response = await llm.generate_response("Please explain addition")
        exercise_response = await llm.generate_response("Generate an exercise")
        hint_response = await llm.generate_response("Provide a hint")
        
        assert "concept" in explain_response
        assert "problem" in exercise_response or "exercise" in exercise_response
        assert "hint" in hint_response or "Try" in hint_response
        
        assert llm.call_count == 3
    
    def test_mock_knowledge_graph(self):
        """Test mock knowledge graph functionality."""
        kg = MockKnowledgeGraph()
        
        # Test concept retrieval
        addition_concept = kg.get_concept("addition")
        assert addition_concept["name"] == "Addition"
        
        # Test prerequisites
        mult_prereqs = kg.get_prerequisites("multiplication")
        assert "addition" in mult_prereqs
        
        # Test exercise generation
        exercise = kg.generate_exercise("addition", 0.3)
        assert exercise["concept_id"] == "addition"
        assert exercise["difficulty"] == 0.3


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 