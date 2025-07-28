"""
Shared test fixtures and utilities for math recognition tests.
"""

import pytest
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass

# Sample test data
SAMPLE_QUESTIONS = [
    {
        "id": "Q1",
        "question": "Solve for x: 2x + 3 = 11",
        "correct_answer": "x = 4",
        "student_answers": {
            "correct": "x = 4",
            "arithmetic_error": "x = 5",
            "conceptual_error": "x = 8/2",
            "procedural_error": "x = 11 - 3/2",
            "notation_error": "4"
        },
        "work_shown": {
            "correct": "2x + 3 = 11\n2x = 11 - 3\n2x = 8\nx = 4",
            "arithmetic_error": "2x + 3 = 11\n2x = 11 - 3\n2x = 8\nx = 5",
            "conceptual_error": "2x + 3 = 11\n2x = 11 - 3\nx = 8/2",
            "procedural_error": "2x + 3 = 11\nx = 11 - 3/2"
        }
    },
    {
        "id": "Q2", 
        "question": "Add the fractions: 1/3 + 1/4",
        "correct_answer": "7/12",
        "student_answers": {
            "correct": "7/12",
            "arithmetic_error": "6/12",
            "conceptual_error": "2/7",
            "procedural_error": "1/3 + 1/4 = 2/7"
        },
        "work_shown": {
            "correct": "1/3 + 1/4 = 4/12 + 3/12 = 7/12",
            "arithmetic_error": "1/3 + 1/4 = 4/12 + 2/12 = 6/12",
            "conceptual_error": "1/3 + 1/4 = (1+1)/(3+4) = 2/7"
        }
    },
    {
        "id": "Q3",
        "question": "Factor: x² - 9",
        "correct_answer": "(x-3)*(x+3)",
        "student_answers": {
            "correct": "(x-3)*(x+3)",
            "arithmetic_error": "(x-3)*(x-3)",
            "conceptual_error": "x**2 - 9",
            "procedural_error": "(x**2-9)"
        }
    },
    {
        "id": "Q4",
        "question": "What is 15% of 80?",
        "correct_answer": "12",
        "student_answers": {
            "correct": "12",
            "arithmetic_error": "15",
            "conceptual_error": "53.33",
            "procedural_error": "1200"
        }
    }
]

SAMPLE_STUDENT_HISTORY = {
    "alice": {
        "student_id": "ALICE_123",
        "grade_level": 8,
        "previous_scores": [85, 78, 92, 73],
        "weak_concepts": ["fractions", "negative_numbers"],
        "strong_concepts": ["basic_algebra", "geometry"],
        "learning_style": "visual",
        "confidence_level": "medium"
    },
    "bob": {
        "student_id": "BOB_456", 
        "grade_level": 9,
        "previous_scores": [92, 88, 95, 90],
        "weak_concepts": ["quadratic_equations"],
        "strong_concepts": ["linear_equations", "fractions", "percentages"],
        "learning_style": "analytical",
        "confidence_level": "high"
    },
    "charlie": {
        "student_id": "CHARLIE_789",
        "grade_level": 7,
        "previous_scores": [45, 52, 38, 60],
        "weak_concepts": ["algebra", "fractions", "decimals"],
        "strong_concepts": ["basic_arithmetic"],
        "learning_style": "kinesthetic", 
        "confidence_level": "low"
    }
}

@dataclass
class MockLLMResponse:
    """Mock LLM response for testing AI components."""
    content: str
    confidence: float = 0.8

class MockLLMClient:
    """Mock LLM client for testing AI integration."""
    
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""
    
    async def generate(self, prompt: str) -> MockLLMResponse:
        """Generate a mock response based on prompt keywords."""
        self.call_count += 1
        self.last_prompt = prompt
        
        # Return predefined response if available
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return MockLLMResponse(response)
        
        # Default mock response
        return MockLLMResponse(self._generate_default_response(prompt))
    
    def _generate_default_response(self, prompt: str) -> str:
        """Generate a realistic mock AI response."""
        if "error type" in prompt.lower():
            return """
1. ERROR TYPE CLASSIFICATION:
   - Primary error type: Arithmetic
   - Specific error subtype: Calculation mistake
   - Confidence: 0.85

2. ROOT CAUSE ANALYSIS:
   - Most likely root cause: Careless calculation error
   - Contributing factors: Rushed work, lack of verification
   - Missing prerequisite: None identified
   - Misconceptions: None identified

3. STUDENT UNDERSTANDING ASSESSMENT:
   - Understands: Problem-solving approach, algebraic concepts
   - Missing: Careful calculation habits
   - Problem-solving approach: Correct method, execution error
   - Metacognitive awareness: Medium - recognizes method but not error

4. EDUCATIONAL INSIGHTS:
   - Appropriate grade level for student
   - Problem difficulty is appropriate
   - Student would benefit from verification strategies
   - Confidence appears adequate

5. REMEDIATION RECOMMENDATIONS:
   - Practice careful calculation
   - Implement answer checking strategies
   - Review arithmetic operations
   - Build verification habits

6. DIAGNOSTIC QUESTIONS:
   - "Can you check your arithmetic in step 3?"
   - "What is 11 - 3?"
   - "How do you usually verify your answers?"
   """
        
        return "Mock AI analysis response"

@pytest.fixture
def sample_questions():
    """Provide sample questions for testing."""
    return SAMPLE_QUESTIONS

@pytest.fixture 
def sample_student_history():
    """Provide sample student history data."""
    return SAMPLE_STUDENT_HISTORY

@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client for testing."""
    return MockLLMClient()

@pytest.fixture
def mock_llm_with_responses():
    """Provide a mock LLM client with predefined responses."""
    responses = {
        "arithmetic": "This is an arithmetic error - student made a calculation mistake.",
        "conceptual": "This is a conceptual error - student misunderstands the underlying concept.",
        "procedural": "This is a procedural error - student used the wrong method or steps.",
        "fraction": "Student struggles with fraction operations and common denominators."
    }
    return MockLLMClient(responses)

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Test utilities
def assert_validation_result(result, expected_correct: bool, expected_confidence_min: float = 0.0):
    """Assert validation result properties."""
    assert result.is_correct == expected_correct
    assert result.confidence >= expected_confidence_min
    assert 0.0 <= result.confidence <= 1.0

def assert_error_analysis_result(result, expected_error_type: str = None):
    """Assert error analysis result properties."""
    assert result.error_type is not None
    assert result.confidence > 0.0
    assert result.description is not None
    assert len(result.description) > 0
    if expected_error_type:
        assert expected_error_type.lower() in result.error_type.value.lower()

def assert_root_cause_result(result):
    """Assert root cause analysis result properties."""
    assert result.primary_cause is not None
    assert len(result.primary_cause) > 0
    assert result.confidence > 0.0
    assert isinstance(result.learning_pathway, list)

def assert_gap_analysis_result(result):
    """Assert gap analysis result properties."""
    assert isinstance(result.identified_gaps, list)
    assert result.confidence_score >= 0.0
    assert isinstance(result.priority_gaps, list)

def assert_hybrid_analysis_result(result):
    """Assert hybrid analysis result properties."""
    assert result.correctness_result is not None
    assert result.error_analysis is not None
    assert result.root_cause_analysis is not None
    assert result.gap_analysis is not None
    assert result.ai_insights is not None
    assert result.primary_diagnosis is not None
    assert len(result.recommended_actions) > 0
    assert result.learning_priority in ["immediate", "short_term", "long_term"] 