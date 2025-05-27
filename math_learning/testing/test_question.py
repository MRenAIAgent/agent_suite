"""
Test Question class for algebra assessment with error analysis.
Maps student errors to specific concept misunderstandings.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    ALGEBRAIC_EXPRESSION = "algebraic_expression"

class DifficultyLevel(Enum):
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4

@dataclass
class ErrorMapping:
    """Maps a specific wrong answer to a concept misunderstanding."""
    wrong_answer: str
    concept_id: str
    concept_name: str
    error_description: str
    remediation_hint: str

@dataclass
class TestQuestion:
    """Represents an algebra test question with error analysis capabilities."""
    
    id: str
    question_text: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    correct_answer: str
    explanation: str
    
    # For multiple choice questions
    options: Optional[List[str]] = None
    
    # Concept mappings
    primary_concepts: List[str] = None  # Main concepts being tested
    prerequisite_concepts: List[str] = None  # Prerequisites needed
    
    # Error analysis
    common_errors: List[ErrorMapping] = None
    
    # Metadata
    grade_level: str = ""
    topic_area: str = ""
    time_limit_minutes: int = 5
    
    def __post_init__(self):
        if self.primary_concepts is None:
            self.primary_concepts = []
        if self.prerequisite_concepts is None:
            self.prerequisite_concepts = []
        if self.common_errors is None:
            self.common_errors = []
    
    def check_answer(self, student_answer: str) -> Tuple[bool, Optional[ErrorMapping]]:
        """
        Check if student answer is correct and identify specific error if wrong.
        
        Returns:
            (is_correct, error_mapping_if_wrong)
        """
        # Normalize answers for comparison
        student_answer = student_answer.strip()
        correct = student_answer.lower() == self.correct_answer.lower()
        
        if correct:
            return True, None
        
        # Find matching error mapping
        for error in self.common_errors:
            if student_answer.lower() == error.wrong_answer.lower():
                return False, error
        
        # Generic error if no specific mapping found
        generic_error = ErrorMapping(
            wrong_answer=student_answer,
            concept_id="UNKNOWN",
            concept_name="Unknown Concept Gap",
            error_description=f"Unexpected answer: {student_answer}",
            remediation_hint="Review the primary concepts for this question"
        )
        
        return False, generic_error
    
    def get_concept_dependencies(self) -> List[str]:
        """Get all concepts (primary + prerequisites) needed for this question."""
        return self.primary_concepts + self.prerequisite_concepts 