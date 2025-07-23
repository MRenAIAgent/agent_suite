"""
Error Analysis Module for Mathematical Problem Solving

This module analyzes student errors in mathematical problems and maps them
to specific concepts that need reinforcement.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ErrorType(Enum):
    """Types of mathematical errors."""
    ARITHMETIC = "arithmetic"
    ALGEBRAIC = "algebraic"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    NOTATION = "notation"
    READING_COMPREHENSION = "reading_comprehension"
    CARELESS = "careless"

@dataclass
class ErrorAnalysis:
    """Result of error analysis."""
    error_type: ErrorType
    missing_concepts: List[str]
    confidence: float
    description: str
    suggested_remediation: str
    difficulty_level: str

class MathErrorAnalyzer:
    """Analyzes mathematical errors and identifies missing concepts."""
    
    def __init__(self, knowledge_graph=None):
        self.knowledge_graph = knowledge_graph
        self.error_patterns = self._initialize_error_patterns()
        self.concept_mappings = self._initialize_concept_mappings()
    
    def _initialize_error_patterns(self) -> Dict[str, Dict]:
        """Initialize common error patterns and their characteristics."""
        return {
            # Arithmetic errors
            "addition_error": {
                "pattern": r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)",
                "error_type": ErrorType.ARITHMETIC,
                "concepts": ["addition", "basic_arithmetic"]
            },
            "subtraction_error": {
                "pattern": r"(\d+)\s*-\s*(\d+)\s*=\s*(\d+)",
                "error_type": ErrorType.ARITHMETIC,
                "concepts": ["subtraction", "basic_arithmetic"]
            },
            "multiplication_error": {
                "pattern": r"(\d+)\s*[×*]\s*(\d+)\s*=\s*(\d+)",
                "error_type": ErrorType.ARITHMETIC,
                "concepts": ["multiplication", "basic_arithmetic"]
            },
            "division_error": {
                "pattern": r"(\d+)\s*[÷/]\s*(\d+)\s*=\s*(\d+)",
                "error_type": ErrorType.ARITHMETIC,
                "concepts": ["division", "basic_arithmetic"]
            },
            
            # Fraction errors
            "fraction_addition": {
                "pattern": r"(\d+)/(\d+)\s*\+\s*(\d+)/(\d+)\s*=\s*(\d+)/(\d+)",
                "error_type": ErrorType.CONCEPTUAL,
                "concepts": ["fractions", "fraction_addition", "common_denominators"]
            },
            "fraction_multiplication": {
                "pattern": r"(\d+)/(\d+)\s*[×*]\s*(\d+)/(\d+)\s*=\s*(\d+)/(\d+)",
                "error_type": ErrorType.CONCEPTUAL,
                "concepts": ["fractions", "fraction_multiplication"]
            },
            
            # Algebraic errors
            "linear_equation": {
                "pattern": r"(\w+)\s*=\s*(.+)",
                "error_type": ErrorType.ALGEBRAIC,
                "concepts": ["algebra", "linear_equations", "solving_equations"]
            },
            "quadratic_equation": {
                "pattern": r"(\w+)\^2.*=.*",
                "error_type": ErrorType.ALGEBRAIC,
                "concepts": ["algebra", "quadratic_equations", "factoring"]
            },
            
            # Decimal errors
            "decimal_operations": {
                "pattern": r"(\d+\.\d+)\s*[+\-×÷*/]\s*(\d+\.\d+)\s*=\s*(\d+\.\d*)",
                "error_type": ErrorType.CONCEPTUAL,
                "concepts": ["decimals", "decimal_operations"]
            },
            
            # Percentage errors
            "percentage_calculation": {
                "pattern": r"(\d+)%.*=.*(\d+)",
                "error_type": ErrorType.CONCEPTUAL,
                "concepts": ["percentages", "percentage_calculations"]
            }
        }
    
    def _initialize_concept_mappings(self) -> Dict[str, List[str]]:
        """Map error types to prerequisite concepts."""
        return {
            "addition": ["counting", "number_recognition", "place_value"],
            "subtraction": ["addition", "counting", "number_recognition"],
            "multiplication": ["addition", "counting", "multiplication_tables"],
            "division": ["multiplication", "subtraction", "division_facts"],
            "fractions": ["division", "parts_of_whole", "equivalent_fractions"],
            "decimals": ["fractions", "place_value", "division"],
            "percentages": ["fractions", "decimals", "proportions"],
            "algebra": ["arithmetic_operations", "variables", "equations"],
            "geometry": ["shapes", "measurement", "spatial_reasoning"]
        }
    
    def analyze_error(self, problem_text: str, student_answer: str, correct_answer: str) -> ErrorAnalysis:
        """
        Analyze a mathematical error and identify missing concepts.
        
        Args:
            problem_text: The original math problem
            student_answer: Student's incorrect answer
            correct_answer: The correct answer
            
        Returns:
            ErrorAnalysis object with detailed analysis
        """
        # Clean and normalize inputs
        problem_text = self._clean_text(problem_text)
        student_answer = self._clean_text(student_answer)
        correct_answer = self._clean_text(correct_answer)
        
        # Identify the type of problem
        problem_type = self._identify_problem_type(problem_text)
        
        # Analyze the specific error
        error_analysis = self._analyze_specific_error(
            problem_text, student_answer, correct_answer, problem_type
        )
        
        return error_analysis
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize mathematical text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize mathematical symbols
        text = text.replace('×', '*').replace('÷', '/')
        text = text.replace('−', '-')  # Replace minus sign with hyphen
        
        return text
    
    def _identify_problem_type(self, problem_text: str) -> str:
        """Identify the type of mathematical problem."""
        problem_text_lower = problem_text.lower()
        
        # Check for different problem types
        if any(word in problem_text_lower for word in ['fraction', '/', 'numerator', 'denominator']):
            return "fractions"
        elif any(word in problem_text_lower for word in ['decimal', '.']):
            return "decimals"
        elif any(word in problem_text_lower for word in ['percent', '%']):
            return "percentages"
        elif any(word in problem_text_lower for word in ['x', 'y', 'solve', 'equation']):
            return "algebra"
        elif any(word in problem_text_lower for word in ['area', 'perimeter', 'volume', 'angle']):
            return "geometry"
        elif any(op in problem_text for op in ['+', '-', '*', '/', '×', '÷']):
            return "arithmetic"
        else:
            return "general"
    
    def _analyze_specific_error(self, problem_text: str, student_answer: str, 
                              correct_answer: str, problem_type: str) -> ErrorAnalysis:
        """Analyze the specific error based on problem type."""
        
        # Try to parse numerical values
        student_num = self._extract_number(student_answer)
        correct_num = self._extract_number(correct_answer)
        
        if student_num is not None and correct_num is not None:
            # Numerical comparison analysis
            return self._analyze_numerical_error(
                problem_text, student_num, correct_num, problem_type
            )
        else:
            # Text-based analysis
            return self._analyze_text_error(
                problem_text, student_answer, correct_answer, problem_type
            )
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text."""
        try:
            # Handle fractions
            if '/' in text:
                parts = text.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            
            # Handle decimals and integers
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if numbers:
                return float(numbers[0])
        except (ValueError, ZeroDivisionError):
            pass
        
        return None
    
    def _analyze_numerical_error(self, problem_text: str, student_answer: float, 
                                correct_answer: float, problem_type: str) -> ErrorAnalysis:
        """Analyze numerical errors."""
        
        error_magnitude = abs(student_answer - correct_answer)
        relative_error = error_magnitude / abs(correct_answer) if correct_answer != 0 else float('inf')
        
        # Determine error type based on problem type and error characteristics
        if problem_type == "fractions":
            return self._analyze_fraction_error(problem_text, student_answer, correct_answer)
        elif problem_type == "decimals":
            return self._analyze_decimal_error(problem_text, student_answer, correct_answer)
        elif problem_type == "percentages":
            return self._analyze_percentage_error(problem_text, student_answer, correct_answer)
        elif problem_type == "arithmetic":
            return self._analyze_arithmetic_error(problem_text, student_answer, correct_answer)
        else:
            # General numerical error
            if relative_error < 0.1:  # Small error, likely careless
                return ErrorAnalysis(
                    error_type=ErrorType.CARELESS,
                    missing_concepts=[problem_type],
                    confidence=0.7,
                    description="Small numerical error, likely a careless mistake",
                    suggested_remediation="Practice careful calculation and double-checking",
                    difficulty_level="review"
                )
            else:  # Larger error, likely conceptual
                return ErrorAnalysis(
                    error_type=ErrorType.CONCEPTUAL,
                    missing_concepts=self.concept_mappings.get(problem_type, [problem_type]),
                    confidence=0.8,
                    description=f"Significant error in {problem_type} problem",
                    suggested_remediation=f"Review fundamental concepts in {problem_type}",
                    difficulty_level="foundational"
                )
    
    def _analyze_fraction_error(self, problem_text: str, student_answer: float, 
                               correct_answer: float) -> ErrorAnalysis:
        """Analyze fraction-specific errors."""
        # Check if student added numerators and denominators separately
        # This is a common fraction addition error
        
        return ErrorAnalysis(
            error_type=ErrorType.CONCEPTUAL,
            missing_concepts=["fractions", "common_denominators", "fraction_operations"],
            confidence=0.85,
            description="Error in fraction operations, possibly adding numerators and denominators separately",
            suggested_remediation="Practice finding common denominators and proper fraction addition",
            difficulty_level="intermediate"
        )
    
    def _analyze_decimal_error(self, problem_text: str, student_answer: float, 
                              correct_answer: float) -> ErrorAnalysis:
        """Analyze decimal-specific errors."""
        return ErrorAnalysis(
            error_type=ErrorType.CONCEPTUAL,
            missing_concepts=["decimals", "place_value", "decimal_operations"],
            confidence=0.8,
            description="Error in decimal operations or place value understanding",
            suggested_remediation="Review decimal place value and decimal arithmetic",
            difficulty_level="intermediate"
        )
    
    def _analyze_percentage_error(self, problem_text: str, student_answer: float, 
                                 correct_answer: float) -> ErrorAnalysis:
        """Analyze percentage-specific errors."""
        return ErrorAnalysis(
            error_type=ErrorType.CONCEPTUAL,
            missing_concepts=["percentages", "proportions", "decimal_conversion"],
            confidence=0.8,
            description="Error in percentage calculation or conversion",
            suggested_remediation="Practice converting between percentages, decimals, and fractions",
            difficulty_level="intermediate"
        )
    
    def _analyze_arithmetic_error(self, problem_text: str, student_answer: float, 
                                 correct_answer: float) -> ErrorAnalysis:
        """Analyze basic arithmetic errors."""
        error_magnitude = abs(student_answer - correct_answer)
        
        if error_magnitude <= 2:  # Small arithmetic error
            return ErrorAnalysis(
                error_type=ErrorType.ARITHMETIC,
                missing_concepts=["basic_arithmetic", "calculation_accuracy"],
                confidence=0.9,
                description="Small arithmetic calculation error",
                suggested_remediation="Practice basic arithmetic facts and careful calculation",
                difficulty_level="foundational"
            )
        else:  # Larger arithmetic error
            return ErrorAnalysis(
                error_type=ErrorType.CONCEPTUAL,
                missing_concepts=["basic_arithmetic", "number_sense"],
                confidence=0.8,
                description="Significant arithmetic error indicating conceptual gaps",
                suggested_remediation="Review fundamental arithmetic operations and number sense",
                difficulty_level="foundational"
            )
    
    def _analyze_text_error(self, problem_text: str, student_answer: str, 
                           correct_answer: str, problem_type: str) -> ErrorAnalysis:
        """Analyze non-numerical errors."""
        return ErrorAnalysis(
            error_type=ErrorType.PROCEDURAL,
            missing_concepts=[problem_type, "problem_solving"],
            confidence=0.6,
            description=f"Error in {problem_type} problem solving approach",
            suggested_remediation=f"Review {problem_type} problem-solving strategies",
            difficulty_level="intermediate"
        )
    
    def get_prerequisite_concepts(self, missing_concepts: List[str]) -> List[str]:
        """Get prerequisite concepts for the missing concepts."""
        prerequisites = set()
        
        for concept in missing_concepts:
            if concept in self.concept_mappings:
                prerequisites.update(self.concept_mappings[concept])
        
        return list(prerequisites) 