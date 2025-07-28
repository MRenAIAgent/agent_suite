"""
Answer Correctness Engine

This module provides comprehensive validation of student answers using multiple
validation methods depending on the answer type.
"""

import re
import sympy as sp
from sympy import sympify, simplify, Eq, solve
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from fractions import Fraction
import math

class AnswerType(Enum):
    """Types of mathematical answers."""
    NUMERICAL = "numerical"
    ALGEBRAIC = "algebraic"
    FRACTION = "fraction"
    EQUATION = "equation"
    EXPRESSION = "expression"
    WORD_ANSWER = "word_answer"
    MULTIPLE_CHOICE = "multiple_choice"
    SET = "set"
    INTERVAL = "interval"

class CorrectnessLevel(Enum):
    """Levels of answer correctness."""
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    INVALID_FORMAT = "invalid_format"
    UNABLE_TO_EVALUATE = "unable_to_evaluate"

@dataclass
class ValidationResult:
    """Result of answer validation."""
    is_correct: bool
    correctness_level: CorrectnessLevel
    confidence: float
    answer_type: AnswerType
    student_answer_parsed: str
    expected_answer_parsed: str
    partial_credit: float = 0.0
    error_details: Optional[str] = None
    equivalent_forms: List[str] = None
    validation_method: str = ""

class BaseValidator:
    """Base class for answer validators."""
    
    def __init__(self):
        self.tolerance = 1e-10
    
    def can_handle(self, answer_type: AnswerType) -> bool:
        """Check if this validator can handle the given answer type."""
        raise NotImplementedError
    
    def validate(self, student_answer: str, expected_answer: str, 
                question: str = "") -> ValidationResult:
        """Validate the student answer against expected answer."""
        raise NotImplementedError
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and normalize an answer string."""
        if not answer:
            return ""
        
        # Remove extra whitespace
        answer = answer.strip()
        
        # Replace common mathematical symbols
        replacements = {
            '×': '*',
            '÷': '/',
            '−': '-',
            '√': 'sqrt',
            '²': '**2',
            '³': '**3',
            'π': 'pi',
            '∞': 'oo'
        }
        
        for old, new in replacements.items():
            answer = answer.replace(old, new)
        
        # Remove units (basic handling)
        units = ['cm', 'mm', 'm', 'km', 'in', 'ft', 'yards', 'miles', 
                'seconds', 'minutes', 'hours', 'degrees', '°', '$', '%']
        for unit in units:
            answer = re.sub(rf'\s*{unit}\b', '', answer, flags=re.IGNORECASE)
        
        return answer.strip()

class NumericalValidator(BaseValidator):
    """Validator for numerical answers."""
    
    def can_handle(self, answer_type: AnswerType) -> bool:
        return answer_type == AnswerType.NUMERICAL
    
    def validate(self, student_answer: str, expected_answer: str, 
                question: str = "") -> ValidationResult:
        """Validate numerical answers."""
        try:
            student_cleaned = self._clean_answer(student_answer)
            expected_cleaned = self._clean_answer(expected_answer)
            
            # Try to parse as numbers
            student_num = float(student_cleaned.replace(' ', ''))
            expected_num = float(expected_cleaned.replace(' ', ''))
            
            # Check if they're equal within tolerance
            if abs(student_num - expected_num) <= self.tolerance:
                return ValidationResult(
                    is_correct=True,
                    correctness_level=CorrectnessLevel.CORRECT,
                    confidence=1.0,
                    answer_type=AnswerType.NUMERICAL,
                    student_answer_parsed=str(student_num),
                    expected_answer_parsed=str(expected_num),
                    partial_credit=1.0,
                    validation_method="exact_numerical"
                )
            else:
                # Check if it's close (for partial credit)
                relative_error = abs(student_num - expected_num) / abs(expected_num) if expected_num != 0 else float('inf')
                
                if relative_error < 0.01:  # Within 1%
                    return ValidationResult(
                        is_correct=False,
                        correctness_level=CorrectnessLevel.PARTIALLY_CORRECT,
                        confidence=0.8,
                        answer_type=AnswerType.NUMERICAL,
                        student_answer_parsed=str(student_num),
                        expected_answer_parsed=str(expected_num),
                        partial_credit=0.8,
                        error_details=f"Close but not exact. Relative error: {relative_error:.3%}",
                        validation_method="approximate_numerical"
                    )
                else:
                    return ValidationResult(
                        is_correct=False,
                        correctness_level=CorrectnessLevel.INCORRECT,
                        confidence=1.0,
                        answer_type=AnswerType.NUMERICAL,
                        student_answer_parsed=str(student_num),
                        expected_answer_parsed=str(expected_num),
                        partial_credit=0.0,
                        error_details=f"Numerical values don't match. Difference: {abs(student_num - expected_num):.6f}",
                        validation_method="exact_numerical"
                    )
                    
        except ValueError as e:
            return ValidationResult(
                is_correct=False,
                correctness_level=CorrectnessLevel.INVALID_FORMAT,
                confidence=1.0,
                answer_type=AnswerType.NUMERICAL,
                student_answer_parsed=student_answer,
                expected_answer_parsed=expected_answer,
                partial_credit=0.0,
                error_details=f"Could not parse as numbers: {str(e)}",
                validation_method="numerical_parsing"
            )

class AlgebraicValidator(BaseValidator):
    """Validator for algebraic expressions and equations."""
    
    def can_handle(self, answer_type: AnswerType) -> bool:
        return answer_type in [AnswerType.ALGEBRAIC, AnswerType.EXPRESSION]
    
    def validate(self, student_answer: str, expected_answer: str, 
                question: str = "") -> ValidationResult:
        """Validate algebraic expressions."""
        try:
            student_cleaned = self._clean_answer(student_answer)
            expected_cleaned = self._clean_answer(expected_answer)
            
            # Parse using SymPy
            student_expr = sympify(student_cleaned)
            expected_expr = sympify(expected_cleaned)
            
            # Check if expressions are equivalent
            difference = simplify(student_expr - expected_expr)
            
            if difference == 0:
                return ValidationResult(
                    is_correct=True,
                    correctness_level=CorrectnessLevel.CORRECT,
                    confidence=1.0,
                    answer_type=AnswerType.ALGEBRAIC,
                    student_answer_parsed=str(student_expr),
                    expected_answer_parsed=str(expected_expr),
                    partial_credit=1.0,
                    equivalent_forms=[str(simplify(student_expr)), str(student_expr.expand())],
                    validation_method="symbolic_equivalence"
                )
            else:
                # Check if they're equivalent for specific values (numerical verification)
                variables = list(student_expr.free_symbols.union(expected_expr.free_symbols))
                
                if variables:
                    # Test multiple random values
                    matches = 0
                    tests = 10
                    
                    for _ in range(tests):
                        test_values = {var: np.random.randint(-10, 11) for var in variables}
                        try:
                            student_val = float(student_expr.subs(test_values))
                            expected_val = float(expected_expr.subs(test_values))
                            
                            if abs(student_val - expected_val) <= self.tolerance:
                                matches += 1
                        except:
                            continue
                    
                    if matches == tests:
                        return ValidationResult(
                            is_correct=True,
                            correctness_level=CorrectnessLevel.CORRECT,
                            confidence=0.95,
                            answer_type=AnswerType.ALGEBRAIC,
                            student_answer_parsed=str(student_expr),
                            expected_answer_parsed=str(expected_expr),
                            partial_credit=1.0,
                            error_details="Expressions are equivalent (verified numerically)",
                            validation_method="numerical_verification"
                        )
                
                return ValidationResult(
                    is_correct=False,
                    correctness_level=CorrectnessLevel.INCORRECT,
                    confidence=0.9,
                    answer_type=AnswerType.ALGEBRAIC,
                    student_answer_parsed=str(student_expr),
                    expected_answer_parsed=str(expected_expr),
                    partial_credit=0.0,
                    error_details=f"Algebraic expressions are not equivalent. Difference: {difference}",
                    validation_method="symbolic_equivalence"
                )
                
        except Exception as e:
            return ValidationResult(
                is_correct=False,
                correctness_level=CorrectnessLevel.INVALID_FORMAT,
                confidence=1.0,
                answer_type=AnswerType.ALGEBRAIC,
                student_answer_parsed=student_answer,
                expected_answer_parsed=expected_answer,
                partial_credit=0.0,
                error_details=f"Could not parse algebraic expressions: {str(e)}",
                validation_method="symbolic_parsing"
            )

class FractionValidator(BaseValidator):
    """Validator for fraction answers."""
    
    def can_handle(self, answer_type: AnswerType) -> bool:
        return answer_type == AnswerType.FRACTION
    
    def validate(self, student_answer: str, expected_answer: str, 
                question: str = "") -> ValidationResult:
        """Validate fraction answers."""
        try:
            student_cleaned = self._clean_answer(student_answer)
            expected_cleaned = self._clean_answer(expected_answer)
            
            # Parse fractions
            student_fraction = Fraction(student_cleaned.replace(' ', ''))
            expected_fraction = Fraction(expected_cleaned.replace(' ', ''))
            
            if student_fraction == expected_fraction:
                return ValidationResult(
                    is_correct=True,
                    correctness_level=CorrectnessLevel.CORRECT,
                    confidence=1.0,
                    answer_type=AnswerType.FRACTION,
                    student_answer_parsed=str(student_fraction),
                    expected_answer_parsed=str(expected_fraction),
                    partial_credit=1.0,
                    equivalent_forms=[str(float(student_fraction)), f"{student_fraction.numerator}/{student_fraction.denominator}"],
                    validation_method="fraction_equivalence"
                )
            else:
                return ValidationResult(
                    is_correct=False,
                    correctness_level=CorrectnessLevel.INCORRECT,
                    confidence=1.0,
                    answer_type=AnswerType.FRACTION,
                    student_answer_parsed=str(student_fraction),
                    expected_answer_parsed=str(expected_fraction),
                    partial_credit=0.0,
                    error_details=f"Fractions are not equal: {student_fraction} ≠ {expected_fraction}",
                    validation_method="fraction_equivalence"
                )
                
        except Exception as e:
            return ValidationResult(
                is_correct=False,
                correctness_level=CorrectnessLevel.INVALID_FORMAT,
                confidence=1.0,
                answer_type=AnswerType.FRACTION,
                student_answer_parsed=student_answer,
                expected_answer_parsed=expected_answer,
                partial_credit=0.0,
                error_details=f"Could not parse as fractions: {str(e)}",
                validation_method="fraction_parsing"
            )

class EquationValidator(BaseValidator):
    """Validator for equation answers."""
    
    def can_handle(self, answer_type: AnswerType) -> bool:
        return answer_type == AnswerType.EQUATION
    
    def validate(self, student_answer: str, expected_answer: str, 
                question: str = "") -> ValidationResult:
        """Validate equation answers."""
        try:
            student_cleaned = self._clean_answer(student_answer)
            expected_cleaned = self._clean_answer(expected_answer)
            
            # Parse equations
            if '=' not in student_cleaned or '=' not in expected_cleaned:
                return ValidationResult(
                    is_correct=False,
                    correctness_level=CorrectnessLevel.INVALID_FORMAT,
                    confidence=1.0,
                    answer_type=AnswerType.EQUATION,
                    student_answer_parsed=student_answer,
                    expected_answer_parsed=expected_answer,
                    partial_credit=0.0,
                    error_details="Not valid equations (missing equals sign)",
                    validation_method="equation_format"
                )
            
            # Split equations and parse both sides
            student_left, student_right = student_cleaned.split('=', 1)
            expected_left, expected_right = expected_cleaned.split('=', 1)
            
            student_eq = Eq(sympify(student_left.strip()), sympify(student_right.strip()))
            expected_eq = Eq(sympify(expected_left.strip()), sympify(expected_right.strip()))
            
            # Check if equations are equivalent
            if student_eq.equals(expected_eq):
                return ValidationResult(
                    is_correct=True,
                    correctness_level=CorrectnessLevel.CORRECT,
                    confidence=1.0,
                    answer_type=AnswerType.EQUATION,
                    student_answer_parsed=str(student_eq),
                    expected_answer_parsed=str(expected_eq),
                    partial_credit=1.0,
                    validation_method="equation_equivalence"
                )
            else:
                # Check if they have the same solution set
                try:
                    student_solutions = solve(student_eq)
                    expected_solutions = solve(expected_eq)
                    
                    if set(student_solutions) == set(expected_solutions):
                        return ValidationResult(
                            is_correct=True,
                            correctness_level=CorrectnessLevel.CORRECT,
                            confidence=0.9,
                            answer_type=AnswerType.EQUATION,
                            student_answer_parsed=str(student_eq),
                            expected_answer_parsed=str(expected_eq),
                            partial_credit=1.0,
                            error_details="Equations have same solution set",
                            validation_method="solution_set_equivalence"
                        )
                except:
                    pass
                
                return ValidationResult(
                    is_correct=False,
                    correctness_level=CorrectnessLevel.INCORRECT,
                    confidence=0.9,
                    answer_type=AnswerType.EQUATION,
                    student_answer_parsed=str(student_eq),
                    expected_answer_parsed=str(expected_eq),
                    partial_credit=0.0,
                    error_details="Equations are not equivalent",
                    validation_method="equation_equivalence"
                )
                
        except Exception as e:
            return ValidationResult(
                is_correct=False,
                correctness_level=CorrectnessLevel.INVALID_FORMAT,
                confidence=1.0,
                answer_type=AnswerType.EQUATION,
                student_answer_parsed=student_answer,
                expected_answer_parsed=expected_answer,
                partial_credit=0.0,
                error_details=f"Could not parse equations: {str(e)}",
                validation_method="equation_parsing"
            )

class TextValidator(BaseValidator):
    """Validator for text-based answers."""
    
    def can_handle(self, answer_type: AnswerType) -> bool:
        return answer_type in [AnswerType.WORD_ANSWER, AnswerType.MULTIPLE_CHOICE]
    
    def validate(self, student_answer: str, expected_answer: str, 
                question: str = "") -> ValidationResult:
        """Validate text-based answers."""
        student_cleaned = self._clean_answer(student_answer).lower()
        expected_cleaned = self._clean_answer(expected_answer).lower()
        
        # Exact match
        if student_cleaned == expected_cleaned:
            return ValidationResult(
                is_correct=True,
                correctness_level=CorrectnessLevel.CORRECT,
                confidence=1.0,
                answer_type=AnswerType.WORD_ANSWER,
                student_answer_parsed=student_cleaned,
                expected_answer_parsed=expected_cleaned,
                partial_credit=1.0,
                validation_method="exact_text_match"
            )
        
        # Check for partial matches
        student_words = set(student_cleaned.split())
        expected_words = set(expected_cleaned.split())
        
        if student_words and expected_words:
            common_words = student_words.intersection(expected_words)
            if len(common_words) >= len(expected_words) * 0.7:
                return ValidationResult(
                    is_correct=False,
                    correctness_level=CorrectnessLevel.PARTIALLY_CORRECT,
                    confidence=0.7,
                    answer_type=AnswerType.WORD_ANSWER,
                    student_answer_parsed=student_cleaned,
                    expected_answer_parsed=expected_cleaned,
                    partial_credit=0.7,
                    error_details="Partially correct text answer",
                    validation_method="partial_text_match"
                )
        
        return ValidationResult(
            is_correct=False,
            correctness_level=CorrectnessLevel.INCORRECT,
            confidence=1.0,
            answer_type=AnswerType.WORD_ANSWER,
            student_answer_parsed=student_cleaned,
            expected_answer_parsed=expected_cleaned,
            partial_credit=0.0,
            error_details="Text answers do not match",
            validation_method="text_comparison"
        )

class AnswerCorrectnessEngine:
    """Main engine for determining answer correctness using multiple validators."""
    
    def __init__(self):
        self.validators = [
            NumericalValidator(),
            AlgebraicValidator(),
            FractionValidator(),
            EquationValidator(),
            TextValidator()
        ]
    
    def check_correctness(self, student_answer: str, expected_answer: str, 
                         question: str = "") -> ValidationResult:
        """
        Check if a student's answer is correct using appropriate validators.
        
        Args:
            student_answer: The student's submitted answer
            expected_answer: The expected correct answer
            question: Additional context about the question
            
        Returns:
            ValidationResult with detailed evaluation
        """
        try:
            # Determine answer type
            answer_type = self._determine_answer_type(expected_answer, question)
            
            # Find appropriate validators
            applicable_validators = [v for v in self.validators if v.can_handle(answer_type)]
            
            if not applicable_validators:
                return ValidationResult(
                    is_correct=False,
                    correctness_level=CorrectnessLevel.UNABLE_TO_EVALUATE,
                    confidence=0.0,
                    answer_type=answer_type,
                    student_answer_parsed=student_answer,
                    expected_answer_parsed=expected_answer,
                    partial_credit=0.0,
                    error_details=f"No validator available for answer type: {answer_type}",
                    validation_method="no_validator"
                )
            
            # Use the first applicable validator
            validator = applicable_validators[0]
            result = validator.validate(student_answer, expected_answer, question)
            
            return result
            
        except Exception as e:
            return ValidationResult(
                is_correct=False,
                correctness_level=CorrectnessLevel.UNABLE_TO_EVALUATE,
                confidence=0.0,
                answer_type=AnswerType.NUMERICAL,
                student_answer_parsed=student_answer,
                expected_answer_parsed=expected_answer,
                partial_credit=0.0,
                error_details=f"Error during validation: {str(e)}",
                validation_method="error_handling"
            )
    
    def _determine_answer_type(self, answer: str, context: str) -> AnswerType:
        """Determine the type of mathematical answer."""
        context_lower = context.lower()
        answer_lower = answer.lower()
        
        # Check for equations (contains equals sign)
        if '=' in answer and not answer.startswith('='):
            return AnswerType.EQUATION
        
        # Check for fractions
        if '/' in answer and re.match(r'^\d+/\d+$', answer.replace(' ', '')):
            return AnswerType.FRACTION
        
        # Check for algebraic expressions (contains variables)
        if re.search(r'[a-z]', answer_lower) and not any(word in answer_lower for word in ['pi', 'sqrt', 'sin', 'cos', 'tan']):
            return AnswerType.ALGEBRAIC
        
        # Check for pure numbers
        try:
            float(answer.replace(' ', ''))
            return AnswerType.NUMERICAL
        except ValueError:
            pass
        
        # Check for mathematical expressions
        if any(op in answer for op in ['+', '-', '*', '/', '^', 'sqrt', 'sin', 'cos']):
            return AnswerType.EXPRESSION
        
        return AnswerType.WORD_ANSWER 