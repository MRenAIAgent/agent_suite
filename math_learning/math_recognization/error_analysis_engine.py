"""
Error Analysis Engine

This module provides comprehensive error analysis for mathematical problems,
classifying error types and identifying root causes for incorrect answers.
"""

import re
import sympy as sp
from sympy import sympify, simplify
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

class ErrorType(Enum):
    """Types of mathematical errors."""
    ARITHMETIC = "arithmetic"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    NOTATION = "notation"
    CARELESS = "careless"
    READING_COMPREHENSION = "reading_comprehension"
    INCOMPLETE = "incomplete"

class ErrorSeverity(Enum):
    """Severity levels of errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorPattern:
    """Represents a specific error pattern."""
    pattern_id: str
    error_type: ErrorType
    pattern_regex: str
    description: str
    common_causes: List[str]
    missing_concepts: List[str]
    severity: ErrorSeverity
    examples: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class ErrorAnalysisResult:
    """Result of error analysis."""
    error_type: ErrorType
    error_subtype: str
    root_cause: str
    missing_concepts: List[str]
    confidence: float
    severity: ErrorSeverity
    description: str
    suggested_remediation: str
    diagnostic_questions: List[str]
    prerequisite_gaps: List[str] = field(default_factory=list)
    work_analysis: Optional[Dict[str, Any]] = None

class ArithmeticErrorDetector:
    """Detector for arithmetic calculation errors."""
    
    def __init__(self):
        self.error_patterns = {
            'addition_error': {
                'pattern': r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)',
                'operation': lambda a, b: a + b,
                'concepts': ['addition', 'basic_arithmetic']
            },
            'subtraction_error': {
                'pattern': r'(\d+)\s*-\s*(\d+)\s*=\s*(\d+)',
                'operation': lambda a, b: a - b,
                'concepts': ['subtraction', 'basic_arithmetic']
            },
            'multiplication_error': {
                'pattern': r'(\d+)\s*[×*]\s*(\d+)\s*=\s*(\d+)',
                'operation': lambda a, b: a * b,
                'concepts': ['multiplication', 'multiplication_tables']
            },
            'division_error': {
                'pattern': r'(\d+)\s*[÷/]\s*(\d+)\s*=\s*(\d+)',
                'operation': lambda a, b: a / b if b != 0 else float('inf'),
                'concepts': ['division', 'basic_arithmetic']
            }
        }
    
    def detect_arithmetic_error(self, question: str, student_answer: str, 
                              correct_answer: str, work_shown: str = "") -> Optional[ErrorAnalysisResult]:
        """Detect arithmetic calculation errors."""
        text_to_analyze = f"{question} {student_answer} {work_shown}"
        
        for error_name, pattern_info in self.error_patterns.items():
            matches = re.findall(pattern_info['pattern'], text_to_analyze)
            
            for match in matches:
                try:
                    a, b, student_result = int(match[0]), int(match[1]), int(match[2])
                    correct_result = pattern_info['operation'](a, b)
                    
                    if abs(student_result - correct_result) > 0.001:
                        # Analyze the type of arithmetic error
                        error_subtype = self._classify_arithmetic_error(a, b, student_result, correct_result, error_name)
                        
                        return ErrorAnalysisResult(
                            error_type=ErrorType.ARITHMETIC,
                            error_subtype=error_subtype,
                            root_cause=self._identify_arithmetic_root_cause(a, b, student_result, correct_result, error_name),
                            missing_concepts=pattern_info['concepts'],
                            confidence=0.9,
                            severity=ErrorSeverity.LOW if abs(student_result - correct_result) < 5 else ErrorSeverity.MEDIUM,
                            description=f"Arithmetic error in {error_name.replace('_', ' ')}: {a} {self._get_operation_symbol(error_name)} {b} = {student_result} (should be {correct_result})",
                            suggested_remediation=f"Practice {error_name.replace('_', ' ')} with similar problems",
                            diagnostic_questions=[
                                f"What is {a} {self._get_operation_symbol(error_name)} {b}?",
                                f"Can you show me step by step how you calculated this?"
                            ]
                        )
                except (ValueError, ZeroDivisionError):
                    continue
        
        return None
    
    def _classify_arithmetic_error(self, a: int, b: int, student_result: int, 
                                 correct_result: float, operation: str) -> str:
        """Classify the specific type of arithmetic error."""
        if operation == 'multiplication_error':
            # Check for common multiplication errors
            if student_result == a + b:
                return "addition_instead_of_multiplication"
            elif student_result == abs(a - b):
                return "subtraction_instead_of_multiplication"
            elif student_result in [a * (b + 1), a * (b - 1), (a + 1) * b, (a - 1) * b]:
                return "off_by_one_factor"
            else:
                return "multiplication_fact_error"
        elif operation == 'addition_error':
            if abs(student_result - correct_result) == 1:
                return "off_by_one_addition"
            else:
                return "addition_calculation_error"
        elif operation == 'subtraction_error':
            if student_result == b - a:  # Reversed operands
                return "operand_reversal"
            else:
                return "subtraction_calculation_error"
        elif operation == 'division_error':
            if student_result == a * b:
                return "multiplication_instead_of_division"
            else:
                return "division_calculation_error"
        
        return "general_arithmetic_error"
    
    def _identify_arithmetic_root_cause(self, a: int, b: int, student_result: int, 
                                      correct_result: float, operation: str) -> str:
        """Identify the root cause of arithmetic error."""
        error_subtype = self._classify_arithmetic_error(a, b, student_result, correct_result, operation)
        
        root_causes = {
            "multiplication_fact_error": "Student doesn't know multiplication tables",
            "addition_instead_of_multiplication": "Student confused operation symbols",
            "off_by_one_addition": "Careless counting error",
            "operand_reversal": "Student reversed the order of numbers",
            "general_arithmetic_error": "Basic calculation mistake"
        }
        
        return root_causes.get(error_subtype, "Arithmetic calculation error")
    
    def _get_operation_symbol(self, operation: str) -> str:
        """Get the symbol for the operation."""
        symbols = {
            'addition_error': '+',
            'subtraction_error': '-',
            'multiplication_error': '×',
            'division_error': '÷'
        }
        return symbols.get(operation, '?')

class ConceptualErrorDetector:
    """Detector for conceptual understanding errors."""
    
    def __init__(self):
        self.concept_patterns = {
            'negative_exponents': {
                'indicators': [r'x\^{-\d+}', r'x\^-\d+', r'negative.*exponent'],
                'common_errors': ['treating_as_positive', 'multiplying_by_negative'],
                'concepts': ['exponent_rules', 'negative_exponents']
            },
            'fraction_operations': {
                'indicators': [r'\d+/\d+\s*[+\-*/]\s*\d+/\d+', r'fraction.*add', r'fraction.*multiply'],
                'common_errors': ['adding_denominators', 'cross_multiplying_addition'],
                'concepts': ['fraction_addition', 'fraction_multiplication', 'common_denominators']
            },
            'order_of_operations': {
                'indicators': [r'.*[+\-*/].*[+\-*/].*', r'PEMDAS', r'BODMAS'],
                'common_errors': ['left_to_right_only', 'ignoring_parentheses'],
                'concepts': ['order_of_operations', 'PEMDAS']
            },
            'algebraic_distribution': {
                'indicators': [r'\([^)]+\)\s*[+\-*/]\s*\([^)]+\)', r'distribute', r'expand'],
                'common_errors': ['forgetting_middle_term', 'incorrect_signs'],
                'concepts': ['distributive_property', 'algebraic_expansion']
            }
        }
    
    def detect_conceptual_error(self, question: str, student_answer: str, 
                              correct_answer: str, work_shown: str = "") -> Optional[ErrorAnalysisResult]:
        """Detect conceptual understanding errors."""
        combined_text = f"{question} {student_answer} {work_shown}".lower()
        
        for concept, pattern_info in self.concept_patterns.items():
            # Check if this concept is relevant to the problem
            if any(re.search(indicator, combined_text) for indicator in pattern_info['indicators']):
                # Analyze the specific conceptual error
                error_analysis = self._analyze_concept_error(
                    concept, question, student_answer, correct_answer, work_shown
                )
                
                if error_analysis:
                    return error_analysis
        
        return None
    
    def _analyze_concept_error(self, concept: str, question: str, student_answer: str, 
                             correct_answer: str, work_shown: str) -> Optional[ErrorAnalysisResult]:
        """Analyze a specific conceptual error."""
        pattern_info = self.concept_patterns[concept]
        
        # Specific analysis based on concept type
        if concept == 'fraction_operations':
            return self._analyze_fraction_error(question, student_answer, correct_answer, pattern_info)
        elif concept == 'negative_exponents':
            return self._analyze_exponent_error(question, student_answer, correct_answer, pattern_info)
        elif concept == 'order_of_operations':
            return self._analyze_order_error(question, student_answer, correct_answer, pattern_info)
        elif concept == 'algebraic_distribution':
            return self._analyze_distribution_error(question, student_answer, correct_answer, pattern_info)
        
        return None
    
    def _analyze_fraction_error(self, question: str, student_answer: str, 
                              correct_answer: str, pattern_info: Dict) -> Optional[ErrorAnalysisResult]:
        """Analyze fraction operation errors."""
        # Look for common fraction errors
        if '+' in question and '/' in question:
            # Check if student added numerators and denominators separately
            fraction_pattern = r'(\d+)/(\d+)\s*\+\s*(\d+)/(\d+)'
            match = re.search(fraction_pattern, question)
            if match:
                n1, d1, n2, d2 = map(int, match.groups())
                
                # Check if student answer matches "wrong" addition
                wrong_addition = f"{n1 + n2}/{d1 + d2}"
                if wrong_addition in student_answer.replace(' ', ''):
                    return ErrorAnalysisResult(
                        error_type=ErrorType.CONCEPTUAL,
                        error_subtype="fraction_addition_misconception",
                        root_cause="Student added numerators and denominators separately",
                        missing_concepts=pattern_info['concepts'],
                        confidence=0.95,
                        severity=ErrorSeverity.HIGH,
                        description="Misconception: Adding fractions by adding numerators and denominators separately",
                        suggested_remediation="Review finding common denominators before adding fractions",
                        diagnostic_questions=[
                            "How do you add fractions with different denominators?",
                            "What is a common denominator?",
                            f"What is the LCD of {d1} and {d2}?"
                        ]
                    )
        
        return None
    
    def _analyze_exponent_error(self, question: str, student_answer: str, 
                              correct_answer: str, pattern_info: Dict) -> Optional[ErrorAnalysisResult]:
        """Analyze exponent rule errors."""
        # Check for negative exponent misconceptions
        if 'x^-' in question or 'x^{-' in question:
            try:
                # Try to evaluate student's understanding
                if 'x^-1' in question and ('x' in student_answer or '-x' in student_answer):
                    return ErrorAnalysisResult(
                        error_type=ErrorType.CONCEPTUAL,
                        error_subtype="negative_exponent_misconception",
                        root_cause="Student doesn't understand that x^-1 = 1/x",
                        missing_concepts=pattern_info['concepts'],
                        confidence=0.85,
                        severity=ErrorSeverity.HIGH,
                        description="Misconception about negative exponents",
                        suggested_remediation="Review the rule: x^-n = 1/x^n",
                        diagnostic_questions=[
                            "What does x^-1 equal?",
                            "How do you handle negative exponents?",
                            "What is the relationship between x^-n and x^n?"
                        ]
                    )
            except:
                pass
        
        return None
    
    def _analyze_order_error(self, question: str, student_answer: str, 
                           correct_answer: str, pattern_info: Dict) -> Optional[ErrorAnalysisResult]:
        """Analyze order of operations errors."""
        # Check for expressions with multiple operations
        if re.search(r'\d+\s*[+\-]\s*\d+\s*[*/]\s*\d+', question):
            return ErrorAnalysisResult(
                error_type=ErrorType.CONCEPTUAL,
                error_subtype="order_of_operations_error",
                root_cause="Student didn't follow correct order of operations",
                missing_concepts=pattern_info['concepts'],
                confidence=0.8,
                severity=ErrorSeverity.MEDIUM,
                description="Error in applying order of operations (PEMDAS/BODMAS)",
                suggested_remediation="Review PEMDAS: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction",
                diagnostic_questions=[
                    "What is the correct order of operations?",
                    "Which operations do you do first?",
                    "How do you handle expressions with multiple operations?"
                ]
            )
        
        return None
    
    def _analyze_distribution_error(self, question: str, student_answer: str, 
                                  correct_answer: str, pattern_info: Dict) -> Optional[ErrorAnalysisResult]:
        """Analyze algebraic distribution errors."""
        # Check for (a+b)^2 = a^2 + b^2 error
        if '(a+b)^2' in question.replace(' ', '') or '(x+y)^2' in question.replace(' ', ''):
            if 'a^2+b^2' in student_answer.replace(' ', '') or 'x^2+y^2' in student_answer.replace(' ', ''):
                return ErrorAnalysisResult(
                    error_type=ErrorType.CONCEPTUAL,
                    error_subtype="binomial_expansion_error",
                    root_cause="Student forgot the middle term in (a+b)^2 expansion",
                    missing_concepts=['binomial_expansion', 'distributive_property'],
                    confidence=0.9,
                    severity=ErrorSeverity.HIGH,
                    description="Misconception: (a+b)^2 = a^2 + b^2 (missing middle term)",
                    suggested_remediation="Review binomial expansion: (a+b)^2 = a^2 + 2ab + b^2",
                    diagnostic_questions=[
                        "What is (a+b)^2 equal to?",
                        "How do you expand (x+3)^2?",
                        "Why is there a middle term in binomial expansion?"
                    ]
                )
        
        return None

class ProceduralErrorDetector:
    """Detector for procedural/method errors."""
    
    def __init__(self):
        self.procedure_patterns = {
            'equation_solving': {
                'indicators': ['solve', 'equation', '=', 'x ='],
                'common_errors': ['not_balancing', 'wrong_inverse_operation'],
                'concepts': ['equation_solving', 'inverse_operations', 'algebraic_manipulation']
            },
            'factoring': {
                'indicators': ['factor', 'x^2', 'quadratic'],
                'common_errors': ['wrong_factoring_pattern', 'sign_errors'],
                'concepts': ['factoring', 'quadratic_expressions', 'algebraic_factoring']
            }
        }
    
    def detect_procedural_error(self, question: str, student_answer: str, 
                              correct_answer: str, work_shown: str = "") -> Optional[ErrorAnalysisResult]:
        """Detect procedural/method errors."""
        combined_text = f"{question} {student_answer} {work_shown}".lower()
        
        # Check for equation solving errors
        if 'solve' in combined_text and '=' in question:
            return self._analyze_equation_solving_error(question, student_answer, correct_answer, work_shown)
        
        # Check for factoring errors
        if 'factor' in combined_text or 'x^2' in question:
            return self._analyze_factoring_error(question, student_answer, correct_answer, work_shown)
        
        return None
    
    def _analyze_equation_solving_error(self, question: str, student_answer: str, 
                                      correct_answer: str, work_shown: str) -> Optional[ErrorAnalysisResult]:
        """Analyze equation solving procedural errors."""
        # Look for signs of not balancing the equation
        if work_shown:
            lines = work_shown.split('\n')
            for line in lines:
                if '=' in line:
                    # Check if operations were applied to both sides
                    left, right = line.split('=', 1)
                    # This is a simplified check - in practice, you'd need more sophisticated analysis
                    if len(left.strip()) != len(right.strip()):  # Very basic heuristic
                        return ErrorAnalysisResult(
                            error_type=ErrorType.PROCEDURAL,
                            error_subtype="equation_balancing_error",
                            root_cause="Student didn't apply operations to both sides of equation",
                            missing_concepts=['equation_solving', 'balance_property'],
                            confidence=0.7,
                            severity=ErrorSeverity.HIGH,
                            description="Procedural error: Not balancing equation when solving",
                            suggested_remediation="Remember: Whatever you do to one side, do to the other side",
                            diagnostic_questions=[
                                "How do you solve equations?",
                                "What does it mean to 'balance' an equation?",
                                "If you add 5 to the left side, what must you do to the right side?"
                            ]
                        )
        
        return None
    
    def _analyze_factoring_error(self, question: str, student_answer: str, 
                                correct_answer: str, work_shown: str) -> Optional[ErrorAnalysisResult]:
        """Analyze factoring procedural errors."""
        # Check for difference of squares errors
        if 'x^2 - ' in question and '(' in student_answer and ')' in student_answer:
            # Look for (x-a)(x-a) instead of (x-a)(x+a) pattern
            if student_answer.count('(x-') == 2 or student_answer.count('(x+') == 2:
                return ErrorAnalysisResult(
                    error_type=ErrorType.PROCEDURAL,
                    error_subtype="difference_of_squares_error",
                    root_cause="Student used wrong factoring pattern for difference of squares",
                    missing_concepts=['difference_of_squares', 'factoring_patterns'],
                    confidence=0.8,
                    severity=ErrorSeverity.MEDIUM,
                    description="Procedural error: Wrong pattern for difference of squares",
                    suggested_remediation="Review: x^2 - a^2 = (x-a)(x+a), not (x-a)(x-a)",
                    diagnostic_questions=[
                        "How do you factor x^2 - 9?",
                        "What is the difference of squares pattern?",
                        "Why do the signs differ in the factors?"
                    ]
                )
        
        return None

class ErrorAnalysisEngine:
    """Main engine for analyzing mathematical errors."""
    
    def __init__(self):
        self.arithmetic_detector = ArithmeticErrorDetector()
        self.conceptual_detector = ConceptualErrorDetector()
        self.procedural_detector = ProceduralErrorDetector()
    
    def analyze_error(self, question: str, student_answer: str, correct_answer: str, 
                     work_shown: str = "") -> ErrorAnalysisResult:
        """
        Analyze a mathematical error and determine its type and cause.
        
        Args:
            question: The original question
            student_answer: The student's incorrect answer
            correct_answer: The correct answer
            work_shown: Any work shown by the student
            
        Returns:
            ErrorAnalysisResult with detailed analysis
        """
        # Try different error detectors in order of specificity
        detectors = [
            self.arithmetic_detector.detect_arithmetic_error,
            self.conceptual_detector.detect_conceptual_error,
            self.procedural_detector.detect_procedural_error
        ]
        
        for detector in detectors:
            result = detector(question, student_answer, correct_answer, work_shown)
            if result:
                return result
        
        # If no specific error pattern found, provide general analysis
        return self._general_error_analysis(question, student_answer, correct_answer, work_shown)
    
    def _general_error_analysis(self, question: str, student_answer: str, 
                               correct_answer: str, work_shown: str) -> ErrorAnalysisResult:
        """Provide general error analysis when specific patterns aren't detected."""
        # Determine if it's likely a careless error (small difference) or conceptual
        try:
            student_num = float(re.search(r'-?\d+\.?\d*', student_answer).group())
            correct_num = float(re.search(r'-?\d+\.?\d*', correct_answer).group())
            
            relative_error = abs(student_num - correct_num) / abs(correct_num) if correct_num != 0 else float('inf')
            
            if relative_error < 0.1:  # Small error, likely careless
                return ErrorAnalysisResult(
                    error_type=ErrorType.CARELESS,
                    error_subtype="minor_calculation_error",
                    root_cause="Small calculation mistake, likely careless error",
                    missing_concepts=[],
                    confidence=0.6,
                    severity=ErrorSeverity.LOW,
                    description="Minor error, possibly due to carelessness",
                    suggested_remediation="Double-check calculations and work more carefully",
                    diagnostic_questions=[
                        "Can you check your calculation again?",
                        "What steps did you follow to get this answer?"
                    ]
                )
            else:
                return ErrorAnalysisResult(
                    error_type=ErrorType.CONCEPTUAL,
                    error_subtype="general_conceptual_error",
                    root_cause="Significant error suggesting conceptual misunderstanding",
                    missing_concepts=["problem_solving", "mathematical_reasoning"],
                    confidence=0.5,
                    severity=ErrorSeverity.MEDIUM,
                    description="Significant error suggesting deeper misunderstanding",
                    suggested_remediation="Review the underlying concepts for this problem type",
                    diagnostic_questions=[
                        "Can you explain your reasoning?",
                        "What method did you use to solve this?",
                        "Have you seen similar problems before?"
                    ]
                )
        except:
            # Non-numerical answer
            return ErrorAnalysisResult(
                error_type=ErrorType.READING_COMPREHENSION,
                error_subtype="answer_format_error",
                root_cause="Student may not have understood the question format",
                missing_concepts=["reading_comprehension", "problem_interpretation"],
                confidence=0.4,
                severity=ErrorSeverity.MEDIUM,
                description="Answer format suggests misunderstanding of question",
                suggested_remediation="Review question carefully and ensure understanding of what's being asked",
                diagnostic_questions=[
                    "Can you reread the question and tell me what it's asking?",
                    "What type of answer is expected?",
                    "Do you understand all the words in the question?"
                ]
            ) 