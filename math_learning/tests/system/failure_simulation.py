"""
Failure Simulation Engine
Generates realistic student errors mapped to specific concept gaps
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FailurePattern:
    """Represents a specific type of student error"""
    error_type: str
    wrong_answer: str
    missing_concept: str
    explanation: str
    frequency: float  # How common this error is (0.0 to 1.0)

class FailureSimulator:
    """Simulates realistic student failures for diagnostic testing"""
    
    def __init__(self):
        self.failure_patterns = self._initialize_failure_patterns()
    
    def _initialize_failure_patterns(self) -> Dict[str, List[FailurePattern]]:
        """Initialize comprehensive failure patterns for each concept"""
        return {
            # Counting Sequence: Count from 1 to 10: 1, 2, 3, ?, 5
            "counting": [
                FailurePattern(
                    error_type="skipped_number",
                    wrong_answer="6",
                    missing_concept="COUNTING_SEQUENCE",
                    explanation="Skipped the missing number and continued sequence",
                    frequency=0.40
                ),
                FailurePattern(
                    error_type="random_number",
                    wrong_answer="7",
                    missing_concept="COUNTING_SEQUENCE",
                    explanation="Provided random number instead of following sequence",
                    frequency=0.30
                ),
                FailurePattern(
                    error_type="backward_counting",
                    wrong_answer="2",
                    missing_concept="COUNTING_SEQUENCE",
                    explanation="Counted backward instead of forward",
                    frequency=0.20
                ),
                FailurePattern(
                    error_type="off_by_one",
                    wrong_answer="5",
                    missing_concept="COUNTING_SEQUENCE",
                    explanation="Off by one error in sequence",
                    frequency=0.10
                )
            ],
            
            # Place Value: What is the value of the digit 5 in 352?
            "place_value": [
                FailurePattern(
                    error_type="digit_value_confusion",
                    wrong_answer="5",
                    missing_concept="PLACE_VALUE",
                    explanation="Gave digit value instead of place value",
                    frequency=0.50
                ),
                FailurePattern(
                    error_type="wrong_place_value",
                    wrong_answer="500",
                    missing_concept="PLACE_VALUE",
                    explanation="Confused tens place with hundreds place",
                    frequency=0.30
                ),
                FailurePattern(
                    error_type="position_counting_error",
                    wrong_answer="5000",
                    missing_concept="PLACE_VALUE",
                    explanation="Counted position incorrectly",
                    frequency=0.20
                )
            ],
            
            # Commutative Property: Which shows the commutative property? A) 3 + 5 = 5 + 3  B) 3 + 0 = 3
            "commutative_property": [
                FailurePattern(
                    error_type="identity_property_confusion",
                    wrong_answer="B",
                    missing_concept="COMMUTATIVE_PROPERTY",
                    explanation="Confused identity property (adding zero) with commutative property",
                    frequency=0.60
                ),
                FailurePattern(
                    error_type="random_choice",
                    wrong_answer="B",
                    missing_concept="COMMUTATIVE_PROPERTY",
                    explanation="Made random choice without understanding properties",
                    frequency=0.40
                )
            ],
            
            # Two-Step Linear Equations: 3x + 7 = 22
            "two_step_equations": [
                FailurePattern(
                    error_type="addition_instead_of_subtraction",
                    wrong_answer="x = 29",
                    missing_concept="ONE_STEP_EQUATIONS",
                    explanation="🚨 Added 7 instead of subtracting: 3x = 22 + 7 = 29",
                    frequency=0.40
                ),
                FailurePattern(
                    error_type="forgot_to_divide_by_coefficient",
                    wrong_answer="x = 8",
                    missing_concept="ONE_STEP_EQUATIONS",
                    explanation="🚨 Subtracted correctly (2x = 8) but forgot to divide by coefficient",
                    frequency=0.35
                ),
                FailurePattern(
                    error_type="calculation_error_in_division",
                    wrong_answer="x = 3.5",
                    missing_concept="INTEGER_OPERATIONS",
                    explanation="🚨 Set up equation correctly (4x = 20) but made division error: 20 ÷ 4 = 3.5",
                    frequency=0.25
                )
            ],
            
            # Distributive Property: 2(x + 3) + 4x
            "distributive_property": [
                FailurePattern(
                    error_type="forgot_to_distribute_simple",
                    wrong_answer="2x + 3",
                    missing_concept="DISTRIBUTIVE_PROPERTY",
                    explanation="🚨 Didn't distribute the 2 to the constant: wrote 2x + 3 instead of 2x + 6",
                    frequency=0.50
                ),
                FailurePattern(
                    error_type="distributed_to_variable_only",
                    wrong_answer="6x + 4",
                    missing_concept="DISTRIBUTIVE_PROPERTY",
                    explanation="🚨 Distributed to variable (3×2x=6x) but not to constant (kept 4 instead of 3×4=12)",
                    frequency=0.30
                ),
                FailurePattern(
                    error_type="partial_distribution",
                    wrong_answer="2x + 6",
                    missing_concept="DISTRIBUTIVE_PROPERTY",
                    explanation="🚨 Distributed to constant but not variable: 2(x + 3) = 2x + 6",
                    frequency=0.20
                ),
                FailurePattern(
                    error_type="no_factoring_recognition",
                    wrong_answer="4x + 8",
                    missing_concept="FACTORING",
                    explanation="Didn't recognize factoring opportunity, left expression unchanged",
                    frequency=0.40
                ),
                FailurePattern(
                    error_type="distributed_incorrectly",
                    wrong_answer="2x + 6 + 4x",
                    missing_concept="COMBINE_LIKE_TERMS",
                    explanation="Distributed correctly but didn't combine like terms",
                    frequency=0.30
                ),
                FailurePattern(
                    error_type="multiplication_error",
                    wrong_answer="2x + 5 + 4x",
                    missing_concept="MULTIPLICATION_FACTS",
                    explanation="Error in 2 × 3 = 5 instead of 6",
                    frequency=0.20
                ),
                FailurePattern(
                    error_type="added_coefficients_wrong",
                    wrong_answer="7x + 6",
                    missing_concept="ADDITION_FACTS",
                    explanation="Added coefficients incorrectly: 2 + 4 = 7 instead of 6",
                    frequency=0.10
                )
            ],
            
            # Function Composition: f(g(x)) where f(x) = x + 2, g(x) = 3x
            "function_composition": [
                FailurePattern(
                    error_type="order_confusion",
                    wrong_answer="11",
                    missing_concept="FUNCTION_COMPOSITION",
                    explanation="🚨 Calculated g(f(1)) = g(3) = 9 instead of f(g(1)) = f(3) = 5",
                    frequency=0.40
                ),
                FailurePattern(
                    error_type="calculation_error_in_composition",
                    wrong_answer="7",
                    missing_concept="FUNCTION_EVALUATION",
                    explanation="🚨 Calculated h(k(4)) = h(3) = 2(3) = 6, then added 1 to get 7",
                    frequency=0.35
                ),
                FailurePattern(
                    error_type="no_factoring_recognition",
                    wrong_answer="2x + 3 + 4x",
                    missing_concept="DISTRIBUTIVE_PROPERTY",
                    explanation="Didn't recognize factoring opportunity, left expression unchanged",
                    frequency=0.40
                )
            ],
            
            # Fraction simplification: 6/8
            "fraction_simplification": [
                FailurePattern(
                    error_type="no_simplification_recognition",
                    wrong_answer="6/8",
                    missing_concept="FRACTION_SIMPLIFICATION",
                    explanation="🚨 Didn't recognize that 6/8 can be simplified by dividing both by 2",
                    frequency=0.50
                )
            ],
            
            # Fraction operations: 2/3 × 1.5
            "fraction_operations": [
                FailurePattern(
                    error_type="added_instead_of_multiplying",
                    wrong_answer="2.17",
                    missing_concept="FRACTIONS_AS_PART_WHOLE",
                    explanation="🚨 Added 2/3 + 1.5 = 0.67 + 1.5 = 2.17 instead of multiplying",
                    frequency=0.45
                )
            ],
            
            # Order of Operations: 3 + 4 × 2 = ?
            "order_of_operations": [
                FailurePattern(
                    error_type="left_to_right_calculation",
                    wrong_answer="14",
                    missing_concept="ORDER_OF_OPERATIONS",
                    explanation="🚨 Calculated left to right: (3 + 4) × 2 = 14",
                    frequency=0.45
                ),
                FailurePattern(
                    error_type="addition_before_multiplication",
                    wrong_answer="20",
                    missing_concept="ORDER_OF_OPERATIONS",
                    explanation="🚨 Added before multiplying: 2 + 3 = 5, then 5 × 4 = 20",
                    frequency=0.35
                ),
                FailurePattern(
                    error_type="subtraction_error_with_parentheses",
                    wrong_answer="18",
                    missing_concept="ORDER_OF_OPERATIONS",
                    explanation="🚨 Calculated (2 + 3) × 4 = 20, but made error in final subtraction",
                    frequency=0.20
                )
            ],
            
            # Integer Operations: -5 + (-3)
            "integer_operations": [
                FailurePattern(
                    error_type="treated_as_subtraction_of_absolute_values",
                    wrong_answer="2",
                    missing_concept="INTEGER_OPERATIONS",
                    explanation="🚨 Treated as subtraction of absolute values: |5| - |3| = 2",
                    frequency=0.40
                ),
                FailurePattern(
                    error_type="added_absolute_values_with_negative_result",
                    wrong_answer="-11",
                    missing_concept="INTEGER_OPERATIONS",
                    explanation="🚨 Added absolute values and made result negative: -(4 + 7) = -11",
                    frequency=0.30
                ),
                FailurePattern(
                    error_type="subtracted_instead_of_adding",
                    wrong_answer="5",
                    missing_concept="INTEGER_OPERATIONS",
                    explanation="🚨 Subtracted instead of adding: 8 - 3 = 5 (ignored negative sign)",
                    frequency=0.30
                )
            ],
            
            # Function evaluation: f(x) = 2x + 3, find f(4)
            "function_evaluation": [
                FailurePattern(
                    error_type="calculation_error_in_evaluation",
                    wrong_answer="14",
                    missing_concept="FUNCTION_EVALUATION",
                    explanation="🚨 Set up correctly f(4) = 2(4) + 3 but calculated 2×4=8, 8+3=14 instead of 11",
                    frequency=0.40
                ),
                FailurePattern(
                    error_type="didnt_square_input_correctly",
                    wrong_answer="6",
                    missing_concept="EXPONENT_RULES",
                    explanation="🚨 Calculated g(3) = 3 + 3 - 1 = 6 instead of 3² - 1 = 8",
                    frequency=0.35
                ),
                FailurePattern(
                    error_type="minor_calculation_error_in_composition",
                    wrong_answer="7",
                    missing_concept="FUNCTION_EVALUATION",
                    explanation="🚨 Calculated h(k(4)) = h(3) = 2(3) = 6, then added 1 to get 7",
                    frequency=0.25
                )
            ],
            
            # Quadratic equations: x² - 5x + 6 = 0
            "quadratic_equations": [
                FailurePattern(
                    error_type="incorrect_factoring",
                    wrong_answer="x = 1, 6",
                    missing_concept="FACTORING",
                    explanation="🚨 Incorrectly factored as (x-1)(x-6) instead of (x-2)(x-3)",
                    frequency=0.40
                ),
                FailurePattern(
                    error_type="sign_error_in_vertex",
                    wrong_answer="(-2, 15)",
                    missing_concept="COMPLETING_THE_SQUARE",
                    explanation="🚨 Used x = -(-4)/2 = -2 instead of x = -(-4)/2 = 2 for vertex",
                    frequency=0.35
                )
            ],
            
            # Basic arithmetic: 3 + 2 = ?
            "basic_arithmetic": [
                FailurePattern(
                    error_type="counting_error",
                    wrong_answer="6",
                    missing_concept="ADDITION_FACTS",
                    explanation="🚨 Counted incorrectly: 3 + 2 = 6 instead of 5",
                    frequency=0.40
                )
            ]
        }
    
    def simulate_failure(self, question_type: str, student_ability: float = 0.5) -> Optional[FailurePattern]:
        """
        Simulate a realistic failure for a given question type
        
        Args:
            question_type: Type of question (e.g., "two_step_equations")
            student_ability: Student's ability level (0.0 = struggling, 1.0 = expert)
            
        Returns:
            FailurePattern if failure occurs, None if student gets it right
        """
        if question_type not in self.failure_patterns:
            return None
        
        patterns = self.failure_patterns[question_type]
        
        # Higher ability students are less likely to make errors
        failure_probability = 1.0 - student_ability
        
        if random.random() > failure_probability:
            return None  # Student got it right
        
        # Weight patterns by frequency and student ability
        weighted_patterns = []
        for pattern in patterns:
            # More basic errors are more likely for struggling students
            if student_ability < 0.3:
                weight = pattern.frequency * 1.5
            elif student_ability < 0.7:
                weight = pattern.frequency
            else:
                weight = pattern.frequency * 0.5
            
            weighted_patterns.extend([pattern] * int(weight * 100))
        
        return random.choice(weighted_patterns) if weighted_patterns else None
    
    def get_all_failures_for_concept(self, concept: str) -> List[FailurePattern]:
        """Get all possible failure patterns for a specific concept"""
        all_failures = []
        for question_type, patterns in self.failure_patterns.items():
            for pattern in patterns:
                if pattern.missing_concept == concept:
                    all_failures.append(pattern)
        return all_failures
    
    def analyze_student_answer(self, question_type: str, student_answer: str, correct_answer: str) -> Optional[FailurePattern]:
        """
        Analyze a student's wrong answer to identify the likely error pattern
        
        Args:
            question_type: Type of question
            student_answer: What the student wrote
            correct_answer: The correct answer
            
        Returns:
            Most likely FailurePattern that explains the error
        """
        if student_answer.strip() == correct_answer.strip():
            return None  # No error
        
        if question_type not in self.failure_patterns:
            return None
        
        # Find the pattern that best matches the student's answer
        patterns = self.failure_patterns[question_type]
        for pattern in patterns:
            if pattern.wrong_answer.strip() == student_answer.strip():
                return pattern
        
        # If no exact match, return the most common error for this question type
        most_common = max(patterns, key=lambda p: p.frequency)
        return FailurePattern(
            error_type="unknown_error",
            wrong_answer=student_answer,
            missing_concept=most_common.missing_concept,
            explanation=f"Unexpected answer '{student_answer}', likely related to {most_common.missing_concept}",
            frequency=0.1
        )

def demonstrate_failure_simulation():
    """Demonstrate the failure simulation system"""
    simulator = FailureSimulator()
    
    print("🚨 FAILURE SIMULATION DEMONSTRATION")
    print("=" * 60)
    
    # Test different question types
    question_types = [
        ("two_step_equations", "Solve: 3x + 7 = 22"),
        ("distributive_property", "Simplify: 2(x + 3) + 4x"),
        ("function_composition", "Find f(g(1)) where f(x) = x + 2, g(x) = 3x"),
        ("order_of_operations", "Calculate: 3 + 4 × 2")
    ]
    
    for question_type, question_text in question_types:
        print(f"\n📝 QUESTION: {question_text}")
        print(f"   Type: {question_type}")
        
        # Simulate failures for different ability levels
        for ability_level, level_name in [(0.2, "Struggling"), (0.5, "Average"), (0.8, "Strong")]:
            failure = simulator.simulate_failure(question_type, ability_level)
            
            if failure:
                print(f"\n   🚨 {level_name} Student Error:")
                print(f"      Wrong Answer: '{failure.wrong_answer}'")
                print(f"      Missing Concept: {failure.missing_concept}")
                print(f"      Explanation: {failure.explanation}")
                print(f"      Error Frequency: {failure.frequency:.1%}")
            else:
                print(f"\n   ✅ {level_name} Student: Got it right!")

if __name__ == "__main__":
    demonstrate_failure_simulation() 