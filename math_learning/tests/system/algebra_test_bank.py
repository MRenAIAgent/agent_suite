"""
Comprehensive algebra test bank with real test questions and error analysis.
Maps student errors to specific concept gaps in the knowledge graph.
"""

from typing import List, Dict
from math_learning.testing.test_question import TestQuestion, QuestionType, DifficultyLevel, ErrorMapping

class AlgebraTestBank:
    """Collection of algebra test questions with comprehensive error analysis."""
    
    def __init__(self):
        self.questions = self._build_test_bank()
        self.questions_by_concept = self._index_by_concept()
        self.questions_by_difficulty = self._index_by_difficulty()
    
    def _build_test_bank(self) -> List[TestQuestion]:
        """Build comprehensive test bank with real algebra questions."""
        questions = []
        
        # === NUMBER SENSE & BASIC OPERATIONS ===
        
        questions.append(TestQuestion(
            id="NS_001",
            question_text="What is 3 + 4 × 2?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            difficulty=DifficultyLevel.BASIC,
            correct_answer="11",
            explanation="Order of operations: multiplication before addition. 4 × 2 = 8, then 3 + 8 = 11",
            options=["14", "11", "10", "24"],
            primary_concepts=["NS-15"],  # Order of Operations (PEMDAS)
            prerequisite_concepts=["NS-06", "NS-03"],  # Multiplication Facts, Addition
            grade_level="6-8",
            topic_area="Order of Operations",
            common_errors=[
                ErrorMapping(
                    wrong_answer="14",
                    concept_id="NS-15",
                    concept_name="Order of Operations (PEMDAS)",
                    error_description="Added first, then multiplied: (3 + 4) × 2 = 14",
                    remediation_hint="Remember PEMDAS: Multiplication comes before addition"
                ),
                ErrorMapping(
                    wrong_answer="10",
                    concept_id="NS-06",
                    concept_name="Multiplication Facts 0–12",
                    error_description="Calculation error in 4 × 2",
                    remediation_hint="Review basic multiplication facts"
                )
            ]
        ))
        
        questions.append(TestQuestion(
            id="NS_002",
            question_text="Simplify: -5 + (-3)",
            question_type=QuestionType.NUMERIC,
            difficulty=DifficultyLevel.BASIC,
            correct_answer="-8",
            explanation="Adding two negative numbers: -5 + (-3) = -5 - 3 = -8",
            primary_concepts=["NS-11"],  # Integer Addition/Subtraction
            prerequisite_concepts=["NS-01"],  # Counting
            grade_level="6-8",
            topic_area="Integer Operations",
            common_errors=[
                ErrorMapping(
                    wrong_answer="-2",
                    concept_id="NS-11",
                    concept_name="Integer Addition/Subtraction",
                    error_description="Subtracted instead of adding: -5 - (-3) = -2",
                    remediation_hint="When adding negatives, combine the absolute values and keep negative sign"
                ),
                ErrorMapping(
                    wrong_answer="8",
                    concept_id="NS-11",
                    concept_name="Integer Addition/Subtraction",
                    error_description="Ignored negative signs completely",
                    remediation_hint="Pay attention to positive and negative signs in integer operations"
                ),
                ErrorMapping(
                    wrong_answer="2",
                    concept_id="NS-11",
                    concept_name="Integer Addition/Subtraction",
                    error_description="Treated as subtraction: 5 - 3 = 2",
                    remediation_hint="Remember that adding a negative is the same as subtracting the positive"
                )
            ]
        ))
        
        # === ALGEBRAIC THINKING ===
        
        questions.append(TestQuestion(
            id="AT_001",
            question_text="Translate into an algebraic expression: 'Five more than twice a number'",
            question_type=QuestionType.ALGEBRAIC_EXPRESSION,
            difficulty=DifficultyLevel.INTERMEDIATE,
            correct_answer="2x + 5",
            explanation="'Twice a number' = 2x, 'five more than' means +5, so 2x + 5",
            primary_concepts=["AT-18"],  # Translate Expressions
            prerequisite_concepts=["AT-17", "NS-06"],  # Identify Like Terms, Multiplication
            grade_level="6-8",
            topic_area="Algebraic Expressions",
            common_errors=[
                ErrorMapping(
                    wrong_answer="5 + 2x",
                    concept_id="AT-18",
                    concept_name="Translate Expressions",
                    error_description="Correct terms but wrong order (mathematically equivalent but shows literal translation)",
                    remediation_hint="Both 2x + 5 and 5 + 2x are correct due to commutative property"
                ),
                ErrorMapping(
                    wrong_answer="2x - 5",
                    concept_id="AT-18",
                    concept_name="Translate Expressions",
                    error_description="Used subtraction instead of addition for 'more than'",
                    remediation_hint="'More than' indicates addition, not subtraction"
                ),
                ErrorMapping(
                    wrong_answer="5x + 2",
                    concept_id="AT-18",
                    concept_name="Translate Expressions",
                    error_description="Confused 'twice a number' (2x) with 'five times a number' (5x)",
                    remediation_hint="'Twice' means multiply by 2, so 'twice a number' is 2x"
                ),
                ErrorMapping(
                    wrong_answer="x + 5 + 2",
                    concept_id="AT-18",
                    concept_name="Translate Expressions",
                    error_description="Didn't understand 'twice a number' means 2x",
                    remediation_hint="'Twice a number' means 2 times the number, written as 2x"
                )
            ]
        ))
        
        questions.append(TestQuestion(
            id="AT_002",
            question_text="Combine like terms: 3x + 5 + 2x - 1",
            question_type=QuestionType.ALGEBRAIC_EXPRESSION,
            difficulty=DifficultyLevel.INTERMEDIATE,
            correct_answer="5x + 4",
            explanation="Combine x terms: 3x + 2x = 5x. Combine constants: 5 + (-1) = 4. Result: 5x + 4",
            primary_concepts=["AT-20"],  # Combine Like Terms
            prerequisite_concepts=["AT-17", "NS-11"],  # Identify Like Terms, Integer Addition
            grade_level="6-8",
            topic_area="Algebraic Expressions",
            common_errors=[
                ErrorMapping(
                    wrong_answer="5x + 6",
                    concept_id="NS-11",
                    concept_name="Integer Addition/Subtraction",
                    error_description="Added constants incorrectly: 5 + (-1) calculated as 5 + 1 = 6",
                    remediation_hint="When subtracting, remember 5 - 1 = 4, not 5 + 1 = 6"
                ),
                ErrorMapping(
                    wrong_answer="6x + 4",
                    concept_id="AT-17",
                    concept_name="Identify Like Terms",
                    error_description="Added coefficients incorrectly: 3 + 2 = 5, not 6",
                    remediation_hint="Carefully add the coefficients of like terms: 3x + 2x = 5x"
                ),
                ErrorMapping(
                    wrong_answer="3x + 7x",
                    concept_id="AT-20",
                    concept_name="Combine Like Terms",
                    error_description="Didn't combine like terms, just rewrote expression",
                    remediation_hint="Combine terms with the same variable: 3x + 2x = 5x, and 5 - 1 = 4"
                )
            ]
        ))
        
        # === LINEAR EQUATIONS ===
        
        questions.append(TestQuestion(
            id="LF_001",
            question_text="Solve for x: 2x + 3 = 11",
            question_type=QuestionType.NUMERIC,
            difficulty=DifficultyLevel.INTERMEDIATE,
            correct_answer="4",
            explanation="Subtract 3 from both sides: 2x = 8. Divide by 2: x = 4",
            primary_concepts=["AT-24"],  # Two-Step Linear Equations
            prerequisite_concepts=["AT-22", "AT-23"],  # One-Step Equations
            grade_level="6-8",
            topic_area="Linear Equations",
            common_errors=[
                ErrorMapping(
                    wrong_answer="7",
                    concept_id="AT-24",
                    concept_name="Two-Step Linear Equations",
                    error_description="Added 3 instead of subtracting: 2x = 11 + 3 = 14, x = 7",
                    remediation_hint="To isolate 2x, subtract 3 from both sides (inverse operation)"
                ),
                ErrorMapping(
                    wrong_answer="8",
                    concept_id="AT-24",
                    concept_name="Two-Step Linear Equations",
                    error_description="Forgot to divide by 2 in final step",
                    remediation_hint="After getting 2x = 8, divide both sides by 2 to get x = 4"
                ),
                ErrorMapping(
                    wrong_answer="2",
                    concept_id="AT-24",
                    concept_name="Two-Step Linear Equations",
                    error_description="Divided 8 by 4 instead of by 2",
                    remediation_hint="The coefficient of x is 2, so divide 8 by 2 to get x = 4"
                )
            ]
        ))
        
        questions.append(TestQuestion(
            id="LF_002",
            question_text="What is the slope of the line passing through points (2, 3) and (6, 11)?",
            question_type=QuestionType.NUMERIC,
            difficulty=DifficultyLevel.INTERMEDIATE,
            correct_answer="2",
            explanation="Slope = (y₂ - y₁)/(x₂ - x₁) = (11 - 3)/(6 - 2) = 8/4 = 2",
            primary_concepts=["LF-29"],  # Slope from Two Points
            prerequisite_concepts=["NS-07", "NS-10"],  # Division, Fractions
            grade_level="8-9",
            topic_area="Linear Functions",
            common_errors=[
                ErrorMapping(
                    wrong_answer="-2",
                    concept_id="LF-29",
                    concept_name="Slope from Two Points",
                    error_description="Used wrong order: (3 - 11)/(2 - 6) = -8/-4 = 2, but calculated as -2",
                    remediation_hint="Be careful with order: (y₂ - y₁)/(x₂ - x₁). Also, negative/negative = positive"
                ),
                ErrorMapping(
                    wrong_answer="1/2",
                    concept_id="LF-29",
                    concept_name="Slope from Two Points",
                    error_description="Inverted the slope formula: (x₂ - x₁)/(y₂ - y₁) = 4/8 = 1/2",
                    remediation_hint="Slope formula is rise over run: (y₂ - y₁)/(x₂ - x₁), not the inverse"
                ),
                ErrorMapping(
                    wrong_answer="8",
                    concept_id="LF-29",
                    concept_name="Slope from Two Points",
                    error_description="Calculated numerator correctly (8) but forgot to divide by denominator (4)",
                    remediation_hint="Complete the division: 8 ÷ 4 = 2"
                )
            ]
        ))
        
        # === QUADRATIC EQUATIONS ===
        
        questions.append(TestQuestion(
            id="Q_001",
            question_text="Solve using the quadratic formula: x² - 5x + 6 = 0",
            question_type=QuestionType.MULTIPLE_CHOICE,
            difficulty=DifficultyLevel.ADVANCED,
            correct_answer="x = 2, 3",
            explanation="Using x = (-b ± √(b² - 4ac))/2a with a=1, b=-5, c=6: x = (5 ± √(25-24))/2 = (5 ± 1)/2 = 3, 2",
            options=["x = 2, 3", "x = -2, -3", "x = 1, 6", "x = -1, -6"],
            primary_concepts=["Q-59"],  # Quadratic Formula
            prerequisite_concepts=["EXP-41", "NS-16"],  # Exponent Product Rule, Square Roots
            grade_level="9-10",
            topic_area="Quadratic Equations",
            common_errors=[
                ErrorMapping(
                    wrong_answer="x = -2, -3",
                    concept_id="Q-59",
                    concept_name="Quadratic Formula",
                    error_description="Sign error: used +b instead of -b in formula",
                    remediation_hint="Formula is x = (-b ± √(b² - 4ac))/2a. Note the negative sign before b"
                ),
                ErrorMapping(
                    wrong_answer="x = 1, 6",
                    concept_id="Q-59",
                    concept_name="Quadratic Formula",
                    error_description="Calculation error in discriminant or final arithmetic",
                    remediation_hint="Check discriminant: b² - 4ac = 25 - 24 = 1, so √1 = 1"
                )
            ]
        ))
        
        # === EXPONENTIAL EXPRESSIONS ===
        
        questions.append(TestQuestion(
            id="EXP_001",
            question_text="Simplify: x³ · x⁵",
            question_type=QuestionType.ALGEBRAIC_EXPRESSION,
            difficulty=DifficultyLevel.INTERMEDIATE,
            correct_answer="x⁸",
            explanation="When multiplying same base, add exponents: x³ · x⁵ = x³⁺⁵ = x⁸",
            primary_concepts=["EXP-41"],  # Exponent Product Rule
            prerequisite_concepts=["NS-03"],  # Addition
            grade_level="8-9",
            topic_area="Exponent Rules",
            common_errors=[
                ErrorMapping(
                    wrong_answer="x¹⁵",
                    concept_id="EXP-41",
                    concept_name="Exponent Product Rule",
                    error_description="Multiplied exponents instead of adding: 3 × 5 = 15",
                    remediation_hint="When multiplying same base, ADD the exponents: x³ · x⁵ = x³⁺⁵ = x⁸"
                ),
                ErrorMapping(
                    wrong_answer="x³⁵",
                    concept_id="EXP-41",
                    concept_name="Exponent Product Rule",
                    error_description="Wrote exponents side by side instead of adding",
                    remediation_hint="Combine the exponents by addition: 3 + 5 = 8, so x³ · x⁵ = x⁸"
                ),
                ErrorMapping(
                    wrong_answer="2x⁸",
                    concept_id="EXP-41",
                    concept_name="Exponent Product Rule",
                    error_description="Added coefficient of 2, but there's no coefficient to add",
                    remediation_hint="x³ · x⁵ has no numerical coefficients to multiply, just add exponents"
                )
            ]
        ))
        
        # === FUNCTION CONCEPTS ===
        
        questions.append(TestQuestion(
            id="FN_001",
            question_text="If f(x) = 2x + 3, what is f(4)?",
            question_type=QuestionType.NUMERIC,
            difficulty=DifficultyLevel.INTERMEDIATE,
            correct_answer="11",
            explanation="Substitute x = 4 into f(x) = 2x + 3: f(4) = 2(4) + 3 = 8 + 3 = 11",
            primary_concepts=["FN-62"],  # Function Notation
            prerequisite_concepts=["AT-21", "NS-15"],  # Evaluate Expression, Order of Operations
            grade_level="9-10",
            topic_area="Functions",
            common_errors=[
                ErrorMapping(
                    wrong_answer="7",
                    concept_id="FN-62",
                    concept_name="Function Notation",
                    error_description="Added 4 + 3 = 7, forgot to multiply by 2",
                    remediation_hint="Substitute x = 4: f(4) = 2(4) + 3. Remember to multiply 2 × 4 first"
                ),
                ErrorMapping(
                    wrong_answer="14",
                    concept_id="NS-15",
                    concept_name="Order of Operations (PEMDAS)",
                    error_description="Calculated 2(4 + 3) = 2(7) = 14 instead of 2(4) + 3",
                    remediation_hint="Follow order of operations: 2(4) + 3 = 8 + 3 = 11"
                ),
                ErrorMapping(
                    wrong_answer="9",
                    concept_id="AT-21",
                    concept_name="Evaluate Expression",
                    error_description="Calculation error: 2(4) = 8, not 6",
                    remediation_hint="Check multiplication: 2 × 4 = 8, then 8 + 3 = 11"
                )
            ]
        ))
        
        return questions
    
    def _index_by_concept(self) -> Dict[str, List[TestQuestion]]:
        """Index questions by the concepts they test."""
        index = {}
        for question in self.questions:
            for concept in question.get_concept_dependencies():
                if concept not in index:
                    index[concept] = []
                index[concept].append(question)
        return index
    
    def _index_by_difficulty(self) -> Dict[DifficultyLevel, List[TestQuestion]]:
        """Index questions by difficulty level."""
        index = {}
        for question in self.questions:
            if question.difficulty not in index:
                index[question.difficulty] = []
            index[question.difficulty].append(question)
        return index
    
    def get_questions_for_concept(self, concept_id: str) -> List[TestQuestion]:
        """Get all questions that test a specific concept."""
        return self.questions_by_concept.get(concept_id, [])
    
    def get_questions_by_difficulty(self, difficulty: DifficultyLevel) -> List[TestQuestion]:
        """Get all questions at a specific difficulty level."""
        return self.questions_by_difficulty.get(difficulty, [])
    
    def get_all_questions(self) -> List[TestQuestion]:
        """Get all questions in the test bank."""
        return self.questions.copy()
    
    def get_question_by_id(self, question_id: str) -> TestQuestion:
        """Get a specific question by ID."""
        for question in self.questions:
            if question.id == question_id:
                return question
        raise ValueError(f"Question with ID {question_id} not found") 