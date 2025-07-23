"""
Remediation Question Bank
Targeted practice questions designed to fill specific concept gaps
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import random

@dataclass
class RemediationQuestion:
    """A practice question designed to address a specific concept gap"""
    question_id: str
    concept: str
    difficulty_level: int  # 1 = basic, 2 = intermediate, 3 = advanced
    question_text: str
    correct_answer: str
    explanation: str
    hints: List[str]
    prerequisite_concepts: List[str]
    question_type: str  # "multiple_choice", "fill_in", "step_by_step"
    options: Optional[List[str]] = None  # For multiple choice

class RemediationQuestionBank:
    """Bank of practice questions organized by concept gaps"""
    
    def __init__(self):
        self.questions = self._initialize_questions()
    
    def _initialize_questions(self) -> Dict[str, List[RemediationQuestion]]:
        """Initialize comprehensive remediation questions for each concept"""
        return {
            # One-Step Linear Equations
            "ONE_STEP_EQUATIONS": [
                RemediationQuestion(
                    question_id="OSE_001",
                    concept="ONE_STEP_EQUATIONS",
                    difficulty_level=1,
                    question_text="Solve for x: x + 5 = 12",
                    correct_answer="x = 7",
                    explanation="To isolate x, subtract 5 from both sides: x + 5 - 5 = 12 - 5, so x = 7",
                    hints=[
                        "What operation undoes addition?",
                        "Subtract 5 from both sides",
                        "Check: 7 + 5 = 12 ✓"
                    ],
                    prerequisite_concepts=["ADDITION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="OSE_002", 
                    concept="ONE_STEP_EQUATIONS",
                    difficulty_level=1,
                    question_text="Solve for x: 3x = 15",
                    correct_answer="x = 5",
                    explanation="To isolate x, divide both sides by 3: 3x ÷ 3 = 15 ÷ 3, so x = 5",
                    hints=[
                        "What operation undoes multiplication?",
                        "Divide both sides by 3",
                        "Check: 3 × 5 = 15 ✓"
                    ],
                    prerequisite_concepts=["MULTIPLICATION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="OSE_003",
                    concept="ONE_STEP_EQUATIONS", 
                    difficulty_level=2,
                    question_text="Solve for x: x - 8 = -3",
                    correct_answer="x = 5",
                    explanation="To isolate x, add 8 to both sides: x - 8 + 8 = -3 + 8, so x = 5",
                    hints=[
                        "What operation undoes subtraction?",
                        "Add 8 to both sides",
                        "Be careful with negative numbers"
                    ],
                    prerequisite_concepts=["INTEGER_OPERATIONS"],
                    question_type="fill_in"
                )
            ],
            
            # Order of Operations
            "ORDER_OF_OPERATIONS": [
                RemediationQuestion(
                    question_id="OOO_001",
                    concept="ORDER_OF_OPERATIONS",
                    difficulty_level=1,
                    question_text="Calculate: 2 + 3 × 4",
                    correct_answer="14",
                    explanation="PEMDAS: Multiplication before addition. 3 × 4 = 12, then 2 + 12 = 14",
                    hints=[
                        "Remember PEMDAS",
                        "Do multiplication first",
                        "Then do addition"
                    ],
                    prerequisite_concepts=["MULTIPLICATION_FACTS", "ADDITION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="OOO_002",
                    concept="ORDER_OF_OPERATIONS",
                    difficulty_level=2,
                    question_text="Which operation should you do first in: 5 + 2 × 3?",
                    correct_answer="Multiplication (2 × 3)",
                    explanation="According to PEMDAS, multiplication comes before addition",
                    hints=[
                        "Think about PEMDAS",
                        "P-E-M-D-A-S",
                        "Multiplication comes before Addition"
                    ],
                    prerequisite_concepts=["MULTIPLICATION_FACTS"],
                    question_type="multiple_choice",
                    options=["Addition (5 + 2)", "Multiplication (2 × 3)", "Do them left to right", "It doesn't matter"]
                ),
                RemediationQuestion(
                    question_id="OOO_003",
                    concept="ORDER_OF_OPERATIONS",
                    difficulty_level=2,
                    question_text="Calculate: (4 + 2) × 3",
                    correct_answer="18",
                    explanation="Parentheses first: (4 + 2) = 6, then 6 × 3 = 18",
                    hints=[
                        "Parentheses come first in PEMDAS",
                        "Calculate 4 + 2 first",
                        "Then multiply by 3"
                    ],
                    prerequisite_concepts=["ADDITION_FACTS", "MULTIPLICATION_FACTS"],
                    question_type="fill_in"
                )
            ],
            
            # Distributive Property
            "DISTRIBUTIVE_PROPERTY": [
                RemediationQuestion(
                    question_id="DP_001",
                    concept="DISTRIBUTIVE_PROPERTY",
                    difficulty_level=1,
                    question_text="Expand: 2(x + 3)",
                    correct_answer="2x + 6",
                    explanation="Distribute the 2: 2 × x + 2 × 3 = 2x + 6",
                    hints=[
                        "Multiply 2 by each term inside the parentheses",
                        "2 × x = 2x",
                        "2 × 3 = 6"
                    ],
                    prerequisite_concepts=["MULTIPLICATION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="DP_002",
                    concept="DISTRIBUTIVE_PROPERTY",
                    difficulty_level=1,
                    question_text="What is 3 × (4 + 5) equal to?",
                    correct_answer="3 × 4 + 3 × 5",
                    explanation="The distributive property says a(b + c) = ab + ac",
                    hints=[
                        "Think about the distributive property",
                        "Multiply 3 by each term inside",
                        "3 × 4 and 3 × 5"
                    ],
                    prerequisite_concepts=["MULTIPLICATION_FACTS"],
                    question_type="multiple_choice",
                    options=["3 × 4 + 3 × 5", "3 × 9", "(3 × 4) × (3 × 5)", "3 + 4 × 5"]
                ),
                RemediationQuestion(
                    question_id="DP_003",
                    concept="DISTRIBUTIVE_PROPERTY",
                    difficulty_level=2,
                    question_text="Expand: -2(x - 4)",
                    correct_answer="-2x + 8",
                    explanation="Distribute -2: (-2) × x + (-2) × (-4) = -2x + 8",
                    hints=[
                        "Distribute the -2 to both terms",
                        "(-2) × x = -2x",
                        "(-2) × (-4) = +8"
                    ],
                    prerequisite_concepts=["INTEGER_OPERATIONS", "MULTIPLICATION_FACTS"],
                    question_type="fill_in"
                )
            ],
            
            # Combine Like Terms
            "COMBINE_LIKE_TERMS": [
                RemediationQuestion(
                    question_id="CLT_001",
                    concept="COMBINE_LIKE_TERMS",
                    difficulty_level=1,
                    question_text="Simplify: 3x + 2x",
                    correct_answer="5x",
                    explanation="Like terms have the same variable: 3x + 2x = (3 + 2)x = 5x",
                    hints=[
                        "These are like terms (both have x)",
                        "Add the coefficients: 3 + 2",
                        "Keep the variable: x"
                    ],
                    prerequisite_concepts=["ADDITION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="CLT_002",
                    concept="COMBINE_LIKE_TERMS",
                    difficulty_level=1,
                    question_text="Simplify: 5 + 3 + 2x",
                    correct_answer="8 + 2x",
                    explanation="Combine the constant terms: 5 + 3 = 8, keep 2x separate",
                    hints=[
                        "Identify like terms",
                        "5 and 3 are constants (like terms)",
                        "2x is different (has a variable)"
                    ],
                    prerequisite_concepts=["ADDITION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="CLT_003",
                    concept="COMBINE_LIKE_TERMS",
                    difficulty_level=2,
                    question_text="Simplify: 4x + 3 - 2x + 1",
                    correct_answer="2x + 4",
                    explanation="Combine like terms: (4x - 2x) + (3 + 1) = 2x + 4",
                    hints=[
                        "Group like terms together",
                        "x terms: 4x - 2x = 2x",
                        "Constants: 3 + 1 = 4"
                    ],
                    prerequisite_concepts=["ADDITION_FACTS", "INTEGER_OPERATIONS"],
                    question_type="fill_in"
                )
            ],
            
            # Integer Operations
            "INTEGER_OPERATIONS": [
                RemediationQuestion(
                    question_id="IO_001",
                    concept="INTEGER_OPERATIONS",
                    difficulty_level=1,
                    question_text="Calculate: -3 + (-2)",
                    correct_answer="-5",
                    explanation="Adding two negatives: combine absolute values and keep negative sign: 3 + 2 = 5, so -3 + (-2) = -5",
                    hints=[
                        "Both numbers are negative",
                        "Add the absolute values: 3 + 2 = 5",
                        "Keep the negative sign: -5"
                    ],
                    prerequisite_concepts=["ADDITION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="IO_002",
                    concept="INTEGER_OPERATIONS",
                    difficulty_level=1,
                    question_text="Calculate: -4 + 6",
                    correct_answer="2",
                    explanation="Different signs: subtract absolute values and use sign of larger: 6 - 4 = 2 (positive)",
                    hints=[
                        "Different signs: one positive, one negative",
                        "Subtract absolute values: 6 - 4 = 2",
                        "Use sign of larger number (6 is positive)"
                    ],
                    prerequisite_concepts=["ADDITION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="IO_003",
                    concept="INTEGER_OPERATIONS",
                    difficulty_level=2,
                    question_text="Calculate: -2 - (-5)",
                    correct_answer="3",
                    explanation="Subtracting a negative is like adding: -2 - (-5) = -2 + 5 = 3",
                    hints=[
                        "Subtracting a negative = adding a positive",
                        "-(-5) becomes +5",
                        "-2 + 5 = 3"
                    ],
                    prerequisite_concepts=["ADDITION_FACTS"],
                    question_type="fill_in"
                )
            ],
            
            # Multiplication Facts
            "MULTIPLICATION_FACTS": [
                RemediationQuestion(
                    question_id="MF_001",
                    concept="MULTIPLICATION_FACTS",
                    difficulty_level=1,
                    question_text="Calculate: 6 × 7",
                    correct_answer="42",
                    explanation="6 × 7 = 42 (basic multiplication fact)",
                    hints=[
                        "Think of the multiplication table",
                        "6 × 7 is a basic fact",
                        "6 × 7 = 42"
                    ],
                    prerequisite_concepts=["COUNTING_UP_DOWN"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="MF_002",
                    concept="MULTIPLICATION_FACTS",
                    difficulty_level=1,
                    question_text="What is 8 × 9?",
                    correct_answer="72",
                    explanation="8 × 9 = 72 (basic multiplication fact)",
                    hints=[
                        "Use the multiplication table",
                        "8 × 9 = 72",
                        "Practice this fact until automatic"
                    ],
                    prerequisite_concepts=["COUNTING_UP_DOWN"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="MF_003",
                    concept="MULTIPLICATION_FACTS",
                    difficulty_level=2,
                    question_text="Calculate: 12 × 8",
                    correct_answer="96",
                    explanation="12 × 8 = 96. Think: 10 × 8 + 2 × 8 = 80 + 16 = 96",
                    hints=[
                        "Break down 12 = 10 + 2",
                        "10 × 8 = 80",
                        "2 × 8 = 16, so 80 + 16 = 96"
                    ],
                    prerequisite_concepts=["ADDITION_FACTS"],
                    question_type="fill_in"
                )
            ],
            
            # Fractions as Part-Whole
            "FRACTIONS_AS_PART_WHOLE": [
                RemediationQuestion(
                    question_id="FPW_001",
                    concept="FRACTIONS_AS_PART_WHOLE",
                    difficulty_level=1,
                    question_text="What is 1/2 of 8?",
                    correct_answer="4",
                    explanation="1/2 of 8 means 8 ÷ 2 = 4",
                    hints=[
                        "'Of' means multiply",
                        "1/2 × 8 = 8/2 = 4",
                        "Half of 8 is 4"
                    ],
                    prerequisite_concepts=["MULTIPLICATION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="FPW_002",
                    concept="FRACTIONS_AS_PART_WHOLE",
                    difficulty_level=2,
                    question_text="Calculate: 2/3 × 12",
                    correct_answer="8",
                    explanation="2/3 × 12 = (2 × 12) ÷ 3 = 24 ÷ 3 = 8",
                    hints=[
                        "Multiply by numerator first: 2 × 12 = 24",
                        "Then divide by denominator: 24 ÷ 3",
                        "24 ÷ 3 = 8"
                    ],
                    prerequisite_concepts=["MULTIPLICATION_FACTS"],
                    question_type="fill_in"
                ),
                RemediationQuestion(
                    question_id="FPW_003",
                    concept="FRACTIONS_AS_PART_WHOLE",
                    difficulty_level=2,
                    question_text="If you eat 3/4 of a pizza with 8 slices, how many slices did you eat?",
                    correct_answer="6",
                    explanation="3/4 of 8 = (3 × 8) ÷ 4 = 24 ÷ 4 = 6 slices",
                    hints=[
                        "Find 3/4 of 8",
                        "3 × 8 = 24",
                        "24 ÷ 4 = 6 slices"
                    ],
                    prerequisite_concepts=["MULTIPLICATION_FACTS"],
                    question_type="fill_in"
                )
            ]
        }
    
    def get_questions_for_concept(self, concept: str, difficulty_level: Optional[int] = None) -> List[RemediationQuestion]:
        """Get all questions for a specific concept, optionally filtered by difficulty"""
        if concept not in self.questions:
            return []
        
        questions = self.questions[concept]
        
        if difficulty_level is not None:
            questions = [q for q in questions if q.difficulty_level == difficulty_level]
        
        return questions
    
    def get_best_question_for_gap(self, missing_concept: str, student_ability: float = 0.5) -> Optional[RemediationQuestion]:
        """
        Get the best remediation question for a specific concept gap
        
        Args:
            missing_concept: The concept the student is missing
            student_ability: Student's overall ability level (0.0 = struggling, 1.0 = expert)
            
        Returns:
            Best-fit remediation question for this student
        """
        questions = self.get_questions_for_concept(missing_concept)
        
        if not questions:
            return None
        
        # Choose difficulty based on student ability
        if student_ability < 0.3:
            target_difficulty = 1  # Basic questions for struggling students
        elif student_ability < 0.7:
            target_difficulty = 2  # Intermediate questions for average students
        else:
            target_difficulty = 3  # Advanced questions for strong students
        
        # Find questions at target difficulty
        target_questions = [q for q in questions if q.difficulty_level == target_difficulty]
        
        # If no questions at target difficulty, use closest available
        if not target_questions:
            target_questions = questions
        
        # Return a random question from the appropriate difficulty level
        return random.choice(target_questions)
    
    def get_progressive_questions(self, concept: str, count: int = 3) -> List[RemediationQuestion]:
        """Get a progressive sequence of questions for a concept (easy to hard)"""
        questions = self.get_questions_for_concept(concept)
        
        if not questions:
            return []
        
        # Sort by difficulty
        questions.sort(key=lambda q: q.difficulty_level)
        
        # Return up to 'count' questions, spread across difficulties
        if len(questions) <= count:
            return questions
        
        # Select questions to cover different difficulty levels
        selected = []
        difficulties = sorted(set(q.difficulty_level for q in questions))
        
        for difficulty in difficulties:
            difficulty_questions = [q for q in questions if q.difficulty_level == difficulty]
            selected.append(random.choice(difficulty_questions))
            
            if len(selected) >= count:
                break
        
        return selected[:count]

def demonstrate_remediation_questions():
    """Demonstrate the remediation question system"""
    bank = RemediationQuestionBank()
    
    print("📚 REMEDIATION QUESTION BANK DEMONSTRATION")
    print("=" * 60)
    
    # Show questions for different concepts
    concepts = ["ONE_STEP_EQUATIONS", "ORDER_OF_OPERATIONS", "DISTRIBUTIVE_PROPERTY"]
    
    for concept in concepts:
        print(f"\n🎯 CONCEPT: {concept}")
        questions = bank.get_questions_for_concept(concept)
        
        for i, question in enumerate(questions[:2], 1):  # Show first 2 questions
            print(f"\n   📝 Question {i} (Difficulty {question.difficulty_level}):")
            print(f"      {question.question_text}")
            print(f"      ✅ Answer: {question.correct_answer}")
            print(f"      💡 Explanation: {question.explanation}")
            print(f"      🔍 Hints: {', '.join(question.hints[:2])}")
    
    print(f"\n🎯 PERSONALIZED RECOMMENDATIONS:")
    
    # Show personalized recommendations for different student levels
    for ability, level_name in [(0.2, "Struggling"), (0.5, "Average"), (0.8, "Strong")]:
        print(f"\n   👤 {level_name} Student (ability: {ability}):")
        
        for concept in ["ONE_STEP_EQUATIONS", "ORDER_OF_OPERATIONS"]:
            best_question = bank.get_best_question_for_gap(concept, ability)
            if best_question:
                print(f"      📚 {concept}: {best_question.question_text}")
                print(f"         Difficulty: {best_question.difficulty_level}")

if __name__ == "__main__":
    demonstrate_remediation_questions() 