"""
Advanced Diagnostic Test with Complex Error Patterns
Simulates real standardized test questions and sophisticated error analysis.
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class DifficultyLevel(Enum):
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4

class ErrorType(Enum):
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    COMPUTATIONAL = "computational"
    TRANSLATION = "translation"

@dataclass
class ErrorMapping:
    wrong_answer: str
    concept_id: str
    concept_name: str
    error_type: ErrorType
    error_description: str
    remediation_hint: str
    frequency: float  # How common this error is (0.0-1.0)

@dataclass
class TestQuestion:
    id: str
    question_text: str
    correct_answer: str
    difficulty: DifficultyLevel
    primary_concepts: List[str]
    common_errors: List[ErrorMapping]
    explanation: str = ""
    source: str = ""  # e.g., "SAT", "Common Core", "State Test"

@dataclass
class StudentResponse:
    question_id: str
    student_answer: str
    is_correct: bool
    error_mapping: Optional[ErrorMapping]
    time_taken_seconds: int
    confidence_level: int
    work_shown: str = ""

@dataclass
class ConceptGap:
    concept_id: str
    concept_name: str
    error_type: ErrorType
    evidence_count: int
    severity: float
    related_errors: List[ErrorMapping]
    prerequisite_gaps: List[str]

class AdvancedDiagnosticTest:
    """Advanced diagnostic test with sophisticated error analysis."""
    
    def __init__(self):
        self.questions = self._create_advanced_questions()
        self.concept_dependencies = self._define_concept_dependencies()
    
    def _define_concept_dependencies(self) -> Dict[str, List[str]]:
        """Define prerequisite relationships between concepts."""
        return {
            "QUADRATIC_FORMULA": ["TWO_STEP_EQUATIONS", "SQUARE_ROOTS", "COMBINE_LIKE_TERMS"],
            "FACTORING_QUADRATICS": ["DISTRIBUTIVE_PROPERTY", "COMBINE_LIKE_TERMS", "MULTIPLICATION_FACTS"],
            "SYSTEMS_OF_EQUATIONS": ["TWO_STEP_EQUATIONS", "SUBSTITUTION", "GRAPHING_LINEAR"],
            "FUNCTION_COMPOSITION": ["FUNCTION_NOTATION", "EVALUATE_EXPRESSIONS", "ORDER_OF_OPERATIONS"],
            "EXPONENTIAL_GROWTH": ["EXPONENT_RULES", "FUNCTION_NOTATION", "GRAPHING_BASICS"],
            "RATIONAL_EXPRESSIONS": ["FACTORING", "FRACTION_OPERATIONS", "POLYNOMIAL_OPERATIONS"]
        }
    
    def _create_advanced_questions(self) -> List[TestQuestion]:
        """Create advanced algebra questions with sophisticated error patterns."""
        return [
            # QUADRATIC FORMULA - High-stakes test question
            TestQuestion(
                id="ADV_001",
                question_text="Use the quadratic formula to solve: x² - 6x + 8 = 0",
                correct_answer="x = 2, 4",
                difficulty=DifficultyLevel.ADVANCED,
                primary_concepts=["QUADRATIC_FORMULA"],
                source="SAT Practice Test",
                explanation="Using x = (-b ± √(b²-4ac))/2a with a=1, b=-6, c=8: x = (6 ± √(36-32))/2 = (6 ± 2)/2 = 4, 2",
                common_errors=[
                    ErrorMapping(
                        wrong_answer="x = -2, -4",
                        concept_id="QUADRATIC_FORMULA",
                        concept_name="Quadratic Formula",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Sign error: used +6 instead of -(-6) in numerator",
                        remediation_hint="Remember: -b means the opposite of b, so -(-6) = +6",
                        frequency=0.35
                    ),
                    ErrorMapping(
                        wrong_answer="x = 3 ± 1",
                        concept_id="QUADRATIC_FORMULA",
                        concept_name="Quadratic Formula",
                        error_type=ErrorType.COMPUTATIONAL,
                        error_description="Arithmetic error: 6/2 = 3, but forgot to apply ± to both terms",
                        remediation_hint="Apply the ± to the entire expression: (6 ± 2)/2 = 4, 2",
                        frequency=0.25
                    ),
                    ErrorMapping(
                        wrong_answer="x = 6 ± 2",
                        concept_id="QUADRATIC_FORMULA",
                        concept_name="Quadratic Formula",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Forgot to divide by 2a in final step",
                        remediation_hint="Complete the formula: divide the entire numerator by 2a",
                        frequency=0.20
                    ),
                    ErrorMapping(
                        wrong_answer="No real solutions",
                        concept_id="DISCRIMINANT",
                        concept_name="Discriminant Analysis",
                        error_type=ErrorType.CONCEPTUAL,
                        error_description="Calculated discriminant incorrectly or misinterpreted result",
                        remediation_hint="b² - 4ac = 36 - 32 = 4 > 0, so there are two real solutions",
                        frequency=0.15
                    )
                ]
            ),
            
            # SYSTEMS OF EQUATIONS - Real-world application
            TestQuestion(
                id="ADV_002",
                question_text="A theater sells adult tickets for $12 and child tickets for $8. If they sold 150 tickets for a total of $1,520, how many adult tickets were sold?",
                correct_answer="80",
                difficulty=DifficultyLevel.ADVANCED,
                primary_concepts=["SYSTEMS_OF_EQUATIONS"],
                source="Common Core Assessment",
                explanation="Let a = adult tickets, c = child tickets. System: a + c = 150, 12a + 8c = 1520. Solving: a = 80",
                common_errors=[
                    ErrorMapping(
                        wrong_answer="70",
                        concept_id="SYSTEMS_OF_EQUATIONS",
                        concept_name="Systems of Linear Equations",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Set up equations correctly but made substitution error",
                        remediation_hint="Check substitution: if a = 70, then c = 80, and 12(70) + 8(80) = 1480 ≠ 1520",
                        frequency=0.30
                    ),
                    ErrorMapping(
                        wrong_answer="95",
                        concept_id="SYSTEMS_OF_EQUATIONS",
                        concept_name="Systems of Linear Equations",
                        error_type=ErrorType.TRANSLATION,
                        error_description="Confused which variable represents which quantity",
                        remediation_hint="Clearly define variables: let a = adult tickets (higher price)",
                        frequency=0.25
                    ),
                    ErrorMapping(
                        wrong_answer="120",
                        concept_id="SYSTEMS_OF_EQUATIONS",
                        concept_name="Systems of Linear Equations",
                        error_type=ErrorType.CONCEPTUAL,
                        error_description="Used wrong equation setup: possibly 12a + 8c = 150",
                        remediation_hint="Revenue equation: (price per adult)(# adults) + (price per child)(# children) = total revenue",
                        frequency=0.20
                    )
                ]
            ),
            
            # FUNCTION COMPOSITION - Abstract thinking
            TestQuestion(
                id="ADV_003",
                question_text="If f(x) = 2x + 1 and g(x) = x² - 3, find (f ∘ g)(2)",
                correct_answer="3",
                difficulty=DifficultyLevel.EXPERT,
                primary_concepts=["FUNCTION_COMPOSITION"],
                source="AP Algebra 2",
                explanation="(f ∘ g)(2) = f(g(2)). First: g(2) = 2² - 3 = 1. Then: f(1) = 2(1) + 1 = 3",
                common_errors=[
                    ErrorMapping(
                        wrong_answer="7",
                        concept_id="FUNCTION_COMPOSITION",
                        concept_name="Function Composition",
                        error_type=ErrorType.CONCEPTUAL,
                        error_description="Calculated (g ∘ f)(2) instead of (f ∘ g)(2)",
                        remediation_hint="(f ∘ g)(x) means f(g(x)), not g(f(x)). Order matters!",
                        frequency=0.40
                    ),
                    ErrorMapping(
                        wrong_answer="5",
                        concept_id="FUNCTION_COMPOSITION",
                        concept_name="Function Composition",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Correctly found g(2) = 1, but calculated f(2) instead of f(1)",
                        remediation_hint="After finding g(2) = 1, substitute 1 into f(x), not 2",
                        frequency=0.30
                    ),
                    ErrorMapping(
                        wrong_answer="1",
                        concept_id="FUNCTION_EVALUATION",
                        concept_name="Function Evaluation",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Only calculated g(2) and stopped",
                        remediation_hint="Composition requires two steps: first g(2), then f(result)",
                        frequency=0.20
                    )
                ]
            ),
            
            # EXPONENTIAL GROWTH - Real-world modeling
            TestQuestion(
                id="ADV_004",
                question_text="A bacteria population doubles every 3 hours. If there are initially 500 bacteria, how many will there be after 12 hours?",
                correct_answer="8000",
                difficulty=DifficultyLevel.ADVANCED,
                primary_concepts=["EXPONENTIAL_GROWTH"],
                source="State Assessment",
                explanation="Formula: P(t) = P₀ × 2^(t/3). After 12 hours: P(12) = 500 × 2^(12/3) = 500 × 2⁴ = 500 × 16 = 8000",
                common_errors=[
                    ErrorMapping(
                        wrong_answer="2000",
                        concept_id="EXPONENTIAL_GROWTH",
                        concept_name="Exponential Growth Models",
                        error_type=ErrorType.CONCEPTUAL,
                        error_description="Used linear growth: 500 + 4(500) instead of exponential",
                        remediation_hint="Doubling is exponential: multiply by 2 each period, don't add",
                        frequency=0.35
                    ),
                    ErrorMapping(
                        wrong_answer="4000",
                        concept_id="EXPONENT_RULES",
                        concept_name="Exponent Rules",
                        error_type=ErrorType.COMPUTATIONAL,
                        error_description="Calculated 2³ = 8 instead of 2⁴ = 16",
                        remediation_hint="12 hours ÷ 3 hours per doubling = 4 doublings, so 2⁴ = 16",
                        frequency=0.25
                    ),
                    ErrorMapping(
                        wrong_answer="6000",
                        concept_id="EXPONENTIAL_GROWTH",
                        concept_name="Exponential Growth Models",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Miscounted number of doubling periods",
                        remediation_hint="From 0 to 12 hours with doubling every 3 hours = 4 periods",
                        frequency=0.20
                    )
                ]
            ),
            
            # RATIONAL EXPRESSIONS - Complex algebraic manipulation
            TestQuestion(
                id="ADV_005",
                question_text="Simplify: (x² - 4)/(x² + 4x + 4)",
                correct_answer="(x - 2)/(x + 2)",
                difficulty=DifficultyLevel.EXPERT,
                primary_concepts=["RATIONAL_EXPRESSIONS"],
                source="Pre-Calculus",
                explanation="Factor: (x² - 4) = (x-2)(x+2), (x² + 4x + 4) = (x+2)². Result: (x-2)(x+2)/(x+2)² = (x-2)/(x+2)",
                common_errors=[
                    ErrorMapping(
                        wrong_answer="(x - 2)/(x + 2)²",
                        concept_id="RATIONAL_EXPRESSIONS",
                        concept_name="Simplifying Rational Expressions",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Factored correctly but didn't cancel common factors",
                        remediation_hint="Cancel the common factor (x+2) from numerator and denominator",
                        frequency=0.30
                    ),
                    ErrorMapping(
                        wrong_answer="x - 4",
                        concept_id="FACTORING",
                        concept_name="Factoring Polynomials",
                        error_type=ErrorType.CONCEPTUAL,
                        error_description="Attempted to 'cancel' terms instead of factors",
                        remediation_hint="You can only cancel common factors, not terms. Factor first!",
                        frequency=0.25
                    ),
                    ErrorMapping(
                        wrong_answer="1/(x + 2)",
                        concept_id="FACTORING",
                        concept_name="Factoring Polynomials",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Incorrectly factored x² - 4 as just (x + 2)",
                        remediation_hint="x² - 4 is difference of squares: (x-2)(x+2), not just (x+2)",
                        frequency=0.20
                    )
                ]
            )
        ]
    
    def simulate_sophisticated_response(self, question: TestQuestion, 
                                      student_ability: float,
                                      concept_mastery: Dict[str, float]) -> StudentResponse:
        """Simulate student response with sophisticated error modeling."""
        
        # Base probability of success
        base_prob = student_ability
        
        # Adjust for question difficulty
        difficulty_penalty = {
            DifficultyLevel.BASIC: 0.0,
            DifficultyLevel.INTERMEDIATE: -0.1,
            DifficultyLevel.ADVANCED: -0.2,
            DifficultyLevel.EXPERT: -0.3
        }
        
        # Adjust for concept mastery
        concept_bonus = 0
        for concept in question.primary_concepts:
            if concept in concept_mastery:
                concept_bonus += (concept_mastery[concept] - 0.5) * 0.2
        
        success_prob = base_prob + difficulty_penalty[question.difficulty] + concept_bonus
        success_prob = max(0.05, min(0.95, success_prob))
        
        is_correct = random.random() < success_prob
        
        if is_correct:
            return StudentResponse(
                question_id=question.id,
                student_answer=question.correct_answer,
                is_correct=True,
                error_mapping=None,
                time_taken_seconds=random.randint(120, 300),
                confidence_level=random.randint(4, 5)
            )
        else:
            # Select error based on frequency weights
            if question.common_errors:
                weights = [error.frequency for error in question.common_errors]
                error = random.choices(question.common_errors, weights=weights)[0]
                
                return StudentResponse(
                    question_id=question.id,
                    student_answer=error.wrong_answer,
                    is_correct=False,
                    error_mapping=error,
                    time_taken_seconds=random.randint(180, 450),
                    confidence_level=random.randint(1, 3)
                )
            else:
                return StudentResponse(
                    question_id=question.id,
                    student_answer="unknown_error",
                    is_correct=False,
                    error_mapping=None,
                    time_taken_seconds=random.randint(300, 600),
                    confidence_level=1
                )
    
    def analyze_concept_gaps(self, responses: List[StudentResponse]) -> List[ConceptGap]:
        """Advanced analysis of concept gaps with prerequisite tracking."""
        
        # Track errors by concept and type
        concept_errors = {}
        concept_attempts = {}
        
        for response in responses:
            if response.error_mapping:
                concept_id = response.error_mapping.concept_id
                error_type = response.error_mapping.error_type
                
                key = (concept_id, error_type)
                if key not in concept_errors:
                    concept_errors[key] = []
                    concept_attempts[key] = 0
                
                concept_errors[key].append(response.error_mapping)
                concept_attempts[key] += 1
        
        # Create concept gaps with prerequisite analysis
        gaps = []
        for (concept_id, error_type), errors in concept_errors.items():
            attempts = concept_attempts[(concept_id, error_type)]
            severity = len(errors) / max(attempts, 1)
            
            # Check for prerequisite gaps
            prerequisite_gaps = []
            if concept_id in self.concept_dependencies:
                for prereq in self.concept_dependencies[concept_id]:
                    # Check if this prerequisite also has errors
                    prereq_has_errors = any(
                        error.concept_id == prereq 
                        for error_list in concept_errors.values() 
                        for error in error_list
                    )
                    if prereq_has_errors:
                        prerequisite_gaps.append(prereq)
            
            gap = ConceptGap(
                concept_id=concept_id,
                concept_name=errors[0].concept_name,
                error_type=error_type,
                evidence_count=len(errors),
                severity=severity,
                related_errors=errors,
                prerequisite_gaps=prerequisite_gaps
            )
            gaps.append(gap)
        
        # Sort by severity and error type priority
        type_priority = {
            ErrorType.CONCEPTUAL: 4,
            ErrorType.PROCEDURAL: 3,
            ErrorType.TRANSLATION: 2,
            ErrorType.COMPUTATIONAL: 1
        }
        
        gaps.sort(key=lambda g: (g.severity, type_priority[g.error_type]), reverse=True)
        return gaps
    
    def run_advanced_diagnostic(self, student_name: str, ability_level: float):
        """Run advanced diagnostic with sophisticated analysis."""
        
        print(f"\n{'='*80}")
        print(f"ADVANCED DIAGNOSTIC ASSESSMENT: {student_name}")
        print(f"Ability Level: {ability_level:.2f} (0.0 = Struggling, 1.0 = Expert)")
        print(f"{'='*80}")
        
        # Simulate varying concept mastery
        concept_mastery = {
            "QUADRATIC_FORMULA": ability_level + random.uniform(-0.2, 0.2),
            "SYSTEMS_OF_EQUATIONS": ability_level + random.uniform(-0.2, 0.2),
            "FUNCTION_COMPOSITION": ability_level + random.uniform(-0.3, 0.1),  # Harder concept
            "EXPONENTIAL_GROWTH": ability_level + random.uniform(-0.1, 0.2),
            "RATIONAL_EXPRESSIONS": ability_level + random.uniform(-0.3, 0.1)   # Harder concept
        }
        
        responses = []
        
        print(f"\n📝 ASSESSMENT IN PROGRESS...")
        for question in self.questions:
            response = self.simulate_sophisticated_response(question, ability_level, concept_mastery)
            responses.append(response)
            
            status = "✅" if response.is_correct else "❌"
            difficulty_icon = {
                DifficultyLevel.BASIC: "🟢",
                DifficultyLevel.INTERMEDIATE: "🟡", 
                DifficultyLevel.ADVANCED: "🟠",
                DifficultyLevel.EXPERT: "🔴"
            }[question.difficulty]
            
            print(f"   {status} {difficulty_icon} {question.id}: {question.question_text[:60]}...")
            print(f"      Source: {question.source}")
            print(f"      Student: '{response.student_answer}' | Correct: '{question.correct_answer}'")
            
            if response.error_mapping:
                print(f"      🚨 {response.error_mapping.error_type.value.title()} Error: {response.error_mapping.error_description}")
                print(f"      💡 Remediation: {response.error_mapping.remediation_hint}")
        
        # Performance analysis
        correct_count = sum(1 for r in responses if r.is_correct)
        accuracy = correct_count / len(responses) * 100
        avg_time = sum(r.time_taken_seconds for r in responses) / len(responses)
        avg_confidence = sum(r.confidence_level for r in responses) / len(responses)
        
        print(f"\n📊 PERFORMANCE ANALYSIS:")
        print(f"   Questions Attempted: {len(responses)}")
        print(f"   Correct Answers: {correct_count}")
        print(f"   Overall Accuracy: {accuracy:.1f}%")
        print(f"   Average Time per Question: {avg_time:.0f} seconds")
        print(f"   Average Confidence: {avg_confidence:.1f}/5")
        
        # Advanced gap analysis
        gaps = self.analyze_concept_gaps(responses)
        
        if gaps:
            print(f"\n🚨 DETAILED CONCEPT GAP ANALYSIS ({len(gaps)} gaps identified):")
            
            for i, gap in enumerate(gaps, 1):
                priority_icon = "🔥" if gap.severity > 0.7 else "⚠️" if gap.severity > 0.4 else "📝"
                error_type_icon = {
                    ErrorType.CONCEPTUAL: "🧠",
                    ErrorType.PROCEDURAL: "⚙️", 
                    ErrorType.TRANSLATION: "🔄",
                    ErrorType.COMPUTATIONAL: "🔢"
                }[gap.error_type]
                
                print(f"\n   {priority_icon} {i}. {gap.concept_name}")
                print(f"      {error_type_icon} Error Type: {gap.error_type.value.title()}")
                print(f"      📈 Severity: {gap.severity:.2f} | Evidence: {gap.evidence_count} errors")
                
                if gap.prerequisite_gaps:
                    print(f"      🔗 Prerequisite Issues: {', '.join(gap.prerequisite_gaps)}")
                
                print(f"      📋 Specific Errors:")
                for error in gap.related_errors:
                    print(f"         • {error.error_description}")
                    print(f"           💡 {error.remediation_hint}")
        else:
            print(f"\n✅ EXCELLENT PERFORMANCE - NO SIGNIFICANT GAPS DETECTED!")
        
        # Advanced recommendations
        print(f"\n🎯 PERSONALIZED LEARNING RECOMMENDATIONS:")
        
        if gaps:
            # Prioritize by error type and prerequisites
            conceptual_gaps = [g for g in gaps if g.error_type == ErrorType.CONCEPTUAL]
            procedural_gaps = [g for g in gaps if g.error_type == ErrorType.PROCEDURAL]
            
            if conceptual_gaps:
                print(f"   🧠 PRIORITY: Address conceptual understanding first")
                for gap in conceptual_gaps[:2]:
                    print(f"      • {gap.concept_name}: Focus on underlying principles")
            
            if procedural_gaps:
                print(f"   ⚙️ PRACTICE: Strengthen procedural skills")
                for gap in procedural_gaps[:2]:
                    print(f"      • {gap.concept_name}: Practice step-by-step procedures")
            
            # Estimate study time based on gap severity
            total_study_time = sum(gap.severity * 45 for gap in gaps)  # 45 min per severe gap
            print(f"   ⏱️ Estimated Study Time: {total_study_time:.0f} minutes")
            
            # Suggest learning sequence
            if any(gap.prerequisite_gaps for gap in gaps):
                print(f"   📚 Suggested Learning Sequence:")
                print(f"      1. Review prerequisite concepts first")
                print(f"      2. Practice with guided examples")
                print(f"      3. Apply to similar problems")
                print(f"      4. Take follow-up assessment")
        else:
            print(f"   🚀 Ready for advanced topics!")
            print(f"   📈 Consider: Calculus preparation, competition math, or real-world applications")

def main():
    """Run the advanced diagnostic demonstration."""
    
    print("🧮 ADVANCED ALGEBRA DIAGNOSTIC SYSTEM")
    print("Sophisticated Error Analysis and Concept Gap Identification")
    
    test = AdvancedDiagnosticTest()
    
    # Test different ability levels
    students = [
        ("Sarah (High Achiever)", 0.85),
        ("Marcus (Average Student)", 0.65),
        ("Aisha (Struggling Student)", 0.35)
    ]
    
    for name, ability in students:
        # Set different seeds for reproducible but varied results
        random.seed(hash(name) % 1000)
        test.run_advanced_diagnostic(name, ability)
    
    print(f"\n{'='*80}")
    print("🎉 ADVANCED DIAGNOSTIC DEMONSTRATION COMPLETE!")
    print("="*80)
    
    print(f"\nAdvanced Features Demonstrated:")
    print("✅ Sophisticated error modeling with frequency weights")
    print("✅ Multi-dimensional gap analysis (conceptual, procedural, etc.)")
    print("✅ Prerequisite dependency tracking")
    print("✅ Real standardized test question formats")
    print("✅ Personalized learning recommendations")
    print("✅ Performance analytics and confidence tracking")
    
    print(f"\nError Types Identified:")
    print("🧠 Conceptual: Fundamental misunderstandings")
    print("⚙️ Procedural: Incorrect step sequences")
    print("🔄 Translation: Word problem interpretation")
    print("🔢 Computational: Arithmetic mistakes")
    
    print(f"\nReal-World Applications:")
    print("• Adaptive learning platforms")
    print("• Standardized test preparation")
    print("• Teacher professional development")
    print("• Curriculum effectiveness analysis")
    print("• Student placement and intervention")

if __name__ == "__main__":
    main() 