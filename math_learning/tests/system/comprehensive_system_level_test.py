#!/usr/bin/env python3
"""
COMPREHENSIVE K-12 ALGEBRA DIAGNOSTIC SYSTEM LEVEL TEST
Complete testing suite covering all concepts, error patterns, student profiles,
and system performance validation with detailed reporting and analysis.
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_diagnostic_system import EnhancedDiagnosticSystem, EnhancedDiagnosticResult
from comprehensive_diagnostic import StudentProfile
from failure_simulation import FailureSimulator
from algebra_comprehensive_question_bank_complete import CompleteAlgebraQuestionBank

@dataclass
class TestCase:
    """Comprehensive test case definition"""
    test_id: str
    category: str
    concept: str
    question: str
    correct_answer: str
    student_answer: str
    student_profile_type: str
    expected_missing_concept: str
    expected_confidence_range: Tuple[float, float]
    expected_remediation_level: str
    description: str
    difficulty_level: str = "medium"
    question_type: str = "general_algebra"

@dataclass
class TestResult:
    """Test result with validation"""
    test_case: TestCase
    diagnostic_result: EnhancedDiagnosticResult
    concept_match: bool
    confidence_in_range: bool
    remediation_appropriate: bool
    processing_time: float
    overall_success: bool

class ComprehensiveSystemTest:
    """Comprehensive system-level testing framework"""
    
    def __init__(self):
        self.diagnostic_system = EnhancedDiagnosticSystem()
        self.failure_simulator = FailureSimulator()
        self.question_bank = CompleteAlgebraQuestionBank()
        
        # Create diverse student profiles
        self.student_profiles = self._create_student_profiles()
        
        # Initialize test cases
        self.test_cases = self._initialize_comprehensive_test_cases()
        
        # Results tracking
        self.test_results: List[TestResult] = []
        self.performance_metrics = defaultdict(list)
    
    def _create_student_profiles(self) -> Dict[str, StudentProfile]:
        """Create diverse student profiles for testing"""
        return {
            "struggling_elementary": StudentProfile(
                name="Emma (Grade 3, Struggling)",
                overall_ability=0.15,
                concept_mastery={
                    "ORDER_OF_OPERATIONS": 0.05,
                    "DISTRIBUTIVE_PROPERTY": 0.10,
                    "ONE_STEP_EQUATIONS": 0.20,
                    "INTEGER_OPERATIONS": 0.15,
                    "FRACTIONS_AS_PART_WHOLE": 0.25
                },
                learning_style="visual",
                total_questions_attempted=12,
                total_questions_correct=2
            ),
            
            "struggling_middle": StudentProfile(
                name="Alex (Grade 6, Struggling)",
                overall_ability=0.25,
                concept_mastery={
                    "ORDER_OF_OPERATIONS": 0.20,
                    "DISTRIBUTIVE_PROPERTY": 0.15,
                    "ONE_STEP_EQUATIONS": 0.30,
                    "TWO_STEP_EQUATIONS": 0.10,
                    "INTEGER_OPERATIONS": 0.35
                },
                learning_style="hands_on",
                total_questions_attempted=20,
                total_questions_correct=5
            ),
            
            "average_middle": StudentProfile(
                name="Jordan (Grade 7, Average)",
                overall_ability=0.55,
                concept_mastery={
                    "ORDER_OF_OPERATIONS": 0.60,
                    "DISTRIBUTIVE_PROPERTY": 0.45,
                    "ONE_STEP_EQUATIONS": 0.70,
                    "TWO_STEP_EQUATIONS": 0.50,
                    "FUNCTION_EVALUATION": 0.40
                },
                learning_style="analytical",
                total_questions_attempted=30,
                total_questions_correct=17
            ),
            
            "average_high": StudentProfile(
                name="Sam (Grade 9, Average)",
                overall_ability=0.65,
                concept_mastery={
                    "ORDER_OF_OPERATIONS": 0.80,
                    "DISTRIBUTIVE_PROPERTY": 0.60,
                    "TWO_STEP_EQUATIONS": 0.70,
                    "FUNCTION_EVALUATION": 0.55,
                    "QUADRATIC_FORMULA": 0.45
                },
                learning_style="visual",
                total_questions_attempted=40,
                total_questions_correct=26
            ),
            
            "advanced_middle": StudentProfile(
                name="Taylor (Grade 8, Advanced)",
                overall_ability=0.85,
                concept_mastery={
                    "ORDER_OF_OPERATIONS": 0.95,
                    "DISTRIBUTIVE_PROPERTY": 0.90,
                    "TWO_STEP_EQUATIONS": 0.85,
                    "FUNCTION_EVALUATION": 0.80,
                    "FUNCTION_COMPOSITION": 0.75
                },
                learning_style="analytical",
                total_questions_attempted=45,
                total_questions_correct=38
            ),
            
            "advanced_high": StudentProfile(
                name="Morgan (Grade 11, Advanced)",
                overall_ability=0.92,
                concept_mastery={
                    "ORDER_OF_OPERATIONS": 0.98,
                    "DISTRIBUTIVE_PROPERTY": 0.95,
                    "FUNCTION_COMPOSITION": 0.90,
                    "QUADRATIC_FORMULA": 0.85,
                    "POLYNOMIAL_OPERATIONS": 0.88
                },
                learning_style="hands_on",
                total_questions_attempted=50,
                total_questions_correct=46
            )
        }
    
    def _initialize_comprehensive_test_cases(self) -> List[TestCase]:
        """Initialize comprehensive test cases covering all scenarios"""
        
        test_cases = []
        
        # NUMBER SENSE CATEGORY (NS-01 to NS-07)
        test_cases.extend([
            TestCase(
                test_id="NS-01-001",
                category="Number Sense",
                concept="NS-01",
                question="Count from 1 to 10: 1, 2, 3, ?, 5",
                correct_answer="4",
                student_answer="6",
                student_profile_type="struggling_elementary",
                expected_missing_concept="COUNTING_SEQUENCE",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="easy",
                description="Basic counting sequence error",
                question_type="counting"
            ),
            
            TestCase(
                test_id="NS-02-001",
                category="Number Sense",
                concept="NS-02",
                question="What is the value of the digit 5 in 352?",
                correct_answer="50",
                student_answer="5",
                student_profile_type="struggling_elementary",
                expected_missing_concept="PLACE_VALUE",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="easy",
                description="Place value misunderstanding",
                question_type="place_value"
            ),
            
            TestCase(
                test_id="NS-03-001",
                category="Number Sense",
                concept="NS-03",
                question="Which shows the commutative property? A) 3 + 5 = 5 + 3  B) 3 + 0 = 3",
                correct_answer="A",
                student_answer="B",
                student_profile_type="struggling_middle",
                expected_missing_concept="COMMUTATIVE_PROPERTY",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="easy",
                description="Commutative vs identity property confusion",
                question_type="commutative_property"
            )
        ])
        
        # ORDER OF OPERATIONS (Critical concept)
        test_cases.extend([
            TestCase(
                test_id="OOO-001",
                category="Order of Operations",
                concept="ORDER_OF_OPERATIONS",
                question="Calculate: 3 + 4 × 2",
                correct_answer="11",
                student_answer="14",
                student_profile_type="struggling_elementary",
                expected_missing_concept="ORDER_OF_OPERATIONS",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="easy",
                description="Classic left-to-right error",
                question_type="order_of_operations"
            ),
            
            TestCase(
                test_id="OOO-002",
                category="Order of Operations",
                concept="ORDER_OF_OPERATIONS",
                question="Calculate: 2 + 3 × 4",
                correct_answer="14",
                student_answer="20",
                student_profile_type="struggling_middle",
                expected_missing_concept="ORDER_OF_OPERATIONS",
                expected_confidence_range=(0.85, 0.95),
                expected_remediation_level="easy",
                description="Addition before multiplication error",
                question_type="order_of_operations"
            ),
            
            TestCase(
                test_id="OOO-003",
                category="Order of Operations",
                concept="ORDER_OF_OPERATIONS",
                question="Calculate: (2 + 3) × 4 - 1",
                correct_answer="19",
                student_answer="18",
                student_profile_type="average_middle",
                expected_missing_concept="ORDER_OF_OPERATIONS",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="medium",
                description="Parentheses handled correctly, subtraction error",
                question_type="order_of_operations"
            )
        ])
        
        # TWO-STEP EQUATIONS
        test_cases.extend([
            TestCase(
                test_id="TSE-001",
                category="Linear Equations",
                concept="TWO_STEP_EQUATIONS",
                question="Solve: 3x + 7 = 22",
                correct_answer="x = 5",
                student_answer="x = 29",
                student_profile_type="struggling_middle",
                expected_missing_concept="ONE_STEP_EQUATIONS",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="easy",
                description="Addition instead of subtraction error",
                question_type="two_step_equations"
            ),
            
            TestCase(
                test_id="TSE-002",
                category="Linear Equations",
                concept="TWO_STEP_EQUATIONS",
                question="Solve: 2x + 5 = 13",
                correct_answer="x = 4",
                student_answer="x = 8",
                student_profile_type="average_middle",
                expected_missing_concept="ONE_STEP_EQUATIONS",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="medium",
                description="Forgot to divide by coefficient",
                question_type="two_step_equations"
            ),
            
            TestCase(
                test_id="TSE-003",
                category="Linear Equations",
                concept="TWO_STEP_EQUATIONS",
                question="Solve: 4x - 3 = 17",
                correct_answer="x = 5",
                student_answer="x = 3.5",
                student_profile_type="advanced_middle",
                expected_missing_concept="INTEGER_OPERATIONS",
                expected_confidence_range=(0.85, 0.95),
                expected_remediation_level="hard",
                description="Calculation error in final step",
                question_type="two_step_equations"
            )
        ])
        
        # DISTRIBUTIVE PROPERTY
        test_cases.extend([
            TestCase(
                test_id="DIST-001",
                category="Algebraic Manipulation",
                concept="DISTRIBUTIVE_PROPERTY",
                question="Expand: 2(x + 3)",
                correct_answer="2x + 6",
                student_answer="2x + 3",
                student_profile_type="struggling_middle",
                expected_missing_concept="DISTRIBUTIVE_PROPERTY",
                expected_confidence_range=(0.85, 0.95),
                expected_remediation_level="easy",
                description="Forgot to distribute to constant",
                question_type="distributive_property"
            ),
            
            TestCase(
                test_id="DIST-002",
                category="Algebraic Manipulation",
                concept="DISTRIBUTIVE_PROPERTY",
                question="Expand: 3(2x + 4)",
                correct_answer="6x + 12",
                student_answer="6x + 4",
                student_profile_type="average_middle",
                expected_missing_concept="DISTRIBUTIVE_PROPERTY",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="medium",
                description="Distributed to variable but not constant",
                question_type="distributive_property"
            ),
            
            TestCase(
                test_id="DIST-003",
                category="Algebraic Manipulation",
                concept="DISTRIBUTIVE_PROPERTY",
                question="Factor: 4x + 8",
                correct_answer="4(x + 2)",
                student_answer="4x + 8",
                student_profile_type="average_high",
                expected_missing_concept="FACTORING",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="medium",
                description="Didn't recognize factoring opportunity",
                question_type="distributive_property"
            )
        ])
        
        # INTEGER OPERATIONS
        test_cases.extend([
            TestCase(
                test_id="INT-001",
                category="Integer Operations",
                concept="INTEGER_OPERATIONS",
                question="Calculate: -5 + (-3)",
                correct_answer="-8",
                student_answer="2",
                student_profile_type="struggling_middle",
                expected_missing_concept="INTEGER_OPERATIONS",
                expected_confidence_range=(0.8, 0.95),
                expected_remediation_level="easy",
                description="Treated as subtraction of absolute values",
                question_type="integer_operations"
            ),
            
            TestCase(
                test_id="INT-002",
                category="Integer Operations",
                concept="INTEGER_OPERATIONS",
                question="Calculate: -4 + 7",
                correct_answer="3",
                student_answer="-11",
                student_profile_type="struggling_elementary",
                expected_missing_concept="INTEGER_OPERATIONS",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="easy",
                description="Added absolute values with negative result",
                question_type="integer_operations"
            ),
            
            TestCase(
                test_id="INT-003",
                category="Integer Operations",
                concept="INTEGER_OPERATIONS",
                question="Calculate: 8 - (-3)",
                correct_answer="11",
                student_answer="5",
                student_profile_type="average_middle",
                expected_missing_concept="INTEGER_OPERATIONS",
                expected_confidence_range=(0.9, 0.98),
                expected_remediation_level="medium",
                description="Subtracted instead of adding",
                question_type="integer_operations"
            )
        ])
        
        # FUNCTION EVALUATION
        test_cases.extend([
            TestCase(
                test_id="FUNC-001",
                category="Functions",
                concept="FUNCTION_EVALUATION",
                question="If f(x) = 2x + 3, find f(4)",
                correct_answer="11",
                student_answer="14",
                student_profile_type="average_middle",
                expected_missing_concept="FUNCTION_EVALUATION",
                expected_confidence_range=(0.7, 0.85),
                expected_remediation_level="medium",
                description="Calculation error in function evaluation",
                question_type="function_evaluation"
            ),
            
            TestCase(
                test_id="FUNC-002",
                category="Functions",
                concept="FUNCTION_EVALUATION",
                question="If g(x) = x² - 1, find g(3)",
                correct_answer="8",
                student_answer="6",
                student_profile_type="average_high",
                expected_missing_concept="EXPONENT_RULES",
                expected_confidence_range=(0.7, 0.85),
                expected_remediation_level="medium",
                description="Didn't square the input correctly",
                question_type="function_evaluation"
            )
        ])
        
        # FUNCTION COMPOSITION
        test_cases.extend([
            TestCase(
                test_id="COMP-001",
                category="Functions",
                concept="FUNCTION_COMPOSITION",
                question="If f(x) = x + 2 and g(x) = 3x, find f(g(1))",
                correct_answer="5",
                student_answer="6",
                student_profile_type="advanced_middle",
                expected_missing_concept="FUNCTION_COMPOSITION",
                expected_confidence_range=(0.8, 0.95),
                expected_remediation_level="hard",
                description="Calculated g(f(1)) instead of f(g(1))",
                question_type="function_composition"
            ),
            
            TestCase(
                test_id="COMP-002",
                category="Function Composition",
                concept="FUNCTION_COMPOSITION",
                question="If h(x) = 2x and k(x) = x - 1, find h(k(4))",
                correct_answer="6",
                student_answer="7",
                student_profile_type="advanced_high",
                expected_missing_concept="FUNCTION_EVALUATION",
                expected_confidence_range=(0.85, 0.95),
                expected_remediation_level="bonus",
                description="Minor calculation error in composition",
                question_type="function_composition"
            )
        ])
        
        # FRACTION OPERATIONS
        test_cases.extend([
            TestCase(
                test_id="FRAC-001",
                category="Fractions",
                concept="FRACTION_OPERATIONS",
                question="Calculate: 2/3 × 1.5",
                correct_answer="1",
                student_answer="2.17",
                student_profile_type="struggling_middle",
                expected_missing_concept="FRACTIONS_AS_PART_WHOLE",
                expected_confidence_range=(0.8, 0.9),
                expected_remediation_level="easy",
                description="Added instead of multiplying",
                question_type="fraction_operations"
            ),
            
            TestCase(
                test_id="FRAC-002",
                category="Fractions",
                concept="FRACTION_OPERATIONS",
                question="Simplify: 6/8",
                correct_answer="3/4",
                student_answer="6/8",
                student_profile_type="struggling_middle",
                expected_missing_concept="FRACTION_SIMPLIFICATION",
                expected_confidence_range=(0.85, 0.95),
                expected_remediation_level="easy",
                description="Didn't simplify fraction",
                question_type="fraction_simplification"
            )
        ])
        
        # QUADRATIC EQUATIONS
        test_cases.extend([
            TestCase(
                test_id="QUAD-001",
                category="Quadratics",
                concept="QUADRATIC_EQUATIONS",
                question="Solve: x² - 5x + 6 = 0",
                correct_answer="x = 2, 3",
                student_answer="x = 1, 6",
                student_profile_type="average_high",
                expected_missing_concept="FACTORING",
                expected_confidence_range=(0.85, 0.95),
                expected_remediation_level="medium",
                description="Incorrect factoring of quadratic",
                question_type="quadratic_equations"
            ),
            
            TestCase(
                test_id="QUAD-002",
                category="Quadratics",
                concept="QUADRATIC_FORMULA",
                question="Find the vertex of y = x² - 4x + 3",
                correct_answer="(2, -1)",
                student_answer="(-2, 15)",
                student_profile_type="advanced_middle",
                expected_missing_concept="COMPLETING_THE_SQUARE",
                expected_confidence_range=(0.85, 0.95),
                expected_remediation_level="hard",
                description="Sign error in vertex formula",
                question_type="quadratic_equations"
            )
        ])
        
        # EDGE CASES AND BOUNDARY CONDITIONS
        test_cases.extend([
            TestCase(
                test_id="EDGE-001",
                category="Edge Cases",
                concept="UNKNOWN_CONCEPT",
                question="What is 2 + 2?",
                correct_answer="4",
                student_answer="fish",
                student_profile_type="struggling_elementary",
                expected_missing_concept="UNKNOWN_CONCEPT",
                expected_confidence_range=(0.1, 0.3),
                expected_remediation_level="easy",
                description="Completely nonsensical answer"
            ),
            
            TestCase(
                test_id="EDGE-002",
                category="Edge Cases",
                concept="ORDER_OF_OPERATIONS",
                question="Calculate: 5 × 3 + 2",
                correct_answer="17",
                student_answer="17",
                student_profile_type="struggling_elementary",
                expected_missing_concept="",
                expected_confidence_range=(0.95, 1.0),
                expected_remediation_level="",
                description="Struggling student gets difficult problem right"
            ),
            
            TestCase(
                test_id="EDGE-003",
                category="Edge Cases",
                concept="BASIC_ARITHMETIC",
                question="What is 3 + 2?",
                correct_answer="5",
                student_answer="6",
                student_profile_type="advanced_high",
                expected_missing_concept="ADDITION_FACTS",
                expected_confidence_range=(0.85, 0.95),
                expected_remediation_level="bonus",
                description="Advanced student makes basic error",
                question_type="basic_arithmetic"
            )
        ])
        
        return test_cases
    
    def run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case and validate results"""
        
        # Get student profile
        student_profile = self.student_profiles[test_case.student_profile_type]
        
        # Record start time
        start_time = time.time()
        
        # Run diagnostic
        diagnostic_result = self.diagnostic_system.diagnose_student_response(
            question_text=test_case.question,
            student_answer=test_case.student_answer,
            correct_answer=test_case.correct_answer,
            student_profile=student_profile,
            question_type=test_case.question_type
        )
        
        # Record processing time
        processing_time = time.time() - start_time
        
        # Validate results
        concept_match = (diagnostic_result.missing_concept == test_case.expected_missing_concept or
                        (diagnostic_result.is_correct and test_case.expected_missing_concept == ""))
        
        confidence_in_range = (test_case.expected_confidence_range[0] <= 
                              diagnostic_result.confidence_score <= 
                              test_case.expected_confidence_range[1])
        
        remediation_appropriate = (diagnostic_result.remediation_level == test_case.expected_remediation_level or
                                 diagnostic_result.is_correct)
        
        overall_success = concept_match and confidence_in_range and remediation_appropriate
        
        return TestResult(
            test_case=test_case,
            diagnostic_result=diagnostic_result,
            concept_match=concept_match,
            confidence_in_range=confidence_in_range,
            remediation_appropriate=remediation_appropriate,
            processing_time=processing_time,
            overall_success=overall_success
        )
    
    def run_comprehensive_test_suite(self) -> Dict:
        """Run the complete test suite and generate comprehensive report"""
        
        print("🚀 COMPREHENSIVE K-12 ALGEBRA DIAGNOSTIC SYSTEM TEST SUITE")
        print("=" * 80)
        print(f"📊 Running {len(self.test_cases)} comprehensive test cases...")
        print("=" * 80)
        
        # Run all tests
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n🧪 Running Test {i}/{len(self.test_cases)}: {test_case.test_id}")
            print(f"   📝 {test_case.description}")
            
            result = self.run_single_test(test_case)
            self.test_results.append(result)
            
            # Track performance metrics
            self.performance_metrics['processing_times'].append(result.processing_time)
            self.performance_metrics['confidence_scores'].append(result.diagnostic_result.confidence_score)
            
            # Show result summary
            status = "✅ PASS" if result.overall_success else "❌ FAIL"
            print(f"   {status} | Confidence: {result.diagnostic_result.confidence_score:.1%} | "
                  f"Concept: {result.diagnostic_result.missing_concept}")
        
        # Generate comprehensive report
        return self._generate_comprehensive_report()
    
    def _generate_comprehensive_report(self) -> Dict:
        """Generate detailed test report with analysis"""
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.overall_success)
        concept_matches = sum(1 for r in self.test_results if r.concept_match)
        confidence_matches = sum(1 for r in self.test_results if r.confidence_in_range)
        remediation_matches = sum(1 for r in self.test_results if r.remediation_appropriate)
        
        # Calculate metrics
        overall_success_rate = successful_tests / total_tests
        concept_accuracy = concept_matches / total_tests
        confidence_accuracy = confidence_matches / total_tests
        remediation_accuracy = remediation_matches / total_tests
        
        avg_processing_time = sum(self.performance_metrics['processing_times']) / len(self.performance_metrics['processing_times'])
        avg_confidence = sum(self.performance_metrics['confidence_scores']) / len(self.performance_metrics['confidence_scores'])
        
        # Category analysis
        category_results = defaultdict(lambda: {'total': 0, 'success': 0})
        for result in self.test_results:
            category = result.test_case.category
            category_results[category]['total'] += 1
            if result.overall_success:
                category_results[category]['success'] += 1
        
        # Student profile analysis
        profile_results = defaultdict(lambda: {'total': 0, 'success': 0})
        for result in self.test_results:
            profile = result.test_case.student_profile_type
            profile_results[profile]['total'] += 1
            if result.overall_success:
                profile_results[profile]['success'] += 1
        
        # Print comprehensive report
        print(f"\n{'📊 COMPREHENSIVE TEST REPORT':-^80}")
        print(f"🎯 OVERALL PERFORMANCE:")
        print(f"   • Total Tests: {total_tests}")
        print(f"   • Overall Success Rate: {overall_success_rate:.1%}")
        print(f"   • Concept Identification Accuracy: {concept_accuracy:.1%}")
        print(f"   • Confidence Score Accuracy: {confidence_accuracy:.1%}")
        print(f"   • Remediation Appropriateness: {remediation_accuracy:.1%}")
        print(f"   • Average Processing Time: {avg_processing_time:.3f}s")
        print(f"   • Average Confidence Score: {avg_confidence:.1%}")
        
        print(f"\n📈 PERFORMANCE BY CATEGORY:")
        for category, stats in category_results.items():
            success_rate = stats['success'] / stats['total']
            print(f"   • {category}: {success_rate:.1%} ({stats['success']}/{stats['total']})")
        
        print(f"\n👥 PERFORMANCE BY STUDENT PROFILE:")
        for profile, stats in profile_results.items():
            success_rate = stats['success'] / stats['total']
            print(f"   • {profile}: {success_rate:.1%} ({stats['success']}/{stats['total']})")
        
        # Identify problem areas
        failed_tests = [r for r in self.test_results if not r.overall_success]
        if failed_tests:
            print(f"\n🚨 FAILED TESTS ANALYSIS:")
            for result in failed_tests[:10]:  # Show first 10 failures
                print(f"   ❌ {result.test_case.test_id}: {result.test_case.description}")
                print(f"      Expected: {result.test_case.expected_missing_concept}")
                print(f"      Got: {result.diagnostic_result.missing_concept}")
                print(f"      Confidence: {result.diagnostic_result.confidence_score:.1%}")
        
        # Performance assessment
        print(f"\n🎯 SYSTEM ASSESSMENT:")
        if overall_success_rate >= 0.85:
            print("   🟢 EXCELLENT: System performing at production level")
        elif overall_success_rate >= 0.70:
            print("   🟡 GOOD: System performing well with minor improvements needed")
        elif overall_success_rate >= 0.50:
            print("   🟠 FAIR: System needs significant improvements")
        else:
            print("   🔴 POOR: System requires major fixes before deployment")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if concept_accuracy < 0.80:
            print("   • Improve concept identification algorithms")
        if confidence_accuracy < 0.80:
            print("   • Calibrate confidence score calculations")
        if remediation_accuracy < 0.80:
            print("   • Enhance remediation level assignment logic")
        if avg_processing_time > 0.1:
            print("   • Optimize processing speed for real-time use")
        
        return {
            'total_tests': total_tests,
            'overall_success_rate': overall_success_rate,
            'concept_accuracy': concept_accuracy,
            'confidence_accuracy': confidence_accuracy,
            'remediation_accuracy': remediation_accuracy,
            'avg_processing_time': avg_processing_time,
            'avg_confidence': avg_confidence,
            'category_results': dict(category_results),
            'profile_results': dict(profile_results),
            'failed_tests': len(failed_tests)
        }

def main():
    """Run the comprehensive system test"""
    
    # Initialize and run test suite
    test_system = ComprehensiveSystemTest()
    results = test_system.run_comprehensive_test_suite()
    
    print(f"\n{'🎉 COMPREHENSIVE TESTING COMPLETE':-^80}")
    print(f"📊 System ready for production: {results['overall_success_rate'] >= 0.80}")
    print(f"🚀 K-12 Algebra Diagnostic System: Comprehensive validation complete!")

if __name__ == "__main__":
    main() 