#!/usr/bin/env python3
"""
QUESTION BANK INTEGRATION & RECOMMENDATION VALIDATION TEST
Specialized testing for question bank integration, recommendation accuracy,
and missing concept targeting with detailed validation of remediation paths.
"""

import sys
import os
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_diagnostic_system import EnhancedDiagnosticSystem, EnhancedDiagnosticResult
from comprehensive_diagnostic import StudentProfile
from algebra_comprehensive_question_bank_complete import CompleteAlgebraQuestionBank

@dataclass
class QuestionBankTestCase:
    """Test case for question bank integration"""
    test_id: str
    missing_concept: str
    student_ability: float
    expected_difficulty_levels: List[str]
    expected_question_count_range: Tuple[int, int]
    expected_sources: List[str]
    learning_style: str
    description: str

@dataclass
class RecommendationTestResult:
    """Result of recommendation validation"""
    test_case: QuestionBankTestCase
    recommended_questions: List[Dict]
    difficulty_match: bool
    count_in_range: bool
    source_diversity: bool
    concept_relevance: bool
    learning_style_match: bool
    overall_success: bool
    processing_time: float

class QuestionBankIntegrationTest:
    """Comprehensive question bank integration testing"""
    
    def __init__(self):
        self.diagnostic_system = EnhancedDiagnosticSystem()
        self.question_bank = CompleteAlgebraQuestionBank()
        
        # Initialize test cases
        self.test_cases = self._initialize_question_bank_test_cases()
        self.test_results: List[RecommendationTestResult] = []
    
    def _initialize_question_bank_test_cases(self) -> List[QuestionBankTestCase]:
        """Initialize comprehensive question bank test cases"""
        
        return [
            # ORDER OF OPERATIONS TESTS
            QuestionBankTestCase(
                test_id="QB-OOO-001",
                missing_concept="ORDER_OF_OPERATIONS",
                student_ability=0.20,
                expected_difficulty_levels=["easy"],
                expected_question_count_range=(3, 8),
                expected_sources=["Khan Academy", "Kumon", "Singapore Math"],
                learning_style="visual",
                description="Struggling student needs basic order of operations practice"
            ),
            
            QuestionBankTestCase(
                test_id="QB-OOO-002",
                missing_concept="ORDER_OF_OPERATIONS",
                student_ability=0.60,
                expected_difficulty_levels=["medium", "hard"],
                expected_question_count_range=(4, 10),
                expected_sources=["Khan Academy", "RSM", "Beast Academy"],
                learning_style="analytical",
                description="Average student needs intermediate order of operations"
            ),
            
            QuestionBankTestCase(
                test_id="QB-OOO-003",
                missing_concept="ORDER_OF_OPERATIONS",
                student_ability=0.85,
                expected_difficulty_levels=["hard", "bonus"],
                expected_question_count_range=(3, 6),
                expected_sources=["AoPS", "Beast Academy", "RSM"],
                learning_style="hands_on",
                description="Advanced student needs challenging order of operations"
            ),
            
            # TWO-STEP EQUATIONS TESTS
            QuestionBankTestCase(
                test_id="QB-TSE-001",
                missing_concept="TWO_STEP_EQUATIONS",
                student_ability=0.25,
                expected_difficulty_levels=["easy"],
                expected_question_count_range=(4, 8),
                expected_sources=["Khan Academy", "Kumon"],
                learning_style="visual",
                description="Struggling student needs basic two-step equations"
            ),
            
            QuestionBankTestCase(
                test_id="QB-TSE-002",
                missing_concept="TWO_STEP_EQUATIONS",
                student_ability=0.55,
                expected_difficulty_levels=["medium"],
                expected_question_count_range=(5, 10),
                expected_sources=["Khan Academy", "Singapore Math", "RSM"],
                learning_style="analytical",
                description="Average student needs standard two-step equations"
            ),
            
            # DISTRIBUTIVE PROPERTY TESTS
            QuestionBankTestCase(
                test_id="QB-DIST-001",
                missing_concept="DISTRIBUTIVE_PROPERTY",
                student_ability=0.30,
                expected_difficulty_levels=["easy"],
                expected_question_count_range=(3, 7),
                expected_sources=["Khan Academy", "Kumon"],
                learning_style="hands_on",
                description="Struggling student needs basic distributive property"
            ),
            
            QuestionBankTestCase(
                test_id="QB-DIST-002",
                missing_concept="DISTRIBUTIVE_PROPERTY",
                student_ability=0.70,
                expected_difficulty_levels=["medium", "hard"],
                expected_question_count_range=(4, 8),
                expected_sources=["RSM", "Beast Academy"],
                learning_style="visual",
                description="Good student needs advanced distributive property"
            ),
            
            # INTEGER OPERATIONS TESTS
            QuestionBankTestCase(
                test_id="QB-INT-001",
                missing_concept="INTEGER_OPERATIONS",
                student_ability=0.15,
                expected_difficulty_levels=["easy"],
                expected_question_count_range=(5, 10),
                expected_sources=["Khan Academy", "Kumon"],
                learning_style="visual",
                description="Very struggling student needs basic integer operations"
            ),
            
            QuestionBankTestCase(
                test_id="QB-INT-002",
                missing_concept="INTEGER_OPERATIONS",
                student_ability=0.45,
                expected_difficulty_levels=["easy", "medium"],
                expected_question_count_range=(4, 8),
                expected_sources=["Khan Academy", "Singapore Math"],
                learning_style="analytical",
                description="Below average student needs integer operations practice"
            ),
            
            # FUNCTION EVALUATION TESTS
            QuestionBankTestCase(
                test_id="QB-FUNC-001",
                missing_concept="FUNCTION_EVALUATION",
                student_ability=0.50,
                expected_difficulty_levels=["medium"],
                expected_question_count_range=(3, 6),
                expected_sources=["Khan Academy", "RSM"],
                learning_style="analytical",
                description="Average student needs function evaluation practice"
            ),
            
            QuestionBankTestCase(
                test_id="QB-FUNC-002",
                missing_concept="FUNCTION_EVALUATION",
                student_ability=0.80,
                expected_difficulty_levels=["hard"],
                expected_question_count_range=(3, 5),
                expected_sources=["AoPS", "Beast Academy"],
                learning_style="hands_on",
                description="Advanced student needs challenging function evaluation"
            ),
            
            # FUNCTION COMPOSITION TESTS
            QuestionBankTestCase(
                test_id="QB-COMP-001",
                missing_concept="FUNCTION_COMPOSITION",
                student_ability=0.65,
                expected_difficulty_levels=["medium", "hard"],
                expected_question_count_range=(3, 6),
                expected_sources=["RSM", "Beast Academy"],
                learning_style="visual",
                description="Good student needs function composition practice"
            ),
            
            QuestionBankTestCase(
                test_id="QB-COMP-002",
                missing_concept="FUNCTION_COMPOSITION",
                student_ability=0.90,
                expected_difficulty_levels=["hard", "bonus"],
                expected_question_count_range=(2, 4),
                expected_sources=["AoPS", "Beast Academy"],
                learning_style="analytical",
                description="Excellent student needs advanced function composition"
            ),
            
            # FRACTION OPERATIONS TESTS
            QuestionBankTestCase(
                test_id="QB-FRAC-001",
                missing_concept="FRACTION_OPERATIONS",
                student_ability=0.35,
                expected_difficulty_levels=["easy"],
                expected_question_count_range=(4, 8),
                expected_sources=["Khan Academy", "Kumon", "Singapore Math"],
                learning_style="visual",
                description="Struggling student needs basic fraction operations"
            ),
            
            QuestionBankTestCase(
                test_id="QB-FRAC-002",
                missing_concept="FRACTION_OPERATIONS",
                student_ability=0.75,
                expected_difficulty_levels=["medium", "hard"],
                expected_question_count_range=(3, 6),
                expected_sources=["RSM", "Beast Academy"],
                learning_style="hands_on",
                description="Strong student needs advanced fraction operations"
            ),
            
            # QUADRATIC FORMULA TESTS
            QuestionBankTestCase(
                test_id="QB-QUAD-001",
                missing_concept="QUADRATIC_FORMULA",
                student_ability=0.55,
                expected_difficulty_levels=["medium"],
                expected_question_count_range=(3, 6),
                expected_sources=["Khan Academy", "RSM"],
                learning_style="analytical",
                description="Average student needs quadratic formula practice"
            ),
            
            QuestionBankTestCase(
                test_id="QB-QUAD-002",
                missing_concept="QUADRATIC_FORMULA",
                student_ability=0.85,
                expected_difficulty_levels=["hard", "bonus"],
                expected_question_count_range=(2, 5),
                expected_sources=["AoPS", "Beast Academy"],
                learning_style="visual",
                description="Advanced student needs challenging quadratics"
            ),
            
            # EDGE CASES
            QuestionBankTestCase(
                test_id="QB-EDGE-001",
                missing_concept="UNKNOWN_CONCEPT",
                student_ability=0.40,
                expected_difficulty_levels=["easy", "medium"],
                expected_question_count_range=(0, 3),
                expected_sources=[],
                learning_style="visual",
                description="Unknown concept should return minimal or no recommendations"
            ),
            
            QuestionBankTestCase(
                test_id="QB-EDGE-002",
                missing_concept="BASIC_ARITHMETIC",
                student_ability=0.95,
                expected_difficulty_levels=["bonus"],
                expected_question_count_range=(1, 3),
                expected_sources=["AoPS"],
                learning_style="analytical",
                description="Excellent student with basic error needs minimal practice"
            )
        ]
    
    def run_single_question_bank_test(self, test_case: QuestionBankTestCase) -> RecommendationTestResult:
        """Run a single question bank integration test"""
        
        # Create student profile for testing
        student_profile = StudentProfile(
            name=f"Test Student ({test_case.student_ability:.0%} ability)",
            overall_ability=test_case.student_ability,
            concept_mastery={test_case.missing_concept: test_case.student_ability * 0.5},
            learning_style=test_case.learning_style,
            total_questions_attempted=50,
            total_questions_correct=int(50 * test_case.student_ability)
        )
        
        # Record start time
        start_time = time.time()
        
        # Get question recommendations
        recommended_questions = self.question_bank.get_remediation_questions(
            missing_concept=test_case.missing_concept,
            student_profile=student_profile,
            num_questions=8  # Request up to 8 questions
        )
        
        # Record processing time
        processing_time = time.time() - start_time
        
        # Validate recommendations
        difficulty_match = self._validate_difficulty_levels(
            recommended_questions, test_case.expected_difficulty_levels
        )
        
        count_in_range = (test_case.expected_question_count_range[0] <= 
                         len(recommended_questions) <= 
                         test_case.expected_question_count_range[1])
        
        source_diversity = self._validate_source_diversity(
            recommended_questions, test_case.expected_sources
        )
        
        concept_relevance = self._validate_concept_relevance(
            recommended_questions, test_case.missing_concept
        )
        
        learning_style_match = self._validate_learning_style_match(
            recommended_questions, test_case.learning_style
        )
        
        overall_success = (difficulty_match and count_in_range and 
                          source_diversity and concept_relevance)
        
        return RecommendationTestResult(
            test_case=test_case,
            recommended_questions=recommended_questions,
            difficulty_match=difficulty_match,
            count_in_range=count_in_range,
            source_diversity=source_diversity,
            concept_relevance=concept_relevance,
            learning_style_match=learning_style_match,
            overall_success=overall_success,
            processing_time=processing_time
        )
    
    def _validate_difficulty_levels(self, questions: List[Dict], expected_levels: List[str]) -> bool:
        """Validate that recommended questions have appropriate difficulty levels"""
        if not questions:
            return len(expected_levels) == 0
        
        question_difficulties = [q.get('difficulty', 'medium') for q in questions]
        
        # Check if at least 70% of questions match expected difficulty levels
        matching_count = sum(1 for diff in question_difficulties if diff in expected_levels)
        return matching_count >= len(questions) * 0.7
    
    def _validate_source_diversity(self, questions: List[Dict], expected_sources: List[str]) -> bool:
        """Validate that questions come from appropriate sources"""
        if not questions:
            return len(expected_sources) == 0
        
        question_sources = [q.get('source', 'Unknown') for q in questions]
        
        # Check if questions come from expected sources
        matching_sources = sum(1 for source in question_sources if source in expected_sources)
        return matching_sources >= len(questions) * 0.6  # At least 60% from expected sources
    
    def _validate_concept_relevance(self, questions: List[Dict], target_concept: str) -> bool:
        """Validate that questions are relevant to the target concept"""
        if not questions:
            return target_concept == "UNKNOWN_CONCEPT"
        
        # Check if questions target the missing concept or related concepts
        relevant_count = 0
        for question in questions:
            question_concepts = question.get('concepts', [])
            if target_concept in question_concepts:
                relevant_count += 1
            elif any(concept.replace('_', ' ').lower() in target_concept.replace('_', ' ').lower() 
                    for concept in question_concepts):
                relevant_count += 1
        
        return relevant_count >= len(questions) * 0.8  # At least 80% relevant
    
    def _validate_learning_style_match(self, questions: List[Dict], learning_style: str) -> bool:
        """Validate that questions match the student's learning style"""
        if not questions:
            return True
        
        # Check if questions have appropriate format for learning style
        style_appropriate_count = 0
        for question in questions:
            question_format = question.get('format', 'text')
            
            if learning_style == "visual" and question_format in ['visual', 'diagram', 'graph']:
                style_appropriate_count += 1
            elif learning_style == "hands_on" and question_format in ['interactive', 'hands_on', 'manipulative']:
                style_appropriate_count += 1
            elif learning_style == "analytical" and question_format in ['text', 'analytical', 'proof']:
                style_appropriate_count += 1
            else:
                # Default text questions are acceptable for any learning style
                style_appropriate_count += 1
        
        return style_appropriate_count >= len(questions) * 0.5  # At least 50% style-appropriate
    
    def run_comprehensive_question_bank_test(self) -> Dict:
        """Run the complete question bank integration test suite"""
        
        print("🎯 QUESTION BANK INTEGRATION & RECOMMENDATION TEST SUITE")
        print("=" * 80)
        print(f"📚 Testing question bank integration with {len(self.test_cases)} test cases...")
        print("=" * 80)
        
        # Run all tests
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n📖 Running Test {i}/{len(self.test_cases)}: {test_case.test_id}")
            print(f"   🎯 {test_case.description}")
            print(f"   👤 Student Ability: {test_case.student_ability:.0%} | Style: {test_case.learning_style}")
            
            result = self.run_single_question_bank_test(test_case)
            self.test_results.append(result)
            
            # Show result summary
            status = "✅ PASS" if result.overall_success else "❌ FAIL"
            print(f"   {status} | Questions: {len(result.recommended_questions)} | "
                  f"Time: {result.processing_time:.3f}s")
            
            # Show detailed validation
            print(f"      Difficulty: {'✓' if result.difficulty_match else '✗'} | "
                  f"Count: {'✓' if result.count_in_range else '✗'} | "
                  f"Sources: {'✓' if result.source_diversity else '✗'} | "
                  f"Relevance: {'✓' if result.concept_relevance else '✗'}")
        
        # Generate comprehensive report
        return self._generate_question_bank_report()
    
    def _generate_question_bank_report(self) -> Dict:
        """Generate detailed question bank integration report"""
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.overall_success)
        difficulty_matches = sum(1 for r in self.test_results if r.difficulty_match)
        count_matches = sum(1 for r in self.test_results if r.count_in_range)
        source_matches = sum(1 for r in self.test_results if r.source_diversity)
        relevance_matches = sum(1 for r in self.test_results if r.concept_relevance)
        style_matches = sum(1 for r in self.test_results if r.learning_style_match)
        
        # Calculate metrics
        overall_success_rate = successful_tests / total_tests
        difficulty_accuracy = difficulty_matches / total_tests
        count_accuracy = count_matches / total_tests
        source_accuracy = source_matches / total_tests
        relevance_accuracy = relevance_matches / total_tests
        style_accuracy = style_matches / total_tests
        
        avg_processing_time = sum(r.processing_time for r in self.test_results) / total_tests
        avg_questions_per_request = sum(len(r.recommended_questions) for r in self.test_results) / total_tests
        
        # Concept analysis
        concept_results = defaultdict(lambda: {'total': 0, 'success': 0})
        for result in self.test_results:
            concept = result.test_case.missing_concept
            concept_results[concept]['total'] += 1
            if result.overall_success:
                concept_results[concept]['success'] += 1
        
        # Ability level analysis
        ability_ranges = {
            'struggling': (0.0, 0.3),
            'below_average': (0.3, 0.5),
            'average': (0.5, 0.7),
            'above_average': (0.7, 0.85),
            'advanced': (0.85, 1.0)
        }
        
        ability_results = defaultdict(lambda: {'total': 0, 'success': 0})
        for result in self.test_results:
            ability = result.test_case.student_ability
            for level, (min_val, max_val) in ability_ranges.items():
                if min_val <= ability < max_val:
                    ability_results[level]['total'] += 1
                    if result.overall_success:
                        ability_results[level]['success'] += 1
                    break
        
        # Print comprehensive report
        print(f"\n{'📊 QUESTION BANK INTEGRATION REPORT':-^80}")
        print(f"🎯 OVERALL PERFORMANCE:")
        print(f"   • Total Tests: {total_tests}")
        print(f"   • Overall Success Rate: {overall_success_rate:.1%}")
        print(f"   • Difficulty Level Accuracy: {difficulty_accuracy:.1%}")
        print(f"   • Question Count Accuracy: {count_accuracy:.1%}")
        print(f"   • Source Diversity: {source_accuracy:.1%}")
        print(f"   • Concept Relevance: {relevance_accuracy:.1%}")
        print(f"   • Learning Style Match: {style_accuracy:.1%}")
        print(f"   • Average Processing Time: {avg_processing_time:.3f}s")
        print(f"   • Average Questions per Request: {avg_questions_per_request:.1f}")
        
        print(f"\n📈 PERFORMANCE BY CONCEPT:")
        for concept, stats in concept_results.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total']
                print(f"   • {concept}: {success_rate:.1%} ({stats['success']}/{stats['total']})")
        
        print(f"\n👥 PERFORMANCE BY ABILITY LEVEL:")
        for level, stats in ability_results.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total']
                print(f"   • {level.replace('_', ' ').title()}: {success_rate:.1%} ({stats['success']}/{stats['total']})")
        
        # Identify problem areas
        failed_tests = [r for r in self.test_results if not r.overall_success]
        if failed_tests:
            print(f"\n🚨 FAILED TESTS ANALYSIS:")
            for result in failed_tests[:5]:  # Show first 5 failures
                print(f"   ❌ {result.test_case.test_id}: {result.test_case.description}")
                print(f"      Questions returned: {len(result.recommended_questions)}")
                print(f"      Expected range: {result.test_case.expected_question_count_range}")
        
        # Question bank assessment
        print(f"\n🎯 QUESTION BANK ASSESSMENT:")
        if overall_success_rate >= 0.85:
            print("   🟢 EXCELLENT: Question bank integration working perfectly")
        elif overall_success_rate >= 0.70:
            print("   🟡 GOOD: Question bank working well with minor issues")
        elif overall_success_rate >= 0.50:
            print("   🟠 FAIR: Question bank needs improvements")
        else:
            print("   🔴 POOR: Question bank integration needs major fixes")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if difficulty_accuracy < 0.80:
            print("   • Improve difficulty level assignment algorithm")
        if count_accuracy < 0.80:
            print("   • Adjust question count logic for different concepts")
        if source_accuracy < 0.70:
            print("   • Enhance source diversity in question selection")
        if relevance_accuracy < 0.85:
            print("   • Improve concept relevance matching")
        if style_accuracy < 0.70:
            print("   • Add more learning style-specific questions")
        
        return {
            'total_tests': total_tests,
            'overall_success_rate': overall_success_rate,
            'difficulty_accuracy': difficulty_accuracy,
            'count_accuracy': count_accuracy,
            'source_accuracy': source_accuracy,
            'relevance_accuracy': relevance_accuracy,
            'style_accuracy': style_accuracy,
            'avg_processing_time': avg_processing_time,
            'avg_questions_per_request': avg_questions_per_request,
            'concept_results': dict(concept_results),
            'ability_results': dict(ability_results),
            'failed_tests': len(failed_tests)
        }

def main():
    """Run the question bank integration test"""
    
    # Initialize and run test suite
    test_system = QuestionBankIntegrationTest()
    results = test_system.run_comprehensive_question_bank_test()
    
    print(f"\n{'🎉 QUESTION BANK TESTING COMPLETE':-^80}")
    print(f"📚 Question bank ready for production: {results['overall_success_rate'] >= 0.75}")
    print(f"🚀 Recommendation system validation complete!")

if __name__ == "__main__":
    main() 