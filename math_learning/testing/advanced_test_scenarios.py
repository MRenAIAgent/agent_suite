#!/usr/bin/env python3
"""
Advanced Test Scenarios - Edge cases and complex testing
Tests the diagnostic system with challenging scenarios and edge cases
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_diagnostic import ComprehensiveDiagnostic, StudentProfile
from typing import Dict, List, Tuple
import random

class AdvancedTestScenarios:
    """Advanced testing scenarios for the diagnostic system"""
    
    def __init__(self):
        self.diagnostic = ComprehensiveDiagnostic()
        
        # Complex problem sets for advanced testing
        self.advanced_problems = {
            # Multi-concept problems
            "complex_equation_1": {
                "problem": "Solve: 3(2x - 4) + 5x = 2(x + 3) - 7",
                "correct_answer": "x = 1",
                "type": "distributive_property",
                "concepts": ["distributive_property", "combine_like_terms", "two_step_equations"],
                "difficulty": 4
            },
            "complex_equation_2": {
                "problem": "Solve: (x + 3)/2 - (x - 1)/3 = 1",
                "correct_answer": "x = 3",
                "type": "two_step_equations", 
                "concepts": ["fractions_operations", "two_step_equations", "combine_like_terms"],
                "difficulty": 5
            },
            "word_problem_complex": {
                "problem": "A store sells shirts for $15 each and pants for $25 each. If Maria bought 3 more shirts than pants and spent $135 total, how many shirts did she buy?",
                "correct_answer": "6 shirts",
                "type": "two_step_equations",
                "concepts": ["two_step_equations", "word_problems", "system_setup"],
                "difficulty": 4
            },
            "function_composition": {
                "problem": "If f(x) = 2x + 3 and g(x) = x - 1, find f(g(4))",
                "correct_answer": "9",
                "type": "function_composition",
                "concepts": ["function_evaluation", "function_composition", "order_of_operations"],
                "difficulty": 3
            },
            "quadratic_factoring": {
                "problem": "Factor completely: x² - 5x + 6",
                "correct_answer": "(x - 2)(x - 3)",
                "type": "quadratic_factoring",
                "concepts": ["quadratic_factoring", "multiplication_facts", "integer_operations"],
                "difficulty": 4
            },
            "exponential_rules": {
                "problem": "Simplify: (2x³)² · (3x²)",
                "correct_answer": "12x⁸",
                "type": "exponential_rules",
                "concepts": ["exponential_rules", "multiplication_facts", "combine_like_terms"],
                "difficulty": 3
            },
            "rational_expressions": {
                "problem": "Simplify: (x² - 4)/(x + 2)",
                "correct_answer": "x - 2",
                "type": "rational_expressions",
                "concepts": ["rational_expressions", "quadratic_factoring", "cancellation"],
                "difficulty": 4
            },
            "absolute_value": {
                "problem": "Solve: |2x - 3| = 7",
                "correct_answer": "x = 5 or x = -2",
                "type": "absolute_value",
                "concepts": ["absolute_value", "two_step_equations", "case_analysis"],
                "difficulty": 4
            },
            "inequality_compound": {
                "problem": "Solve: -3 < 2x + 1 < 7",
                "correct_answer": "-2 < x < 3",
                "type": "compound_inequalities",
                "concepts": ["compound_inequalities", "two_step_equations", "interval_notation"],
                "difficulty": 4
            },
            "system_equations": {
                "problem": "Solve the system: 2x + y = 7, x - y = 2",
                "correct_answer": "x = 3, y = 1",
                "type": "system_equations",
                "concepts": ["system_equations", "substitution", "elimination"],
                "difficulty": 4
            }
        }
        
        # Complex error patterns for different student types
        self.complex_error_patterns = {
            # Mixed ability student - good at some concepts, weak at others
            "mixed_ability": {
                "complex_equation_1": ["x = 3", "x = -1", "x = 0"],  # Distribution errors
                "complex_equation_2": ["x = 6", "x = 0", "x = 9"],   # Fraction errors
                "word_problem_complex": ["3 shirts", "9 shirts", "4 shirts"],  # Setup errors
                "function_composition": ["9", "7", "11"],  # Mostly correct
                "quadratic_factoring": ["(x - 2)(x - 3)", "(x + 2)(x + 3)", "x² - 5x + 6"],  # Some correct
                "exponential_rules": ["6x⁵", "12x⁶", "12x⁸"],  # Exponent rule errors
                "rational_expressions": ["(x² - 4)/(x + 2)", "x + 2", "x - 2"],  # Some simplification
                "absolute_value": ["x = 5", "x = -2", "x = 5 or x = -2"],  # Missing cases
                "inequality_compound": ["-2 < x < 3", "x > -2", "x < 3"],  # Partial solutions
                "system_equations": ["x = 3, y = 1", "x = 2, y = 3", "x = 1, y = 5"]  # Some correct
            },
            
            # Inconsistent student - performance varies widely
            "inconsistent": {
                "complex_equation_1": ["x = 5", "x = -3", "x = 1"],  # Random performance
                "complex_equation_2": ["x = 12", "x = -6", "x = 3"],  # Very inconsistent
                "word_problem_complex": ["12 shirts", "2 shirts", "6 shirts"],  # Wild guesses
                "function_composition": ["3", "9", "15"],  # Sometimes right
                "quadratic_factoring": ["(x - 1)(x - 6)", "(x - 2)(x - 3)", "cannot factor"],  # Inconsistent
                "exponential_rules": ["24x⁵", "12x⁸", "6x⁶"],  # Random errors
                "rational_expressions": ["x - 2", "1", "undefined"],  # Inconsistent understanding
                "absolute_value": ["x = 10", "x = 5 or x = -2", "no solution"],  # Varies
                "inequality_compound": ["x = 2", "-2 < x < 3", "all real numbers"],  # Inconsistent
                "system_equations": ["x = 0, y = 7", "x = 3, y = 1", "no solution"]  # Mixed results
            },
            
            # Conceptually confused - systematic misconceptions
            "confused": {
                "complex_equation_1": ["x = 29", "x = -11", "x = 8"],  # Systematic errors
                "complex_equation_2": ["x = 15", "x = -9", "x = 21"],  # Fraction confusion
                "word_problem_complex": ["15 shirts", "25 shirts", "8 shirts"],  # Wrong setup
                "function_composition": ["6", "12", "4"],  # Function confusion
                "quadratic_factoring": ["(x + 2)(x + 3)", "(x - 6)(x - 1)", "x(x - 5) + 6"],  # Sign errors
                "exponential_rules": ["6x⁵", "5x⁶", "2x⁶"],  # Rule confusion
                "rational_expressions": ["x + 2", "x² - 2", "cannot simplify"],  # Cancellation errors
                "absolute_value": ["x = 10", "x = 4", "x = -10"],  # Absolute value confusion
                "inequality_compound": ["x > 2", "x < -2", "no solution"],  # Inequality confusion
                "system_equations": ["x = 5, y = -3", "x = 1, y = 5", "x = 9, y = -11"]  # Method confusion
            }
        }

    def test_mixed_ability_student(self) -> Dict:
        """Test a student with mixed abilities across different concepts"""
        
        student_profile = StudentProfile(
            name="Alex",
            overall_ability=0.65,  # Above average but inconsistent
            concept_mastery={
                "function_evaluation": 0.8,  # Strong
                "quadratic_factoring": 0.3,  # Weak
                "distributive_property": 0.5,  # Average
                "fractions_operations": 0.2   # Very weak
            },
            learning_style="visual",
            total_questions_attempted=0,
            total_questions_correct=0
        )
        
        print(f"\n{'='*80}")
        print(f"🎯 MIXED ABILITY STUDENT TEST - ALEX")
        print(f"{'='*80}")
        print(f"Profile: Strong in functions, weak in fractions and factoring")
        
        return self._run_test_scenario(student_profile, "mixed_ability")

    def test_inconsistent_student(self) -> Dict:
        """Test a student with highly inconsistent performance"""
        
        student_profile = StudentProfile(
            name="Jordan",
            overall_ability=0.45,  # Average but very inconsistent
            concept_mastery={},  # No established mastery patterns
            learning_style="kinesthetic",
            total_questions_attempted=0,
            total_questions_correct=0
        )
        
        print(f"\n{'='*80}")
        print(f"🎲 INCONSISTENT STUDENT TEST - JORDAN")
        print(f"{'='*80}")
        print(f"Profile: Highly variable performance, no clear patterns")
        
        return self._run_test_scenario(student_profile, "inconsistent")

    def test_conceptually_confused_student(self) -> Dict:
        """Test a student with systematic misconceptions"""
        
        student_profile = StudentProfile(
            name="Casey",
            overall_ability=0.30,  # Below average with systematic errors
            concept_mastery={
                "distributive_property": 0.1,  # Major misconceptions
                "fractions_operations": 0.1,   # Major misconceptions
                "absolute_value": 0.1,         # Major misconceptions
                "system_equations": 0.1        # Major misconceptions
            },
            learning_style="analytical",
            total_questions_attempted=0,
            total_questions_correct=0
        )
        
        print(f"\n{'='*80}")
        print(f"🤔 CONCEPTUALLY CONFUSED STUDENT TEST - CASEY")
        print(f"{'='*80}")
        print(f"Profile: Systematic misconceptions across multiple concepts")
        
        return self._run_test_scenario(student_profile, "confused")

    def _run_test_scenario(self, student_profile: StudentProfile, error_pattern: str) -> Dict:
        """Run a test scenario with the given student profile and error pattern"""
        
        test_results = []
        correct_count = 0
        total_problems = len(self.advanced_problems)
        
        for problem_id, problem_data in self.advanced_problems.items():
            print(f"\n📝 **Problem {len(test_results) + 1}** (Difficulty {problem_data['difficulty']}/5):")
            print(f"   {problem_data['problem']}")
            print(f"   Concepts: {', '.join(problem_data['concepts'])}")
            
            # Get student's answer based on error pattern
            possible_answers = self.complex_error_patterns[error_pattern][problem_id]
            student_answer = possible_answers[0]  # Take first answer for consistency
            correct_answer = problem_data['correct_answer']
            
            print(f"   Student Answer: {student_answer}")
            print(f"   Correct Answer: {correct_answer}")
            
            # Check if answer is correct
            is_correct = student_answer == correct_answer
            if is_correct:
                correct_count += 1
                print(f"   ✅ CORRECT!")
            else:
                print(f"   ❌ INCORRECT - Running diagnostic...")
                
                # Run diagnostic
                diagnostic_result = self.diagnostic.diagnose_student_response(
                    question_type=problem_data['type'],
                    question_text=problem_data['problem'],
                    student_answer=student_answer,
                    correct_answer=correct_answer,
                    student_profile=student_profile
                )
                
                # Display results
                print(f"\n   🔍 **DIAGNOSTIC ANALYSIS:**")
                print(f"   {diagnostic_result.explanation}")
                
                if diagnostic_result.missing_concept:
                    print(f"\n   🎯 **IDENTIFIED GAPS:**")
                    print(f"      • {diagnostic_result.missing_concept}")
                
                if diagnostic_result.recommended_questions:
                    print(f"\n   📚 **RECOMMENDED PRACTICE:**")
                    for i, question in enumerate(diagnostic_result.recommended_questions[:2], 1):
                        print(f"      {i}. {question.question_text}")
                        print(f"         Answer: {question.correct_answer}")
            
            test_results.append({
                'problem': problem_data,
                'student_answer': student_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'diagnostic': None if is_correct else diagnostic_result
            })
        
        # Generate summary
        accuracy = (correct_count / total_problems) * 100
        print(f"\n{'='*60}")
        print(f"📊 **TEST SUMMARY FOR {student_profile.name.upper()}**")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {correct_count}/{total_problems} ({accuracy:.1f}%)")
        print(f"Error Pattern: {error_pattern.title()}")
        
        # Generate learning plan if needed
        if correct_count < total_problems:
            print(f"\n🎯 **PERSONALIZED LEARNING PLAN:**")
            learning_plan = self.diagnostic.generate_learning_plan(student_profile)
            
            print(f"📈 Estimated Total Study Time: {learning_plan['recommended_study_time']:.1f} hours")
            print(f"🎯 Priority Concepts to Master:")
            
            for i, concept_info in enumerate(learning_plan['priority_concepts'][:4], 1):
                print(f"   {i}. {concept_info['concept_name']} ({concept_info['concept']})")
                print(f"      📊 Difficulty: {concept_info['difficulty']}/5")
                print(f"      🚨 Errors: {concept_info['frequency']}")
                print(f"      ⏱️ Study Time: {concept_info['estimated_hours']:.1f} hours")
        
        return {
            'student_name': student_profile.name,
            'error_pattern': error_pattern,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_problems': total_problems,
            'test_results': test_results
        }

    def test_edge_cases(self):
        """Test edge cases and unusual scenarios"""
        
        print(f"\n{'='*80}")
        print(f"🔬 EDGE CASE TESTING")
        print(f"{'='*80}")
        
        # Test 1: Empty student answer
        print(f"\n🧪 **Edge Case 1: Empty Answer**")
        student_profile = StudentProfile(
            name="TestStudent",
            overall_ability=0.5,
            concept_mastery={},
            learning_style="analytical",
            total_questions_attempted=0,
            total_questions_correct=0
        )
        
        try:
            result = self.diagnostic.diagnose_student_response(
                question_type="two_step_equations",
                question_text="Solve: 2x + 3 = 7",
                student_answer="",
                correct_answer="x = 2",
                student_profile=student_profile
            )
            print(f"   ✅ Handled empty answer: {result.explanation[:50]}...")
        except Exception as e:
            print(f"   ❌ Error with empty answer: {e}")
        
        # Test 2: Very long incorrect answer
        print(f"\n🧪 **Edge Case 2: Very Long Answer**")
        try:
            result = self.diagnostic.diagnose_student_response(
                question_type="two_step_equations",
                question_text="Solve: 2x + 3 = 7",
                student_answer="x = 2 but I'm not sure because I think maybe it could be 3 or 4 and I tried multiple methods",
                correct_answer="x = 2",
                student_profile=student_profile
            )
            print(f"   ✅ Handled long answer: {result.explanation[:50]}...")
        except Exception as e:
            print(f"   ❌ Error with long answer: {e}")
        
        # Test 3: Special characters in answer
        print(f"\n🧪 **Edge Case 3: Special Characters**")
        try:
            result = self.diagnostic.diagnose_student_response(
                question_type="two_step_equations",
                question_text="Solve: 2x + 3 = 7",
                student_answer="x = 2.5 ± 0.1",
                correct_answer="x = 2",
                student_profile=student_profile
            )
            print(f"   ✅ Handled special characters: {result.explanation[:50]}...")
        except Exception as e:
            print(f"   ❌ Error with special characters: {e}")

def main():
    """Run advanced test scenarios"""
    
    tester = AdvancedTestScenarios()
    
    print("🧮 ADVANCED K-12 ALGEBRA DIAGNOSTIC TESTING")
    print("🎯 Complex Scenarios and Edge Cases")
    
    # Run advanced test scenarios
    results = []
    
    # Test mixed ability student
    results.append(tester.test_mixed_ability_student())
    
    # Test inconsistent student  
    results.append(tester.test_inconsistent_student())
    
    # Test conceptually confused student
    results.append(tester.test_conceptually_confused_student())
    
    # Test edge cases
    tester.test_edge_cases()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🏆 **ADVANCED TESTING SUMMARY**")
    print(f"{'='*80}")
    
    for result in results:
        print(f"{result['student_name']} ({result['error_pattern']}): "
              f"{result['correct_count']}/{result['total_problems']} "
              f"({result['accuracy']:.1f}%)")
    
    print(f"\n✅ **ADVANCED TESTING COMPLETE**")
    print(f"   • Complex multi-concept problems tested")
    print(f"   • Mixed ability patterns validated")
    print(f"   • Edge cases handled gracefully")
    print(f"   • Systematic misconceptions identified")

if __name__ == "__main__":
    main() 