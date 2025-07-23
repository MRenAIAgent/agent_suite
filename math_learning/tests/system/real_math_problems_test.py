#!/usr/bin/env python3
"""
Real Math Problems Test - Using authentic educational problems
Simulates realistic student failures on problems from Khan Academy, RSM, Kumon styles
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_diagnostic import ComprehensiveDiagnostic, StudentProfile
from typing import Dict, List, Tuple

class RealMathProblemsTest:
    """Test system using real educational math problems"""
    
    def __init__(self):
        self.diagnostic = ComprehensiveDiagnostic()
        
        # Real math problems from educational sources
        self.real_problems = {
            # Khan Academy style problems
            "khan_two_step_1": {
                "problem": "Solve for x: 3x + 7 = 22",
                "correct_answer": "x = 5",
                "type": "two_step_equations",
                "source": "Khan Academy style",
                "difficulty": 2
            },
            "khan_distributive_1": {
                "problem": "Expand and simplify: 2(x + 3) + 4x",
                "correct_answer": "6x + 6",
                "type": "distributive_property", 
                "source": "Khan Academy style",
                "difficulty": 2
            },
            "khan_word_problem_1": {
                "problem": "Sarah has 3 times as many stickers as Tom. If Sarah has 15 stickers, how many does Tom have?",
                "correct_answer": "5 stickers",
                "type": "two_step_equations",
                "source": "Khan Academy style",
                "difficulty": 2
            },
            
            # RSM style problems (more challenging)
            "rsm_multi_step_1": {
                "problem": "Solve: 2(3x - 4) + 5 = 3(x + 1) - 2",
                "correct_answer": "x = 2",
                "type": "distributive_property",
                "source": "RSM style",
                "difficulty": 3
            },
            "rsm_fraction_1": {
                "problem": "Solve: (2x + 1)/3 = (x - 2)/2 + 1",
                "correct_answer": "x = 7",
                "type": "two_step_equations",
                "source": "RSM style", 
                "difficulty": 4
            },
            
            # Kumon style problems (systematic practice)
            "kumon_basic_1": {
                "problem": "Calculate: 3 + 4 × 2 - 1",
                "correct_answer": "10",
                "type": "order_of_operations",
                "source": "Kumon style",
                "difficulty": 1
            },
            "kumon_basic_2": {
                "problem": "Simplify: 5x + 3x - 2x",
                "correct_answer": "6x",
                "type": "distributive_property",
                "source": "Kumon style",
                "difficulty": 1
            },
            "kumon_integers_1": {
                "problem": "Calculate: (-3) + (-5) × 2",
                "correct_answer": "-13",
                "type": "order_of_operations",
                "source": "Kumon style",
                "difficulty": 2
            },
            
            # Common Core style word problems
            "common_core_1": {
                "problem": "A rectangle has a length that is 3 more than twice its width. If the perimeter is 30 units, find the width.",
                "correct_answer": "4 units",
                "type": "two_step_equations",
                "source": "Common Core style",
                "difficulty": 3
            },
            "common_core_2": {
                "problem": "The cost of renting a car is $25 per day plus $0.15 per mile. If the total cost for one day is $43, how many miles were driven?",
                "correct_answer": "120 miles",
                "type": "two_step_equations",
                "source": "Common Core style",
                "difficulty": 3
            }
        }
        
        # Realistic student error patterns for each problem
        self.student_errors = {
            "khan_two_step_1": {
                "struggling": ["x = 29", "x = 15", "x = -15"],  # Common errors: adding instead of subtracting, wrong operations
                "average": ["x = 5", "x = 4"],  # Mostly correct with occasional errors
                "advanced": ["x = 5"]  # Always correct
            },
            "khan_distributive_1": {
                "struggling": ["2x + 6 + 4x", "2x + 3 + 4x", "6x + 3"],  # Didn't combine like terms, forgot to distribute
                "average": ["6x + 6", "6x + 5"],  # Mostly correct
                "advanced": ["6x + 6"]  # Always correct
            },
            "khan_word_problem_1": {
                "struggling": ["45 stickers", "3 stickers", "15 stickers"],  # Misunderstood the relationship
                "average": ["5 stickers", "6 stickers"],  # Close to correct
                "advanced": ["5 stickers"]  # Correct
            },
            "rsm_multi_step_1": {
                "struggling": ["x = 6", "x = 0", "x = -2"],  # Distribution and combining errors
                "average": ["x = 2", "x = 1"],  # Close to correct
                "advanced": ["x = 2"]  # Correct
            },
            "rsm_fraction_1": {
                "struggling": ["x = 3", "x = 1", "x = 14"],  # Fraction operation errors
                "average": ["x = 7", "x = 6"],  # Close to correct
                "advanced": ["x = 7"]  # Correct
            },
            "kumon_basic_1": {
                "struggling": ["8", "12", "14"],  # Order of operations errors
                "average": ["10", "9"],  # Mostly correct
                "advanced": ["10"]  # Always correct
            },
            "kumon_basic_2": {
                "struggling": ["8x", "5x + 3x - 2x", "10x"],  # Didn't combine or wrong combination
                "average": ["6x", "5x"],  # Mostly correct
                "advanced": ["6x"]  # Always correct
            },
            "kumon_integers_1": {
                "struggling": ["8", "-8", "2"],  # Sign errors, order of operations
                "average": ["-13", "-12"],  # Mostly correct
                "advanced": ["-13"]  # Always correct
            },
            "common_core_1": {
                "struggling": ["6 units", "2 units", "12 units"],  # Setup equation errors
                "average": ["4 units", "3 units"],  # Close to correct
                "advanced": ["4 units"]  # Correct
            },
            "common_core_2": {
                "struggling": ["60 miles", "180 miles", "72 miles"],  # Equation setup errors
                "average": ["120 miles", "115 miles"],  # Close to correct
                "advanced": ["120 miles"]  # Correct
            }
        }

    def simulate_student_test(self, student_name: str, ability_level: str) -> Dict:
        """Simulate a student taking the real math problems test"""
        
        # Create student profile based on ability level
        ability_scores = {
            "struggling": 0.25,
            "average": 0.55, 
            "advanced": 0.85
        }
        
        student_profile = StudentProfile(
            name=student_name,
            overall_ability=ability_scores[ability_level],
            concept_mastery={},
            learning_style="analytical",  # Default learning style
            total_questions_attempted=0,
            total_questions_correct=0
        )
        
        print(f"\n{'='*80}")
        print(f"🎓 REAL MATH PROBLEMS TEST - {student_name.upper()} ({ability_level.upper()} STUDENT)")
        print(f"{'='*80}")
        
        test_results = []
        total_problems = len(self.real_problems)
        correct_count = 0
        
        for problem_id, problem_data in self.real_problems.items():
            print(f"\n📝 **Problem {len(test_results) + 1}** ({problem_data['source']}):")
            print(f"   {problem_data['problem']}")
            print(f"   Difficulty: {problem_data['difficulty']}/5")
            
            # Get student's answer based on their ability level
            possible_answers = self.student_errors[problem_id][ability_level]
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
                
                # Run diagnostic on the failed answer
                diagnostic_result = self.diagnostic.diagnose_student_response(
                    question_type=problem_data['type'],
                    question_text=problem_data['problem'],
                    student_answer=student_answer,
                    correct_answer=correct_answer,
                    student_profile=student_profile
                )
                
                # Display diagnostic results
                print(f"\n   🔍 **DIAGNOSTIC ANALYSIS:**")
                print(f"   {diagnostic_result.explanation}")
                
                if diagnostic_result.recommended_questions:
                    print(f"\n   📚 **RECOMMENDED PRACTICE:**")
                    for i, question in enumerate(diagnostic_result.recommended_questions[:2], 1):
                        print(f"      {i}. {question.question_text}")
                        print(f"         Answer: {question.correct_answer}")
                        print(f"         Hint: {question.hints[0] if question.hints else 'No hint available'}")
            
            test_results.append({
                'problem': problem_data,
                'student_answer': student_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'diagnostic': None if is_correct else diagnostic_result
            })
        
        # Generate overall test summary
        accuracy = (correct_count / total_problems) * 100
        print(f"\n{'='*80}")
        print(f"📊 **TEST SUMMARY FOR {student_name.upper()}**")
        print(f"{'='*80}")
        print(f"Overall Accuracy: {correct_count}/{total_problems} ({accuracy:.1f}%)")
        print(f"Student Level: {ability_level.title()}")
        
        # Generate personalized learning plan
        if correct_count < total_problems:
            print(f"\n🎯 **PERSONALIZED LEARNING PLAN:**")
            learning_plan = self.diagnostic.generate_learning_plan(student_profile)
            
            print(f"📈 Estimated Total Study Time: {learning_plan['recommended_study_time']:.1f} hours")
            print(f"🎯 Priority Concepts to Master:")
            
            for i, concept_info in enumerate(learning_plan['priority_concepts'][:3], 1):
                print(f"   {i}. {concept_info['concept_name']} ({concept_info['concept']})")
                print(f"      📖 {concept_info['description']}")
                print(f"      📚 Category: {concept_info['category']}")
                print(f"      📊 Difficulty: {concept_info['difficulty']}/5")
                print(f"      🚨 Errors: {concept_info['frequency']}")
                print(f"      ⏱️ Study Time: {concept_info['estimated_hours']:.1f} hours")
                print(f"      📝 Practice Questions: {concept_info['practice_questions']}")
                if concept_info['examples']:
                    print(f"      💡 Examples: {', '.join(concept_info['examples'][:2])}")
                print()
        else:
            print(f"\n🎉 **EXCELLENT WORK!** {student_name} answered all problems correctly!")
            print(f"   Ready for more advanced topics!")
        
        return {
            'student_name': student_name,
            'ability_level': ability_level,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_problems': total_problems,
            'test_results': test_results
        }

def main():
    """Run comprehensive test with real math problems"""
    
    tester = RealMathProblemsTest()
    
    print("🧮 COMPREHENSIVE K-12 ALGEBRA DIAGNOSTIC SYSTEM")
    print("📚 Testing with Real Educational Math Problems")
    print("🎯 Sources: Khan Academy, RSM, Kumon, Common Core styles")
    
    # Test different student ability levels
    students = [
        ("Emma", "advanced"),    # Strong student
        ("Jake", "struggling"),  # Struggling student  
        ("Maria", "average")     # Average student
    ]
    
    all_results = []
    
    for student_name, ability_level in students:
        result = tester.simulate_student_test(student_name, ability_level)
        all_results.append(result)
    
    # Final summary across all students
    print(f"\n{'='*80}")
    print(f"🏆 **FINAL SUMMARY - ALL STUDENTS**")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"{result['student_name']} ({result['ability_level']}): "
              f"{result['correct_count']}/{result['total_problems']} "
              f"({result['accuracy']:.1f}%)")
    
    print(f"\n✅ **SYSTEM VALIDATION COMPLETE**")
    print(f"   • Real educational problems tested")
    print(f"   • Realistic student errors simulated") 
    print(f"   • Concept gaps identified from knowledge graph")
    print(f"   • Personalized remediation provided")
    print(f"   • Learning plans generated with time estimates")

if __name__ == "__main__":
    main() 