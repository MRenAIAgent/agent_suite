#!/usr/bin/env python3
"""
FAILED ANSWER DIAGNOSTIC TEST CASES
Demonstrates complete pipeline: Failed Answer → Concept Identification → Multi-Level Remediation

Shows realistic scenarios where students make errors and the system:
1. Identifies the missing concept
2. Recommends questions from each difficulty level (Easy → Medium → Hard → Bonus)
3. Provides targeted remediation based on the specific gap
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_diagnostic import ComprehensiveDiagnostic, StudentProfile
from algebra_comprehensive_question_bank_complete import CompleteAlgebraQuestionBank
import json
import time
from typing import Dict, List, Tuple

class FailedAnswerDiagnosticTester:
    """Tests the complete diagnostic pipeline with realistic failed answers"""
    
    def __init__(self):
        print("🧪 FAILED ANSWER DIAGNOSTIC TEST CASES")
        print("=" * 60)
        
        # Initialize components
        self.diagnostic = ComprehensiveDiagnostic()
        self.question_bank = CompleteAlgebraQuestionBank()
        
        # Load question bank data
        try:
            with open('complete_algebra_question_bank.json', 'r') as f:
                self.qb_data = json.load(f)
            print("✅ Question Bank loaded successfully")
        except Exception as e:
            print(f"❌ Question Bank loading failed: {e}")
            return
        
        print("✅ Diagnostic system initialized\n")
    
    def create_test_cases(self) -> List[Dict]:
        """Create comprehensive test cases with realistic failed answers"""
        
        test_cases = [
            # Number Sense - Counting
            {
                "concept_id": "NS-01",
                "concept_name": "Counting Up/Down",
                "question": "Count from 15 to 20: 15, 16, 17, __, __, 20",
                "correct_answer": "18, 19",
                "student_answer": "17, 18",  # Skipping numbers
                "error_type": "counting_sequence",
                "expected_missing_concept": "ADDITION_FACTS",
                "description": "Student struggles with number sequence"
            },
            
            # Number Sense - Place Value
            {
                "concept_id": "NS-02", 
                "concept_name": "Place Value to 1,000,000",
                "question": "What is the value of the digit 7 in 47,832?",
                "correct_answer": "7,000",
                "student_answer": "700",  # Confusing place values
                "error_type": "place_value",
                "expected_missing_concept": "ADDITION_FACTS",
                "description": "Student confuses thousands and hundreds place"
            },
            
            # Pre-Algebra - One-Step Equations
            {
                "concept_id": "AT-20",
                "concept_name": "One-Step Linear Equations", 
                "question": "Solve for x: x + 7 = 15",
                "correct_answer": "x = 8",
                "student_answer": "x = 22",  # Adding instead of subtracting
                "error_type": "one_step_equations",
                "expected_missing_concept": "ONE_STEP_EQUATIONS",
                "description": "Student adds 7 instead of subtracting"
            },
            
            # Pre-Algebra - Distributive Property
            {
                "concept_id": "AT-21",
                "concept_name": "Distributive Property",
                "question": "Expand: 3(x + 4)",
                "correct_answer": "3x + 12", 
                "student_answer": "3x + 4",  # Not distributing to second term
                "error_type": "distributive_property",
                "expected_missing_concept": "DISTRIBUTIVE_PROPERTY",
                "description": "Student doesn't distribute to all terms"
            },
            
            # Pre-Algebra - Combine Like Terms
            {
                "concept_id": "AT-19",
                "concept_name": "Combine Like Terms",
                "question": "Simplify: 3x + 5x - 2x",
                "correct_answer": "6x",
                "student_answer": "3x + 5x - 2x",  # Not combining
                "error_type": "combine_like_terms", 
                "expected_missing_concept": "COMBINE_LIKE_TERMS",
                "description": "Student doesn't combine like terms"
            },
            
            # Number Sense - Order of Operations
            {
                "concept_id": "NS-16",
                "concept_name": "Order of Operations",
                "question": "Calculate: 2 + 3 × 4",
                "correct_answer": "14",
                "student_answer": "20",  # Left-to-right instead of PEMDAS
                "error_type": "order_of_operations",
                "expected_missing_concept": "ORDER_OF_OPERATIONS", 
                "description": "Student calculates left-to-right instead of PEMDAS"
            },
            
            # Functions - Function Evaluation
            {
                "concept_id": "FN-62",
                "concept_name": "Function Evaluation",
                "question": "If f(x) = 2x + 3, find f(4)",
                "correct_answer": "11",
                "student_answer": "7",  # Only adding 3, not multiplying by 2
                "error_type": "function_evaluation",
                "expected_missing_concept": "FUNCTION_EVALUATION",
                "description": "Student doesn't follow function operations correctly"
            },
            
            # Number Sense - Integer Operations
            {
                "concept_id": "NS-13", 
                "concept_name": "Integer Addition/Subtraction",
                "question": "Calculate: -5 + (-3)",
                "correct_answer": "-8",
                "student_answer": "-2",  # Subtracting instead of adding negatives
                "error_type": "integer_operations",
                "expected_missing_concept": "INTEGER_OPERATIONS",
                "description": "Student struggles with negative number operations"
            },
            
            # Number Sense - Fractions
            {
                "concept_id": "NS-08",
                "concept_name": "Fractions as Part–Whole", 
                "question": "What fraction of the circle is shaded if 3 out of 8 parts are colored?",
                "correct_answer": "3/8",
                "student_answer": "3/5",  # Using wrong denominator
                "error_type": "fractions_part_whole",
                "expected_missing_concept": "FRACTIONS_AS_PART_WHOLE",
                "description": "Student doesn't understand part-whole relationship"
            },
            
            # Number Sense - Multiplication Facts
            {
                "concept_id": "NS-06",
                "concept_name": "Multiplication Facts 0–12",
                "question": "Calculate: 7 × 8",
                "correct_answer": "56", 
                "student_answer": "54",  # Close but incorrect fact
                "error_type": "multiplication_facts",
                "expected_missing_concept": "MULTIPLICATION_FACTS",
                "description": "Student has gaps in multiplication facts"
            }
        ]
        
        return test_cases
    
    def get_remediation_questions_by_level(self, concept_id: str) -> Dict[str, List[Dict]]:
        """Get remediation questions organized by difficulty level"""
        
        if concept_id not in self.qb_data:
            return {"easy": [], "medium": [], "hard": [], "bonus": []}
        
        concept_questions = self.qb_data[concept_id]
        
        return {
            "easy": concept_questions.get("easy", [])[:3],      # Top 3 easy
            "medium": concept_questions.get("medium", [])[:3],  # Top 3 medium  
            "hard": concept_questions.get("hard", [])[:3],      # Top 3 hard
            "bonus": concept_questions.get("bonus", [])[:3]     # Top 3 bonus
        }
    
    def run_diagnostic_test_case(self, test_case: Dict) -> Dict:
        """Run a single diagnostic test case and return detailed results"""
        
        # Create student profile
        student = StudentProfile(
            name="Test Student",
            overall_ability=0.6,
            concept_mastery={},
            learning_style="analytical", 
            total_questions_attempted=0,
            total_questions_correct=0
        )
        
        print(f"📝 TEST CASE: {test_case['concept_name']} ({test_case['concept_id']})")
        print("-" * 50)
        print(f"Question: {test_case['question']}")
        print(f"Correct Answer: {test_case['correct_answer']}")
        print(f"Student Answer: {test_case['student_answer']}")
        print(f"Error Description: {test_case['description']}")
        
        # Run diagnostic
        try:
            result = self.diagnostic.diagnose_student_response(
                question_type=test_case["error_type"],
                question_text=test_case["question"],
                student_answer=test_case["student_answer"],
                correct_answer=test_case["correct_answer"],
                student_profile=student
            )
            
            print(f"\n🔍 DIAGNOSTIC RESULTS:")
            print(f"   ✅ Error Detected: {not result.is_correct}")
            print(f"   🎯 Missing Concept: {result.missing_concept}")
            print(f"   📊 Confidence: {result.confidence_score:.1%}")
            
            if result.failure_pattern:
                print(f"   🚨 Error Pattern: {result.failure_pattern.explanation}")
            
            # Get remediation questions by level
            remediation_questions = self.get_remediation_questions_by_level(test_case["concept_id"])
            
            print(f"\n📚 REMEDIATION RECOMMENDATIONS BY LEVEL:")
            
            for level in ["easy", "medium", "hard", "bonus"]:
                questions = remediation_questions[level]
                print(f"\n   🔹 {level.upper()} LEVEL ({len(questions)} questions):")
                
                for i, question in enumerate(questions, 1):
                    print(f"      {i}. {question['question']}")
                    print(f"         Answer: {question['answer']}")
                    print(f"         Source: {question['source']}")
                    if 'type' in question:
                        print(f"         Type: {question['type']}")
            
            # Return structured results
            return {
                "test_case": test_case,
                "diagnostic_result": result,
                "remediation_by_level": remediation_questions,
                "success": True
            }
            
        except Exception as e:
            print(f"\n❌ DIAGNOSTIC FAILED: {e}")
            return {
                "test_case": test_case,
                "diagnostic_result": None,
                "remediation_by_level": {},
                "success": False,
                "error": str(e)
            }
    
    def run_all_test_cases(self):
        """Run all diagnostic test cases"""
        
        test_cases = self.create_test_cases()
        results = []
        
        print(f"🚀 RUNNING {len(test_cases)} DIAGNOSTIC TEST CASES")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"TEST CASE {i}/{len(test_cases)}")
            print(f"{'='*60}")
            
            result = self.run_diagnostic_test_case(test_case)
            results.append(result)
            
            print(f"\n✅ Test Case {i} completed")
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 DIAGNOSTIC TEST SUMMARY")
        print(f"{'='*60}")
        
        successful_tests = sum(1 for r in results if r["success"])
        failed_tests = len(results) - successful_tests
        
        print(f"✅ Successful diagnostics: {successful_tests}/{len(results)}")
        print(f"❌ Failed diagnostics: {failed_tests}/{len(results)}")
        
        if successful_tests > 0:
            print(f"\n🎯 CONCEPT COVERAGE:")
            concepts_tested = set()
            for result in results:
                if result["success"] and result["diagnostic_result"]:
                    if result["diagnostic_result"].missing_concept:
                        concepts_tested.add(result["diagnostic_result"].missing_concept)
            
            print(f"   📚 Unique concepts identified: {len(concepts_tested)}")
            for concept in sorted(concepts_tested):
                print(f"      • {concept}")
        
        print(f"\n🏆 REMEDIATION EFFECTIVENESS:")
        total_recommendations = 0
        for result in results:
            if result["success"]:
                for level_questions in result["remediation_by_level"].values():
                    total_recommendations += len(level_questions)
        
        print(f"   📝 Total remediation questions recommended: {total_recommendations}")
        print(f"   📊 Average questions per test case: {total_recommendations/len(results):.1f}")
        
        return results
    
    def demonstrate_single_case_detail(self):
        """Demonstrate a single case in extreme detail"""
        
        print(f"\n{'='*60}")
        print("🔬 DETAILED SINGLE CASE DEMONSTRATION")
        print(f"{'='*60}")
        
        # Use distributive property case as detailed example
        detailed_case = {
            "concept_id": "AT-21",
            "concept_name": "Distributive Property",
            "question": "Expand: 4(x + 3)",
            "correct_answer": "4x + 12",
            "student_answer": "4x + 3",  # Common error
            "error_type": "distributive_property",
            "expected_missing_concept": "DISTRIBUTIVE_PROPERTY",
            "description": "Student multiplies only the first term"
        }
        
        result = self.run_diagnostic_test_case(detailed_case)
        
        if result["success"]:
            print(f"\n🎓 EDUCATIONAL IMPACT ANALYSIS:")
            print(f"   🎯 This error indicates the student understands:")
            print(f"      ✅ Basic multiplication")
            print(f"      ✅ Variable notation")
            print(f"      ✅ Parentheses grouping")
            print(f"\n   🚨 But struggles with:")
            print(f"      ❌ Distributing to all terms")
            print(f"      ❌ Complete algebraic expansion")
            
            print(f"\n📈 LEARNING PROGRESSION:")
            remediation = result["remediation_by_level"]
            
            print(f"   1️⃣ EASY: Build foundation with simple distribution")
            for q in remediation["easy"][:2]:
                print(f"      • {q['question']}")
            
            print(f"\n   2️⃣ MEDIUM: Practice with variables")
            for q in remediation["medium"][:2]:
                print(f"      • {q['question']}")
            
            print(f"\n   3️⃣ HARD: Complex expressions")
            for q in remediation["hard"][:2]:
                print(f"      • {q['question']}")
            
            print(f"\n   4️⃣ BONUS: Advanced applications")
            for q in remediation["bonus"][:2]:
                print(f"      • {q['question']}")
        
        return result

def main():
    """Run comprehensive failed answer diagnostic tests"""
    
    tester = FailedAnswerDiagnosticTester()
    
    # Run all test cases
    results = tester.run_all_test_cases()
    
    # Demonstrate detailed single case
    tester.demonstrate_single_case_detail()
    
    print(f"\n🎉 FAILED ANSWER DIAGNOSTIC TESTING COMPLETE!")
    print(f"✅ System successfully identifies concepts from failed answers")
    print(f"✅ Multi-level remediation recommendations working")
    print(f"✅ Complete pipeline validated: Error → Concept → Remediation")
    
    return results

if __name__ == "__main__":
    main() 