"""
Comprehensive test runner for the diagnostic system.
Demonstrates error simulation and concept gap identification.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import random
from typing import List, Dict

from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.testing.algebra_test_bank import AlgebraTestBank
from math_learning.testing.diagnostic_engine import DiagnosticEngine, StudentProfile, StudentResponse
from math_learning.testing.test_question import TestQuestion, ErrorMapping, DifficultyLevel

def load_knowledge_graph() -> KnowledgeGraph:
    """Load the validated knowledge graph."""
    try:
        graph = KnowledgeGraph.load_from_file("math_learning/knowledge_graph/k12_algebra_concepts.json")
        return graph
    except FileNotFoundError:
        print("⚠️  Knowledge graph file not found. Creating minimal graph for testing.")
        # Create a minimal graph for testing if file doesn't exist
        graph = KnowledgeGraph()
        return graph

def demonstrate_individual_question_errors():
    """Demonstrate how individual questions map errors to concepts."""
    print("\n" + "="*80)
    print("INDIVIDUAL QUESTION ERROR ANALYSIS DEMONSTRATION")
    print("="*80)
    
    test_bank = AlgebraTestBank()
    
    # Demonstrate specific error mappings for different questions
    demo_questions = [
        "NS_001",  # Order of operations
        "AT_001",  # Translate expressions
        "LF_001",  # Two-step equations
        "EXP_001", # Exponent rules
    ]
    
    for question_id in demo_questions:
        try:
            question = test_bank.get_question_by_id(question_id)
            print(f"\n📝 QUESTION: {question.question_text}")
            print(f"   Correct Answer: {question.correct_answer}")
            print(f"   Primary Concepts: {question.primary_concepts}")
            
            print(f"\n   🚨 COMMON ERROR PATTERNS:")
            for i, error in enumerate(question.common_errors, 1):
                print(f"      {i}. Wrong Answer: '{error.wrong_answer}'")
                print(f"         → Concept Gap: {error.concept_name} ({error.concept_id})")
                print(f"         → Error Type: {error.error_description}")
                print(f"         → Remediation: {error.remediation_hint}")
                print()
        except ValueError:
            print(f"   Question {question_id} not found in test bank")

def simulate_student_with_specific_errors():
    """Simulate a student making specific types of errors."""
    print("\n" + "="*80)
    print("SIMULATED STUDENT ERROR ANALYSIS")
    print("="*80)
    
    knowledge_graph = load_knowledge_graph()
    test_bank = AlgebraTestBank()
    diagnostic_engine = DiagnosticEngine(knowledge_graph, test_bank)
    
    # Create a custom student response with specific error patterns
    print("\n🎭 SIMULATING STUDENT WITH SPECIFIC ERROR PATTERNS...")
    
    # Manually create responses showing specific concept gaps
    simulated_responses = []
    
    # Student gets order of operations wrong (PEMDAS confusion)
    question_1 = test_bank.get_question_by_id("NS_001")
    response_1 = StudentResponse(
        question_id="NS_001",
        student_answer="14",  # Wrong: added first, then multiplied
        is_correct=False,
        error_mapping=question_1.common_errors[0],  # PEMDAS error
        time_taken_seconds=180,
        confidence_level=2
    )
    simulated_responses.append(response_1)
    
    # Student gets integer addition wrong
    question_2 = test_bank.get_question_by_id("NS_002")
    response_2 = StudentResponse(
        question_id="NS_002",
        student_answer="2",  # Wrong: treated as subtraction
        is_correct=False,
        error_mapping=question_2.common_errors[2],  # Integer addition error
        time_taken_seconds=120,
        confidence_level=1
    )
    simulated_responses.append(response_2)
    
    # Student gets expression translation wrong
    question_3 = test_bank.get_question_by_id("AT_001")
    response_3 = StudentResponse(
        question_id="AT_001",
        student_answer="2x - 5",  # Wrong: used subtraction instead of addition
        is_correct=False,
        error_mapping=question_3.common_errors[1],  # Translation error
        time_taken_seconds=240,
        confidence_level=2
    )
    simulated_responses.append(response_3)
    
    # Student gets equation solving wrong
    question_4 = test_bank.get_question_by_id("LF_001")
    response_4 = StudentResponse(
        question_id="LF_001",
        student_answer="7",  # Wrong: added instead of subtracting
        is_correct=False,
        error_mapping=question_4.common_errors[0],  # Two-step equation error
        time_taken_seconds=300,
        confidence_level=1
    )
    simulated_responses.append(response_4)
    
    # Student gets one question right (to show contrast)
    question_5 = test_bank.get_question_by_id("FN_001")
    response_5 = StudentResponse(
        question_id="FN_001",
        student_answer="11",  # Correct
        is_correct=True,
        error_mapping=None,
        time_taken_seconds=150,
        confidence_level=4
    )
    simulated_responses.append(response_5)
    
    # Analyze the responses
    report = diagnostic_engine.analyze_responses(simulated_responses, "struggling_student_demo")
    
    print(f"\n📊 DETAILED ERROR ANALYSIS:")
    print(f"   Total Questions: {len(simulated_responses)}")
    print(f"   Errors Made: {len([r for r in simulated_responses if not r.is_correct])}")
    
    print(f"\n🔍 SPECIFIC ERRORS DETECTED:")
    for i, response in enumerate(simulated_responses, 1):
        question = test_bank.get_question_by_id(response.question_id)
        print(f"\n   Question {i}: {question.question_text}")
        print(f"   Student Answer: '{response.student_answer}' | Correct: {response.is_correct}")
        
        if response.error_mapping:
            print(f"   🚨 Error Identified:")
            print(f"      → Concept Gap: {response.error_mapping.concept_name}")
            print(f"      → Error Type: {response.error_mapping.error_description}")
            print(f"      → Remediation: {response.error_mapping.remediation_hint}")
    
    # Print the full diagnostic report
    diagnostic_engine.print_diagnostic_report(report)

def run_comprehensive_student_profiles():
    """Run comprehensive diagnostics for different student profiles."""
    print("\n" + "="*80)
    print("COMPREHENSIVE STUDENT PROFILE ANALYSIS")
    print("="*80)
    
    knowledge_graph = load_knowledge_graph()
    test_bank = AlgebraTestBank()
    diagnostic_engine = DiagnosticEngine(knowledge_graph, test_bank)
    
    # Test different student profiles
    profiles_to_test = [
        (StudentProfile.STRONG_STUDENT, "Emma (Strong Student)"),
        (StudentProfile.AVERAGE_STUDENT, "Alex (Average Student)"),
        (StudentProfile.STRUGGLING_STUDENT, "Jordan (Struggling Student)"),
    ]
    
    for profile, student_name in profiles_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING PROFILE: {student_name}")
        print(f"{'='*60}")
        
        # Set random seed for reproducible results in demo
        random.seed(42)
        
        # Run comprehensive diagnostic
        report = diagnostic_engine.run_comprehensive_diagnostic(profile, student_name)
        
        # Print detailed report
        diagnostic_engine.print_diagnostic_report(report)
        
        # Show specific error patterns for struggling students
        if profile == StudentProfile.STRUGGLING_STUDENT:
            print(f"\n🔬 DETAILED ERROR PATTERN ANALYSIS:")
            for gap in report.concept_gaps[:3]:  # Top 3 gaps
                print(f"\n   Concept: {gap.concept_name}")
                print(f"   Severity: {gap.severity:.2f}")
                print(f"   Common Errors:")
                for error in gap.related_errors[:2]:  # Show up to 2 errors
                    print(f"      • {error.error_description}")
                    print(f"        Remediation: {error.remediation_hint}")

def demonstrate_prerequisite_gap_detection():
    """Demonstrate how the system detects prerequisite gaps."""
    print("\n" + "="*80)
    print("PREREQUISITE GAP DETECTION DEMONSTRATION")
    print("="*80)
    
    knowledge_graph = load_knowledge_graph()
    test_bank = AlgebraTestBank()
    
    print(f"\n🔗 KNOWLEDGE GRAPH PREREQUISITE RELATIONSHIPS:")
    
    # Show some key prerequisite relationships
    sample_concepts = ["AT-24", "LF-29", "Q-59"]  # Two-step equations, slope, quadratic formula
    
    for concept_id in sample_concepts:
        if concept_id in knowledge_graph.concepts:
            concept = knowledge_graph.concepts[concept_id]
            prerequisites = knowledge_graph.get_prerequisites(concept_id)
            
            print(f"\n   📚 {concept.name} ({concept_id})")
            print(f"      Prerequisites: {prerequisites}")
            
            # Show questions that test this concept
            questions = test_bank.get_questions_for_concept(concept_id)
            if questions:
                print(f"      Test Questions: {[q.id for q in questions]}")
    
    print(f"\n🎯 PREREQUISITE GAP SIMULATION:")
    print(f"   If a student struggles with 'Two-Step Linear Equations',")
    print(f"   the system checks if they also have gaps in prerequisites like:")
    print(f"   • One-Step Equations")
    print(f"   • Integer Addition/Subtraction")
    print(f"   • Order of Operations")
    print(f"\n   This helps identify the ROOT CAUSE of learning difficulties!")

def analyze_test_bank_coverage():
    """Analyze the test bank coverage and error mapping completeness."""
    print("\n" + "="*80)
    print("TEST BANK COVERAGE ANALYSIS")
    print("="*80)
    
    test_bank = AlgebraTestBank()
    all_questions = test_bank.get_all_questions()
    
    print(f"\n📊 TEST BANK STATISTICS:")
    print(f"   Total Questions: {len(all_questions)}")
    
    # Analyze by difficulty
    difficulty_counts = {}
    for question in all_questions:
        difficulty = question.difficulty
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    print(f"\n   Questions by Difficulty:")
    for difficulty, count in difficulty_counts.items():
        print(f"      {difficulty.name}: {count} questions")
    
    # Analyze error mapping coverage
    total_errors = sum(len(q.common_errors) for q in all_questions)
    questions_with_errors = sum(1 for q in all_questions if q.common_errors)
    
    print(f"\n   Error Mapping Coverage:")
    print(f"      Questions with Error Mappings: {questions_with_errors}/{len(all_questions)}")
    print(f"      Total Error Patterns Defined: {total_errors}")
    print(f"      Average Errors per Question: {total_errors/len(all_questions):.1f}")
    
    # Show concept coverage
    all_concepts = set()
    for question in all_questions:
        all_concepts.update(question.get_concept_dependencies())
    
    print(f"\n   Concept Coverage:")
    print(f"      Unique Concepts Tested: {len(all_concepts)}")
    print(f"      Concepts: {sorted(list(all_concepts))}")

def main():
    """Run all diagnostic demonstrations."""
    print("🧮 ALGEBRA DIAGNOSTIC TESTING SYSTEM")
    print("Demonstrating error simulation and concept gap identification")
    
    # Run all demonstrations
    demonstrate_individual_question_errors()
    simulate_student_with_specific_errors()
    run_comprehensive_student_profiles()
    demonstrate_prerequisite_gap_detection()
    analyze_test_bank_coverage()
    
    print(f"\n{'='*80}")
    print("🎉 DIAGNOSTIC SYSTEM DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("✅ Real algebra test questions with authentic error patterns")
    print("✅ Automatic mapping of student errors to concept gaps")
    print("✅ Prerequisite relationship analysis")
    print("✅ Personalized diagnostic reports with remediation suggestions")
    print("✅ Multiple student profile simulations")
    print("✅ Comprehensive error pattern detection")
    
    print(f"\nThis system can be used for:")
    print("• Identifying specific concept misunderstandings")
    print("• Generating personalized study recommendations")
    print("• Tracking student progress over time")
    print("• Curriculum gap analysis")
    print("• Teacher professional development")

if __name__ == "__main__":
    main() 