#!/usr/bin/env python3
"""
Demo Test Case - K-12 Algebra Diagnostic System
Shows a complete diagnostic workflow with a real algebra problem
"""

from comprehensive_diagnostic import ComprehensiveDiagnostic, StudentProfile

def run_demo_test_case():
    """Demonstrate the diagnostic system with a real test case"""
    
    # Initialize the diagnostic system
    print('🔧 Initializing K-12 Algebra Diagnostic System...')
    diagnostic = ComprehensiveDiagnostic()
    print(f'✅ Knowledge Graph loaded: {len(diagnostic.knowledge_graph.concepts)} concepts')
    print()
    
    print('=' * 60)
    print('🧮 K-12 ALGEBRA DIAGNOSTIC TEST CASE DEMO')
    print('=' * 60)
    print()

    # Test Problem - Two-step equation
    problem = {
        'question_text': 'Solve for x: 3x + 7 = 22',
        'question_type': 'two_step_equations',
        'correct_answer': 'x = 5',
        'concepts': ['two_step_equations', 'integer_operations', 'combine_like_terms']
    }

    print('📝 **TEST PROBLEM:**')
    print(f'   Question: {problem["question_text"]}')
    print(f'   Type: {problem["question_type"]}')
    print(f'   Correct Answer: {problem["correct_answer"]}')
    print(f'   Required Concepts: {problem["concepts"]}')
    print()

    # Student Profile - Average ability student
    student = StudentProfile(
        name='Demo Student',
        overall_ability=0.6,  # 60% ability - average student
        concept_mastery={},
        learning_style='analytical',
        total_questions_attempted=0,
        total_questions_correct=0
    )

    print('👤 **STUDENT PROFILE:**')
    print(f'   Student Name: {student.name}')
    print(f'   Ability Level: {student.overall_ability * 100}% (Average)')
    print(f'   Learning Style: {student.learning_style}')
    print()

    # Test Case 1: Correct Answer
    print('🎯 **TEST CASE 1: Correct Answer**')
    student_answer_1 = 'x = 5'
    print(f'   Student Answer: {student_answer_1}')

    result_1 = diagnostic.diagnose_student_response(
        question_text=problem['question_text'],
        question_type=problem['question_type'],
        student_answer=student_answer_1,
        correct_answer=problem['correct_answer'],
        student_profile=student
    )

    if result_1.is_correct:
        print('   ✅ CORRECT!')
        print(f'   📊 Confidence: {result_1.confidence_score:.2f}')
    else:
        print('   ❌ INCORRECT')
        print(f'   🔍 Missing Concept: {result_1.missing_concept if result_1.missing_concept else "None identified"}')
        print(f'   📊 Confidence: {result_1.confidence_score:.2f}')

    print()

    # Test Case 2: Common Error - Adding instead of subtracting
    print('🎯 **TEST CASE 2: Common Error (Adding instead of subtracting)**')
    student_answer_2 = 'x = 29'  # Student added 7 instead of subtracting: 3x = 22 + 7 = 29, x = 29/3 ≈ 10, but student got 29
    print(f'   Student Answer: {student_answer_2}')
    print(f'   🧠 Error Analysis: Student likely added 7 to both sides instead of subtracting')

    result_2 = diagnostic.diagnose_student_response(
        question_text=problem['question_text'],
        question_type=problem['question_type'],
        student_answer=student_answer_2,
        correct_answer=problem['correct_answer'],
        student_profile=student
    )

    if result_2.is_correct:
        print('   ✅ CORRECT!')
    else:
        print('   ❌ INCORRECT')
        if result_2.failure_pattern:
            print(f'   🚨 Error Pattern: {result_2.failure_pattern.error_type}')
        if result_2.missing_concept:
            print(f'   🎯 Missing Concept: {result_2.missing_concept}')
        print(f'   📊 Confidence: {result_2.confidence_score:.2f}')
        print(f'   💡 Explanation: {result_2.explanation}')
        
        if result_2.recommended_questions:
            print('   📚 Recommended Practice Questions:')
            for i, q in enumerate(result_2.recommended_questions[:2], 1):
                print(f'      {i}. {q.question_text}')
                print(f'         Answer: {q.correct_answer}')
                if hasattr(q, 'explanation') and q.explanation:
                    print(f'         Hint: {q.explanation}')

    print()

    # Test Case 3: Different type of error - Wrong operation
    print('🎯 **TEST CASE 3: Wrong Operation Error**')
    student_answer_3 = 'x = 15'  # Student might have done 22 - 7 = 15, then x = 15
    print(f'   Student Answer: {student_answer_3}')
    print(f'   🧠 Error Analysis: Student may have subtracted 7 from 22 without dividing by 3')

    result_3 = diagnostic.diagnose_student_response(
        question_text=problem['question_text'],
        question_type=problem['question_type'],
        student_answer=student_answer_3,
        correct_answer=problem['correct_answer'],
        student_profile=student
    )

    if result_3.is_correct:
        print('   ✅ CORRECT!')
    else:
        print('   ❌ INCORRECT')
        if result_3.failure_pattern:
            print(f'   🚨 Error Pattern: {result_3.failure_pattern.error_type}')
        if result_3.missing_concept:
            print(f'   🎯 Missing Concept: {result_3.missing_concept}')
        print(f'   📊 Confidence: {result_3.confidence_score:.2f}')
        print(f'   💡 Explanation: {result_3.explanation}')

    print()
    print('=' * 60)
    print('📊 **DIAGNOSTIC SUMMARY**')
    print('=' * 60)
    print(f'✅ Correct responses detected: {sum([result_1.is_correct, result_2.is_correct, result_3.is_correct])}/3')
    
    # Count unique missing concepts
    missing_concepts = []
    if result_2.missing_concept:
        missing_concepts.append(result_2.missing_concept)
    if result_3.missing_concept:
        missing_concepts.append(result_3.missing_concept)
    unique_concepts = len(set(missing_concepts))
    
    print(f'🎯 Concept gaps identified: {unique_concepts} unique concepts')
    print(f'📚 Practice questions recommended: {len(result_2.recommended_questions) + len(result_3.recommended_questions)}')
    print()
    print('✨ Demo test case complete!')
    print('=' * 60)

if __name__ == "__main__":
    run_demo_test_case() 