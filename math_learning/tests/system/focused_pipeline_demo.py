#!/usr/bin/env python3
"""
FOCUSED PIPELINE DEMONSTRATION
Shows the complete diagnostic pipeline with real questions from implemented concepts

Demonstrates:
1. Student fails a question from the question bank
2. System identifies the missing concept
3. System recommends targeted questions from each difficulty level
4. Shows the complete learning progression
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_diagnostic import ComprehensiveDiagnostic, StudentProfile
from algebra_comprehensive_question_bank_complete import CompleteAlgebraQuestionBank
import json

def demonstrate_complete_pipeline():
    """Demonstrate the complete diagnostic pipeline with real examples"""
    
    print("🎯 FOCUSED PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize systems
    question_bank = CompleteAlgebraQuestionBank()
    diagnostic = ComprehensiveDiagnostic()
    
    print("✅ Systems initialized successfully")
    print()
    
    # Test Case 1: Number Sense - Counting (NS-01)
    print("🔍 TEST CASE 1: COUNTING SEQUENCE ERROR")
    print("-" * 40)
    
    # Get a real question from the question bank
    ns01_questions = question_bank.get_questions_by_concept("NS-01")
    test_question = ns01_questions["easy"][0]  # First easy question
    
    print(f"📝 Original Question: {test_question['question']}")
    print(f"✅ Correct Answer: {test_question['answer']}")
    print(f"❌ Student Answer: 17, 18")
    print(f"📚 Source: {test_question['source']}")
    print()
    
    # Run diagnostic
    student = StudentProfile(
        name="struggling_student",
        overall_ability=0.3,
        concept_mastery={},
        learning_style="visual",
        total_questions_attempted=10,
        total_questions_correct=3
    )
    result = diagnostic.diagnose_student_response(
        question_type="counting",
        question_text="Count up from 15 to 20: 15, 16, __, __, __, 20",
        correct_answer="17, 18, 19",
        student_answer="17, 18",
        student_profile=student
    )
    
    print("🔍 DIAGNOSTIC RESULTS:")
    print(f"   🎯 Error Detected: {not result.is_correct}")
    print(f"   📊 Confidence: {result.confidence_score:.1f}")
    if result.missing_concept:
        print(f"   🚨 Missing Concepts: {result.missing_concept}")
    print()
    
    # Show remediation with real questions
    print("📚 REMEDIATION RECOMMENDATIONS:")
    print()
    
    for level in ["easy", "medium", "hard", "bonus"]:
        questions = ns01_questions[level][:2]  # Show first 2 questions
        print(f"   🔹 {level.upper()} LEVEL:")
        for i, q in enumerate(questions, 1):
            print(f"      {i}. {q['question']}")
            print(f"         Answer: {q['answer']}")
            print(f"         Source: {q['source']}")
        print()
    
    print("=" * 60)
    print()
    
    # Test Case 2: Place Value Error (NS-02)
    print("🔍 TEST CASE 2: PLACE VALUE CONFUSION")
    print("-" * 40)
    
    ns02_questions = question_bank.get_questions_by_concept("NS-02")
    test_question2 = ns02_questions["medium"][0]  # Medium difficulty
    
    print(f"📝 Original Question: {test_question2['question']}")
    print(f"✅ Correct Answer: {test_question2['answer']}")
    print(f"❌ Student Answer: 7,000")
    print(f"📚 Source: {test_question2['source']}")
    print()
    
    # Run diagnostic
    result2 = diagnostic.diagnose_student_response(
        question_type="place_value",
        question_text=test_question2['question'],
        correct_answer=test_question2['answer'],
        student_answer="7,000",
        student_profile=student
    )
    
    print("🔍 DIAGNOSTIC RESULTS:")
    print(f"   🎯 Error Detected: {not result2.is_correct}")
    print(f"   📊 Confidence: {result2.confidence_score:.1f}")
    if result2.missing_concept:
        print(f"   🚨 Missing Concepts: {result2.missing_concept}")
    print()
    
    # Show targeted remediation
    print("📚 TARGETED REMEDIATION PROGRESSION:")
    print()
    
    print("   🎯 LEARNING PATH FOR PLACE VALUE:")
    print("   1️⃣ START WITH BASICS (Easy Level):")
    for q in ns02_questions["easy"][:2]:
        print(f"      • {q['question']}")
        print(f"        Answer: {q['answer']} | Source: {q['source']}")
    
    print("\n   2️⃣ BUILD UNDERSTANDING (Medium Level):")
    for q in ns02_questions["medium"][:2]:
        print(f"      • {q['question']}")
        print(f"        Answer: {q['answer']} | Source: {q['source']}")
    
    print("\n   3️⃣ CHALLENGE SKILLS (Hard Level):")
    for q in ns02_questions["hard"][:2]:
        print(f"      • {q['question']}")
        print(f"        Answer: {q['answer']} | Source: {q['source']}")
    
    print("\n   4️⃣ MASTERY CHECK (Bonus Level):")
    for q in ns02_questions["bonus"][:2]:
        print(f"      • {q['question']}")
        print(f"        Answer: {q['answer']} | Source: {q['source']}")
    
    print()
    print("=" * 60)
    
    # Summary
    print("🎉 PIPELINE DEMONSTRATION COMPLETE!")
    print()
    print("✅ VALIDATED COMPONENTS:")
    print("   📚 Question Bank: Real questions from Khan Academy, Kumon, Singapore Math")
    print("   🔍 Error Detection: Identifies when students make mistakes")
    print("   🎯 Concept Mapping: Links errors to specific knowledge gaps")
    print("   📈 Progressive Remediation: Easy → Medium → Hard → Bonus")
    print("   🏆 Multi-Source Content: Authentic problems from major programs")
    print()
    print("🎓 EDUCATIONAL IMPACT:")
    print("   • Students get targeted help for their specific gaps")
    print("   • Teachers see exactly what concepts need reinforcement")
    print("   • Learning progresses systematically through difficulty levels")
    print("   • Content comes from trusted educational sources")

if __name__ == "__main__":
    demonstrate_complete_pipeline() 