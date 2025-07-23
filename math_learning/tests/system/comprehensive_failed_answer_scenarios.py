#!/usr/bin/env python3
"""
COMPREHENSIVE FAILED ANSWER SCENARIOS
Demonstrates multiple realistic failure scenarios across different algebra concepts

Shows complete pipeline for each scenario:
1. Original question from question bank
2. Student's failed answer
3. Identified missing concept
4. Complete remediation path (Easy → Medium → Hard → Bonus)
5. Educational rationale for each recommendation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_diagnostic import ComprehensiveDiagnostic, StudentProfile
from algebra_comprehensive_question_bank_complete import CompleteAlgebraQuestionBank
import json

def run_comprehensive_scenarios():
    """Run multiple comprehensive failure scenarios"""
    
    print("🎯 COMPREHENSIVE FAILED ANSWER SCENARIOS")
    print("=" * 70)
    
    # Initialize systems
    question_bank = CompleteAlgebraQuestionBank()
    diagnostic = ComprehensiveDiagnostic()
    
    # Create different student profiles
    students = {
        "struggling": StudentProfile(
            name="Alex (Struggling)",
            overall_ability=0.25,
            concept_mastery={"NS-01": 0.2, "NS-02": 0.1},
            learning_style="visual",
            total_questions_attempted=20,
            total_questions_correct=5
        ),
        "average": StudentProfile(
            name="Jordan (Average)",
            overall_ability=0.65,
            concept_mastery={"NS-01": 0.7, "NS-02": 0.6},
            learning_style="analytical",
            total_questions_attempted=30,
            total_questions_correct=20
        ),
        "advanced": StudentProfile(
            name="Sam (Advanced)",
            overall_ability=0.85,
            concept_mastery={"NS-01": 0.9, "NS-02": 0.8},
            learning_style="hands_on",
            total_questions_attempted=40,
            total_questions_correct=35
        )
    }
    
    print("✅ Systems initialized with 3 student profiles")
    print()
    
    # Scenario 1: Counting Sequence Error (NS-01)
    print("📚 SCENARIO 1: COUNTING SEQUENCE ERROR")
    print("=" * 50)
    
    ns01_questions = question_bank.get_questions_by_concept("NS-01")
    test_question = ns01_questions["medium"][0]  # Medium difficulty counting
    
    print(f"📝 Question: {test_question['question']}")
    print(f"✅ Correct Answer: {test_question['answer']}")
    print(f"📚 Source: {test_question['source']}")
    print()
    
    # Test with struggling student
    student = students["struggling"]
    print(f"👤 Student: {student.name} (Ability: {student.overall_ability:.0%})")
    print(f"❌ Student Answer: 97, 98, 100, 101, 102, 103")
    print(f"🔍 Error: Skipped 99")
    print()
    
    result = diagnostic.diagnose_student_response(
        question_type="counting",
        question_text=test_question['question'],
        correct_answer=test_question['answer'],
        student_answer="97, 98, 100, 101, 102, 103",
        student_profile=student
    )
    
    print("🔍 DIAGNOSTIC RESULTS:")
    print(f"   🎯 Error Detected: {not result.is_correct}")
    print(f"   📊 Confidence: {result.confidence_score:.1f}")
    if result.missing_concept:
        print(f"   🚨 Missing Concept: {result.missing_concept}")
    print(f"   💡 Explanation: {result.explanation}")
    print()
    
    print("📈 REMEDIATION PATH:")
    show_remediation_path("NS-01", question_bank, "Counting sequences")
    
    print("=" * 70)
    print()
    
    # Scenario 2: Place Value Error (NS-02)
    print("📚 SCENARIO 2: PLACE VALUE CONFUSION")
    print("=" * 50)
    
    ns02_questions = question_bank.get_questions_by_concept("NS-02")
    test_question2 = ns02_questions["hard"][0]  # Hard difficulty place value
    
    print(f"📝 Question: {test_question2['question']}")
    print(f"✅ Correct Answer: {test_question2['answer']}")
    print(f"📚 Source: {test_question2['source']}")
    print()
    
    # Test with average student
    student = students["average"]
    print(f"👤 Student: {student.name} (Ability: {student.overall_ability:.0%})")
    print(f"❌ Student Answer: 40,000")
    print(f"🔍 Error: Confused place value position")
    print()
    
    result2 = diagnostic.diagnose_student_response(
        question_type="place_value",
        question_text=test_question2['question'],
        correct_answer=test_question2['answer'],
        student_answer="40,000",
        student_profile=student
    )
    
    print("🔍 DIAGNOSTIC RESULTS:")
    print(f"   🎯 Error Detected: {not result2.is_correct}")
    print(f"   📊 Confidence: {result2.confidence_score:.1f}")
    if result2.missing_concept:
        print(f"   🚨 Missing Concept: {result2.missing_concept}")
    print(f"   💡 Explanation: {result2.explanation}")
    print()
    
    print("📈 REMEDIATION PATH:")
    show_remediation_path("NS-02", question_bank, "Place value understanding")
    
    print("=" * 70)
    print()
    
    # Scenario 3: Commutative Property Error (NS-03)
    print("📚 SCENARIO 3: COMMUTATIVE PROPERTY MISUNDERSTANDING")
    print("=" * 50)
    
    ns03_questions = question_bank.get_questions_by_concept("NS-03")
    test_question3 = ns03_questions["easy"][1]  # Easy commutative property
    
    print(f"📝 Question: {test_question3['question']}")
    print(f"✅ Correct Answer: {test_question3['answer']}")
    print(f"📚 Source: {test_question3['source']}")
    print()
    
    # Test with advanced student making a conceptual error
    student = students["advanced"]
    print(f"👤 Student: {student.name} (Ability: {student.overall_ability:.0%})")
    print(f"❌ Student Answer: No, order matters in multiplication")
    print(f"🔍 Error: Doesn't understand commutative property")
    print()
    
    result3 = diagnostic.diagnose_student_response(
        question_type="commutative_property",
        question_text=test_question3['question'],
        correct_answer=test_question3['answer'],
        student_answer="No, order matters in multiplication",
        student_profile=student
    )
    
    print("🔍 DIAGNOSTIC RESULTS:")
    print(f"   🎯 Error Detected: {not result3.is_correct}")
    print(f"   📊 Confidence: {result3.confidence_score:.1f}")
    if result3.missing_concept:
        print(f"   🚨 Missing Concept: {result3.missing_concept}")
    print(f"   💡 Explanation: {result3.explanation}")
    print()
    
    print("📈 REMEDIATION PATH:")
    show_remediation_path("NS-03", question_bank, "Commutative property")
    
    print("=" * 70)
    print()
    
    # Summary
    print("🎉 COMPREHENSIVE SCENARIOS COMPLETE!")
    print()
    print("✅ DEMONSTRATED CAPABILITIES:")
    print("   🎯 Multi-Level Error Detection: Basic counting → Place value → Properties")
    print("   👥 Adaptive to Student Profiles: Struggling → Average → Advanced")
    print("   📊 Confidence Scoring: System assesses certainty of diagnosis")
    print("   🔍 Detailed Explanations: Clear rationale for each diagnosis")
    print("   📈 Progressive Remediation: Systematic skill building")
    print("   🏆 Multi-Source Questions: Khan Academy, Kumon, Singapore Math, AoPS")
    print()
    print("🎓 EDUCATIONAL VALUE:")
    print("   • Teachers get precise diagnostic information")
    print("   • Students receive targeted, level-appropriate practice")
    print("   • Learning progresses systematically through difficulty levels")
    print("   • Content comes from proven educational programs")
    print("   • System adapts to individual student needs and abilities")

def show_remediation_path(concept_id: str, question_bank, concept_name: str):
    """Show the complete remediation path for a concept"""
    
    questions = question_bank.get_questions_by_concept(concept_id)
    
    print(f"   🎯 TARGET CONCEPT: {concept_name} ({concept_id})")
    print()
    
    levels = [
        ("easy", "🟢 FOUNDATION", "Build basic understanding"),
        ("medium", "🟡 DEVELOPMENT", "Strengthen skills"),
        ("hard", "🟠 MASTERY", "Challenge and extend"),
        ("bonus", "🔴 EXCELLENCE", "Competition-level mastery")
    ]
    
    for level, emoji_name, description in levels:
        level_questions = questions[level][:2]  # Show first 2 questions
        print(f"   {emoji_name} ({description}):")
        
        for i, q in enumerate(level_questions, 1):
            print(f"      {i}. {q['question']}")
            print(f"         ✅ Answer: {q['answer']}")
            print(f"         📚 Source: {q['source']}")
            if 'type' in q:
                print(f"         🏷️  Type: {q['type']}")
        print()

if __name__ == "__main__":
    run_comprehensive_scenarios() 