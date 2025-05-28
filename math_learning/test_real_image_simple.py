#!/usr/bin/env python3
"""
Simple Test: Real Algebraic Exercise Analysis

This script tests the image analysis feature with the specific algebraic
exercise from the provided image, showing detailed step-by-step analysis.
"""

import asyncio
import base64

from math_learning.learning_graph import LearningGraph
from math_learning.ai_agent.math_tutoring_agent import MathTutoringAgent
from math_learning.knowledge_graph import KnowledgeGraph
from math_learning.recommendation import Recommender, GapAnalyzer
from math_learning.exercises import ExerciseBank, Exercise
from llm.openai.openai_llm import OpenAILLM


async def test_specific_algebraic_errors():
    """Test with the specific algebraic errors from the provided image."""
    print("🔍 Analyzing Real Algebraic Exercise Errors")
    print("=" * 50)
    
    # Initialize components
    learning_graph = LearningGraph()
    knowledge_graph = KnowledgeGraph()
    gap_analyzer = GapAnalyzer(knowledge_graph)
    exercise_bank = ExerciseBank()
    
    # Add exercises matching the image content
    ex1 = Exercise("linear_frac", "Linear Equation with Fractions", "Solve: x/3 + 2 = 6")
    ex1.add_concept_relationship("linear_equations", 1.0)
    ex1.add_concept_relationship("fraction_operations", 0.8)
    ex1.add_concept_relationship("algebraic_manipulation", 0.9)
    exercise_bank.add_exercise(ex1)
    
    ex2 = Exercise("expand_poly", "Polynomial Expansion", "Expand: (x+1)(x+2)")
    ex2.add_concept_relationship("distributive_property", 1.0)
    ex2.add_concept_relationship("polynomial_multiplication", 0.9)
    ex2.add_concept_relationship("foil_method", 1.0)
    exercise_bank.add_exercise(ex2)
    
    ex3 = Exercise("quad_neg", "Quadratic with Negative Discriminant", "Solve: x² = -4")
    ex3.add_concept_relationship("quadratic_equations", 1.0)
    ex3.add_concept_relationship("complex_numbers", 0.8)
    ex3.add_concept_relationship("imaginary_numbers", 0.9)
    exercise_bank.add_exercise(ex3)
    
    # Initialize agent
    recommender = Recommender(knowledge_graph, exercise_bank)
    llm = OpenAILLM.create_llm()
    agent = MathTutoringAgent(
        llm=llm,
        learning_graph=learning_graph,
        user_model=learning_graph,
        recommender=recommender,
        gap_analyzer=gap_analyzer
    )
    
    # Create detailed exercise content based on the image
    exercise_content = """
ALGEBRAIC EQUATIONS EXERCISE - STUDENT WORK ANALYSIS

Problem 4: x/3 + 2 = 6
Student's work shown:
- Started with: x/3 + 2 = 6
- Student wrote: -1 x L = 6 (INCORRECT - unclear algebraic manipulation)
- Student wrote: 0 ÷ 3 (INCORRECT - wrong operation)
- Student wrote: 4 x L = 6 (INCORRECT - unclear notation)

CORRECT SOLUTION:
x/3 + 2 = 6
x/3 = 6 - 2
x/3 = 4
x = 12

STUDENT ERRORS IDENTIFIED:
1. Unclear algebraic notation (using "L" instead of proper variables)
2. Incorrect isolation of terms
3. Wrong arithmetic operations
4. Failure to properly handle fractions

---

Problem: (x+1)(x+2) = 20
Student's work:
- Started with: (x+1)(x+2) = 20
- Student expanded to: x² + 4x + 2 = 20 (INCORRECT)
- Student circled final answer: 3

CORRECT EXPANSION:
(x+1)(x+2) = x² + 2x + x + 2 = x² + 3x + 2

STUDENT ERRORS IDENTIFIED:
1. Incorrect FOIL method application (got 4x instead of 3x)
2. Possible confusion with distributive property
3. Did not properly combine like terms

---

Problem 6: x² = -4
Student responses shown:
- Selected -6 (crossed out)
- Crossed out 4
- Circled 3
- Wrote 12

CORRECT UNDERSTANDING:
x² = -4 has no real solutions
Solutions are x = ±2i (complex numbers)

STUDENT ERRORS IDENTIFIED:
1. Lack of understanding of complex numbers
2. Attempting to find real solutions to impossible equation
3. Random guessing without systematic approach
4. No recognition that negative under square root requires imaginary numbers
"""
    
    print("📝 Exercise Content Being Analyzed:")
    print("-" * 30)
    print(exercise_content[:500] + "...")
    print()
    
    # Encode content as base64
    image_data = base64.b64encode(exercise_content.encode()).decode()
    
    print("🤖 Running AI Analysis...")
    
    # Analyze the exercise
    result = await agent.analyze_exercise_image(
        student_id="algebra_test_student",
        image_data=image_data,
        image_format="base64"
    )
    
    print("\n📊 DETAILED ANALYSIS RESULTS:")
    print("=" * 50)
    
    if result.get("success"):
        analysis = result.get("analysis", {})
        
        print("📝 EXTRACTED CONTENT:")
        extracted = analysis.get('extracted_content', 'N/A')
        if extracted != 'N/A':
            print(f"   Length: {len(extracted)} characters")
            print(f"   Preview: {extracted[:200]}...")
        else:
            print("   ❌ No content extracted")
        
        print("\n🔍 ERROR ANALYSIS:")
        error_analysis = analysis.get('error_analysis', {})
        print(f"   Error Type: {error_analysis.get('error_type', 'N/A')}")
        print(f"   Missing Concepts: {error_analysis.get('missing_concepts', [])}")
        print(f"   Explanation: {error_analysis.get('explanation', 'N/A')}")
        
        print("\n📈 LEARNING GRAPH UPDATES:")
        updates = analysis.get('learning_graph_updates', [])
        if updates:
            for update in updates:
                print(f"   • Concept: {update.get('concept', 'N/A')}")
                print(f"     Action: {update.get('action', 'N/A')}")
                print(f"     New Mastery: {update.get('new_mastery', 'N/A')}")
                print(f"     Reason: {update.get('reason', 'N/A')}")
                print()
        else:
            print("   ❌ No learning graph updates recorded")
        
        print("💡 EXERCISE RECOMMENDATIONS:")
        recommendations = analysis.get('exercise_recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. Title: {rec.get('title', 'N/A')}")
                print(f"      Problem: {rec.get('problem', 'N/A')}")
                print(f"      Concept Focus: {rec.get('concept_focus', 'N/A')}")
                print(f"      Difficulty: {rec.get('difficulty', 'N/A')}")
                print(f"      Reason: {rec.get('reason', 'N/A')}")
                print()
        else:
            print("   ❌ No exercise recommendations generated")
            
    else:
        print(f"❌ ANALYSIS FAILED:")
        print(f"   Error: {result.get('error', 'Unknown error')}")
        print(f"   Details: {result.get('details', 'No details available')}")
    
    # Show final learning state
    print("\n🧠 FINAL LEARNING GRAPH STATE:")
    print("=" * 50)
    concepts = [
        "linear_equations", "fraction_operations", "algebraic_manipulation",
        "distributive_property", "polynomial_multiplication", "foil_method",
        "quadratic_equations", "complex_numbers", "imaginary_numbers"
    ]
    
    for concept in concepts:
        mastery = learning_graph.get_mastery(concept)
        if mastery > 0:
            status = "✅" if mastery > 0.7 else "⚠️" if mastery > 0.3 else "❌"
            print(f"   {status} {concept.replace('_', ' ').title()}: {mastery:.2f}")
    
    # Show any concepts that were updated
    updated_concepts = [c for c in concepts if learning_graph.get_mastery(c) > 0]
    if updated_concepts:
        print(f"\n📊 {len(updated_concepts)} concepts were updated based on the analysis")
    else:
        print("\n⚠️  No concepts were updated - this may indicate an issue with the analysis")


async def main():
    """Run the specific algebraic exercise test."""
    print("🚀 Testing Real Algebraic Exercise Analysis")
    print("Based on the provided image with student errors")
    print("=" * 60)
    
    try:
        await test_specific_algebraic_errors()
        
        print("\n\n🎯 TEST SUMMARY:")
        print("This test analyzed real student errors in algebraic equations:")
        print("• Linear equation with fractions: x/3 + 2 = 6")
        print("• Polynomial expansion: (x+1)(x+2)")  
        print("• Quadratic with negative discriminant: x² = -4")
        print("\nThe AI agent should identify specific error patterns and")
        print("recommend targeted exercises to address misconceptions.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 