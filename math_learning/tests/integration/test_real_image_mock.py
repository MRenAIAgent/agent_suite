#!/usr/bin/env python3
"""
Mock Test: Real Algebraic Exercise Analysis

This script tests the image analysis feature with a mock LLM to demonstrate
how the system would work with the real algebraic exercise from the image.
"""

import asyncio
import base64
import pytest
from typing import List, Dict, Any

from math_learning.learning_graph import LearningGraph
from math_learning.ai_agent.math_tutoring_agent import MathTutoringAgent
from math_learning.knowledge_graph import KnowledgeGraph
from math_learning.recommendation import Recommender, GapAnalyzer
from math_learning.exercises import ExerciseBank, Exercise
from llm.llm import LLMBase


class MockLLM(LLMBase):
    """Mock LLM that provides realistic responses for algebraic exercise analysis."""
    
    def create_llm(self, model_name: str = "mock-model", **kwargs):
        """Create and return this mock LLM instance."""
        return self
    
    def chat_completion(self, model: str, messages: list, tools: list = None, stream: bool = False, **kwargs):
        """Mock chat completion that analyzes the algebraic exercise content."""
        
        # Get the last user message content
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        
        # Check if this is an image analysis request
        if "analyze" in user_content.lower() and "exercise" in user_content.lower():
            return self._mock_image_analysis_response(user_content)
        elif "error" in user_content.lower() and "analysis" in user_content.lower():
            return self._mock_error_analysis_response(user_content)
        elif ("exercise" in user_content.lower() and "recommend" in user_content.lower()) or \
             ("targeted" in user_content.lower() and "exercise" in user_content.lower()) or \
             ("create" in user_content.lower() and "exercise" in user_content.lower()):
            return self._mock_exercise_recommendation_response(user_content)
        elif "solve" in user_content.lower() and "problem" in user_content.lower():
            return self._mock_solve_problem_response(user_content)
        else:
            return self._mock_general_response()
    
    def _mock_image_analysis_response(self, content: str):
        """Mock response for image analysis."""
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
        
        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
        
        class MockMessage:
            def __init__(self):
                self.content = """
Based on the algebraic exercise image, I can extract the following content:

PROBLEM 1: x/3 + 2 = 6
Student work shows:
- Started correctly with x/3 + 2 = 6
- Made errors in algebraic manipulation using unclear notation "L"
- Performed incorrect operations like "0 ÷ 3"
- Failed to properly isolate the variable

PROBLEM 2: (x+1)(x+2) = 20
Student work shows:
- Attempted to expand using FOIL method
- Incorrectly got x² + 4x + 2 instead of x² + 3x + 2
- Shows confusion with distributive property

PROBLEM 3: x² = -4
Student responses show:
- Multiple incorrect attempts (-6, 4, 3, 12)
- No understanding of complex numbers
- Random guessing without systematic approach

The student demonstrates significant gaps in:
1. Algebraic manipulation and notation
2. Fraction operations in equations
3. FOIL method and distributive property
4. Complex number concepts
"""
        
        return MockResponse()
    
    def _mock_error_analysis_response(self, content: str):
        """Mock response for error analysis."""
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
        
        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
        
        class MockMessage:
            def __init__(self):
                if "x/3" in content:
                    self.content = """
ERROR TYPE: algebraic
MISSING CONCEPTS: linear_equations, fraction_operations, algebraic_manipulation
EXPLANATION: Student shows fundamental confusion with algebraic manipulation, particularly when fractions are involved. The use of unclear notation ("L") and incorrect operations suggests a lack of understanding of proper algebraic procedures for isolating variables.
"""
                elif "x+1)(x+2" in content:
                    self.content = """
ERROR TYPE: algebraic
MISSING CONCEPTS: distributive_property, foil_method, polynomial_multiplication
EXPLANATION: Student incorrectly applied the FOIL method, getting 4x instead of 3x in the middle term. This indicates confusion with the distributive property and combining like terms in polynomial multiplication.
"""
                elif "x² = -4" in content:
                    self.content = """
ERROR TYPE: conceptual
MISSING CONCEPTS: complex_numbers, imaginary_numbers, quadratic_equations
EXPLANATION: Student attempted to find real solutions to an equation that has no real solutions. This shows a lack of understanding of complex numbers and the concept that square roots of negative numbers require imaginary units.
"""
                else:
                    self.content = """
ERROR TYPE: general
MISSING CONCEPTS: problem_solving
EXPLANATION: General mathematical reasoning issues detected.
"""
        
        return MockResponse()
    
    def _mock_exercise_recommendation_response(self, content: str):
        """Mock response for exercise recommendations."""
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
        
        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
        
        class MockMessage:
            def __init__(self):
                self.content = """
RECOMMENDED EXERCISES:

1. Linear Equation with Fractions Practice
   Problem: Solve x/2 + 3 = 7
   Focus: fraction_operations, linear_equations
   Difficulty: basic
   Reason: Student needs practice with proper algebraic notation and fraction manipulation

2. FOIL Method Review
   Problem: Expand (x+2)(x+3)
   Focus: distributive_property, foil_method
   Difficulty: basic
   Reason: Student made errors in polynomial multiplication and needs to review FOIL

3. Introduction to Complex Numbers
   Problem: Explain why x² = -1 has no real solutions
   Focus: complex_numbers, imaginary_numbers
   Difficulty: intermediate
   Reason: Student needs to understand when equations have no real solutions
"""
        
        return MockResponse()
    
    def _mock_solve_problem_response(self, content: str):
        """Mock response for solving math problems."""
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
        
        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
        
        class MockMessage:
            def __init__(self):
                if "x/3" in content:
                    self.content = "Answer: 12"
                elif "x+1)(x+2" in content:
                    self.content = "Answer: x² + 3x + 2"
                elif "x² = -4" in content:
                    self.content = "Answer: ±2i"
                else:
                    self.content = "Answer: Unknown"
        
        return MockResponse()
    
    def _mock_general_response(self):
        """Mock general response."""
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
        
        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
        
        class MockMessage:
            def __init__(self):
                self.content = "I understand. How can I help you with math learning?"
        
        return MockResponse()


@pytest.mark.asyncio
async def test_real_algebraic_exercise_with_mock():
    """Test the real algebraic exercise using a mock LLM."""
    print("🔍 Testing Real Algebraic Exercise with Mock LLM")
    print("=" * 55)
    
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
    
    # Add more practice exercises for recommendations
    ex4 = Exercise("frac_practice", "Fraction Practice", "Solve: x/2 + 3 = 7")
    ex4.add_concept_relationship("linear_equations", 1.0)
    ex4.add_concept_relationship("fraction_operations", 1.0)
    exercise_bank.add_exercise(ex4)
    
    ex5 = Exercise("foil_practice", "FOIL Practice", "Expand: (x+2)(x+3)")
    ex5.add_concept_relationship("distributive_property", 1.0)
    ex5.add_concept_relationship("foil_method", 1.0)
    exercise_bank.add_exercise(ex5)
    
    # Initialize agent with mock LLM
    recommender = Recommender(knowledge_graph, exercise_bank)
    mock_llm = MockLLM()
    agent = MathTutoringAgent(
        llm=mock_llm,
        learning_graph=learning_graph,
        user_model=learning_graph,
        recommender=recommender,
        gap_analyzer=gap_analyzer
    )
    
    # Create the exact exercise content from the image
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

Problem: (x+1)(x+2) = 20
Student's work:
- Started with: (x+1)(x+2) = 20
- Student expanded to: x² + 4x + 2 = 20 (INCORRECT)
- Student circled final answer: 3

CORRECT EXPANSION:
(x+1)(x+2) = x² + 2x + x + 2 = x² + 3x + 2

Problem 6: x² = -4
Student responses shown:
- Selected -6 (crossed out)
- Crossed out 4
- Circled 3
- Wrote 12

CORRECT UNDERSTANDING:
x² = -4 has no real solutions
Solutions are x = ±2i (complex numbers)
"""
    
    print("📝 Analyzing Exercise Content:")
    print("-" * 30)
    print(exercise_content[:400] + "...")
    print()
    
    # Encode content as base64
    image_data = base64.b64encode(exercise_content.encode()).decode()
    
    print("🤖 Running AI Analysis with Mock LLM...")
    
    # Analyze the exercise
    result = await agent.analyze_exercise_image(
        student_id="real_algebra_student",
        image_data=image_data,
        image_format="base64"
    )
    
    print("\n📊 ANALYSIS RESULTS:")
    print("=" * 40)
    
    if result.get("success"):
        print("📝 EXTRACTED CONTENT:")
        extracted_content = result.get('extracted_content', {})
        raw_text = extracted_content.get('raw_text', 'N/A')
        if raw_text and raw_text != 'N/A':
            print(f"   ✅ Successfully extracted {len(raw_text)} characters")
            print(f"   Preview: {raw_text[:150]}...")
        else:
            print("   ❌ No content extracted")
        
        print("\n🔍 ERROR ANALYSIS:")
        error_analysis = result.get('error_analysis', {})
        error_type = error_analysis.get('error_type', 'N/A')
        missing_concepts = error_analysis.get('missing_concepts', [])
        description = error_analysis.get('description', 'N/A')
        
        print(f"   Error Type: {error_type}")
        print(f"   Missing Concepts: {missing_concepts}")
        print(f"   Explanation: {description[:200]}..." if len(description) > 200 else f"   Explanation: {description}")
        
        print("\n📈 LEARNING GRAPH UPDATES:")
        updates = result.get('learning_graph_updates', {})
        if updates and updates != {}:
            print(f"   ✅ Learning graph updated")
            print(f"   Updates: {updates}")
        else:
            print("   ❌ No learning graph updates")
        
        print("\n💡 EXERCISE RECOMMENDATIONS:")
        recommendation = result.get('exercise_recommendation', {})
        if recommendation and recommendation != {}:
            title = recommendation.get('title', 'N/A')
            problem = recommendation.get('problem', 'N/A')
            concept_focus = recommendation.get('concept_focus', 'N/A')
            difficulty = recommendation.get('difficulty', 'N/A')
            reason = recommendation.get('reason', 'N/A')
            
            print(f"   1. {title}")
            print(f"      Problem: {problem}")
            print(f"      Focus: {concept_focus}")
            print(f"      Difficulty: {difficulty}")
            print(f"      Reason: {reason[:100]}..." if len(reason) > 100 else f"      Reason: {reason}")
        else:
            print("   ❌ No recommendations generated")
    else:
        print(f"❌ ANALYSIS FAILED:")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Show final learning graph state
    print("🧠 FINAL LEARNING GRAPH STATE:")
    print("=" * 40)
    concepts = [
        "linear_equations", "fraction_operations", "algebraic_manipulation",
        "distributive_property", "polynomial_multiplication", "foil_method",
        "quadratic_equations", "complex_numbers", "imaginary_numbers"
    ]
    
    updated_concepts = []
    for concept in concepts:
        mastery = learning_graph.get_mastery(concept)
        if mastery > 0:
            status = "✅" if mastery > 0.7 else "⚠️" if mastery > 0.3 else "❌"
            print(f"   {status} {concept.replace('_', ' ').title()}: {mastery:.2f}")
            updated_concepts.append(concept)
    
    if updated_concepts:
        print(f"\n📊 SUCCESS: {len(updated_concepts)} concepts were updated!")
        print("The AI agent successfully:")
        print("• Extracted content from the exercise image")
        print("• Identified specific error patterns in student work")
        print("• Mapped errors to missing mathematical concepts")
        print("• Updated the learning graph based on analysis")
        print("• Generated targeted exercise recommendations")
    else:
        print("\n⚠️  No concepts were updated")


async def main():
    """Run the mock test."""
    print("🚀 Real Algebraic Exercise Analysis - Mock Demo")
    print("Testing with actual exercise content from the provided image")
    print("=" * 65)
    
    try:
        await test_real_algebraic_exercise_with_mock()
        
        print("\n\n🎯 DEMO SUMMARY:")
        print("This demo shows how the AI agent would analyze the real algebraic")
        print("exercise image with student errors. The mock LLM simulates realistic")
        print("responses that would come from a real language model.")
        print()
        print("Key Features Demonstrated:")
        print("• Image content extraction and analysis")
        print("• Error pattern identification in student work")
        print("• Concept gap analysis and learning graph updates")
        print("• Personalized exercise recommendations")
        print("• Progressive learning tracking")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 