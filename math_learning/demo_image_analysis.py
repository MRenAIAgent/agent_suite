#!/usr/bin/env python3
"""
Demo: Visual Learning Analysis Feature

This script demonstrates how to use the new image analysis feature
to analyze student exercise images and build their learning graph.
"""

import asyncio
import base64
from typing import List

from math_learning.learning_graph import LearningGraph
from math_learning.ai_agent.math_tutoring_agent import MathTutoringAgent
from math_learning.knowledge_graph import KnowledgeGraph
from math_learning.recommendation import Recommender, GapAnalyzer
from math_learning.exercises import ExerciseBank, Exercise
from llm.openai.openai_llm import OpenAILLM


def create_sample_image_data(problem: str, student_answer: str, correct_answer: str) -> str:
    """Create a sample base64 encoded image for testing."""
    # In practice, this would be actual image data
    # For demo purposes, we'll encode text as base64
    text_content = f"Problem: {problem}\nStudent Answer: {student_answer}\nCorrect Answer: {correct_answer}"
    return base64.b64encode(text_content.encode()).decode()


def create_sample_exercise_bank() -> ExerciseBank:
    """Create a sample exercise bank with real exercises."""
    exercise_bank = ExerciseBank("Math Learning Demo Bank")
    
    # Basic Addition Exercise
    ex1 = Exercise(
        title="Basic Addition",
        problem="What is 3 + 4?",
        solution="7",
        difficulty=1,
        estimated_time=2
    )
    ex1.add_concept_relationship("basic_arithmetic", weight=1.0, relationship_type="primary")
    ex1.add_concept_relationship("addition", weight=0.9, relationship_type="primary")
    exercise_bank.add_exercise(ex1)
    
    # Basic Subtraction Exercise
    ex2 = Exercise(
        title="Basic Subtraction",
        problem="What is 8 - 3?",
        solution="5",
        difficulty=1,
        estimated_time=2
    )
    ex2.add_concept_relationship("basic_arithmetic", weight=1.0, relationship_type="primary")
    ex2.add_concept_relationship("subtraction", weight=0.9, relationship_type="primary")
    exercise_bank.add_exercise(ex2)
    
    # Basic Multiplication Exercise
    ex3 = Exercise(
        title="Basic Multiplication",
        problem="What is 5 × 6?",
        solution="30",
        difficulty=2,
        estimated_time=3
    )
    ex3.add_concept_relationship("multiplication", weight=1.0, relationship_type="primary")
    ex3.add_concept_relationship("basic_arithmetic", weight=0.7, relationship_type="secondary")
    exercise_bank.add_exercise(ex3)
    
    # Advanced Addition Exercise
    ex4 = Exercise(
        title="Two-Digit Addition",
        problem="What is 27 + 35?",
        solution="62",
        difficulty=3,
        estimated_time=4
    )
    ex4.add_concept_relationship("addition", weight=1.0, relationship_type="primary")
    ex4.add_concept_relationship("place_value", weight=0.8, relationship_type="secondary")
    exercise_bank.add_exercise(ex4)
    
    # Multiplication Tables Exercise
    ex5 = Exercise(
        title="Multiplication Tables",
        problem="What is 7 × 8?",
        solution="56",
        difficulty=3,
        estimated_time=3
    )
    ex5.add_concept_relationship("multiplication", weight=1.0, relationship_type="primary")
    ex5.add_concept_relationship("multiplication_tables", weight=0.9, relationship_type="primary")
    exercise_bank.add_exercise(ex5)
    
    return exercise_bank


async def demo_single_image_analysis():
    """Demonstrate analyzing a single exercise image."""
    print("🔍 Demo: Single Image Analysis")
    print("=" * 50)
    
    # Initialize components
    learning_graph = LearningGraph()
    knowledge_graph = KnowledgeGraph()
    gap_analyzer = GapAnalyzer(knowledge_graph)
    
    # Create a simple exercise bank for the recommender
    exercise_bank = create_sample_exercise_bank()
    recommender = Recommender(knowledge_graph, exercise_bank)
    
    llm = OpenAILLM.create_llm()
    agent = MathTutoringAgent(
        llm=llm, 
        learning_graph=learning_graph,
        user_model=learning_graph,
        recommender=recommender,
        gap_analyzer=gap_analyzer
    )
    
    # Create sample image data
    image_data = create_sample_image_data(
        problem="What is 15 × 7?",
        student_answer="95",
        correct_answer="105"
    )
    
    # Analyze the image
    print("📸 Analyzing student's multiplication exercise...")
    result = await agent.analyze_exercise_image(
        student_id="demo_student",
        image_data=image_data,
        image_format="base64"
    )
    
    print(f"✅ Analysis complete!")
    print(f"📝 Problem identified: {result.get('extracted_content', {}).get('problem', 'N/A')}")
    print(f"❌ Student's answer: {result.get('extracted_content', {}).get('student_answer', 'N/A')}")
    print(f"✅ Correct answer: {result.get('extracted_content', {}).get('correct_answer', 'N/A')}")
    print(f"🔍 Error type: {result.get('error_analysis', {}).get('error_type', 'N/A')}")
    print(f"📚 Missing concepts: {result.get('error_analysis', {}).get('missing_concepts', [])}")
    print(f"💡 Recommended exercise: {result.get('exercise_recommendation', {}).get('problem', 'N/A')}")
    print()


async def demo_progressive_learning():
    """Demonstrate building a learning graph through multiple images."""
    print("📈 Demo: Progressive Learning Graph Building")
    print("=" * 50)
    
    # Initialize components
    learning_graph = LearningGraph()
    knowledge_graph = KnowledgeGraph()
    gap_analyzer = GapAnalyzer(knowledge_graph)
    
    # Create a simple exercise bank for the recommender
    exercise_bank = create_sample_exercise_bank()
    recommender = Recommender(knowledge_graph, exercise_bank)
    
    llm = OpenAILLM.create_llm()
    agent = MathTutoringAgent(
        llm=llm, 
        learning_graph=learning_graph,
        user_model=learning_graph,
        recommender=recommender,
        gap_analyzer=gap_analyzer
    )
    
    # Sample exercise images showing progression
    exercises = [
        {
            "problem": "What is 8 + 5?",
            "student_answer": "14",
            "correct_answer": "13",
            "description": "Basic addition error"
        },
        {
            "problem": "What is 12 - 7?",
            "student_answer": "6",
            "correct_answer": "5",
            "description": "Subtraction with borrowing"
        },
        {
            "problem": "What is 6 × 4?",
            "student_answer": "26",
            "correct_answer": "24",
            "description": "Multiplication table confusion"
        }
    ]
    
    # Process each exercise
    image_data_list = []
    for i, exercise in enumerate(exercises, 1):
        print(f"📸 Processing exercise {i}: {exercise['description']}")
        image_data = create_sample_image_data(
            exercise["problem"],
            exercise["student_answer"],
            exercise["correct_answer"]
        )
        image_data_list.append(image_data)
    
    # Analyze all images together
    images = [{"data": img_data, "format": "base64"} for img_data in image_data_list]
    results = await agent.process_multiple_images(
        student_id="progressive_student",
        images=images
    )
    
    print(f"\n📊 Progressive Analysis Results:")
    print(f"✅ Processed {len(results['individual_results'])} exercises")
    print(f"📚 Total concepts identified: {results['summary']['total_concepts_identified']}")
    print(f"⚠️  Most problematic concepts: {results['summary']['most_problematic_concepts']}")
    print(f"📈 Learning trajectory: {results['summary']['learning_trajectory']}")
    print()


async def demo_learning_graph_evolution():
    """Demonstrate how the learning graph evolves with each image."""
    print("🧠 Demo: Learning Graph Evolution")
    print("=" * 50)
    
    # Initialize components
    learning_graph = LearningGraph()
    knowledge_graph = KnowledgeGraph()
    gap_analyzer = GapAnalyzer(knowledge_graph)
    
    # Create a simple exercise bank for the recommender
    exercise_bank = create_sample_exercise_bank()
    recommender = Recommender(knowledge_graph, exercise_bank)
    
    llm = OpenAILLM.create_llm()
    agent = MathTutoringAgent(
        llm=llm, 
        learning_graph=learning_graph,
        user_model=learning_graph,
        recommender=recommender,
        gap_analyzer=gap_analyzer
    )
    
    # Set initial mastery levels
    student_id = "evolution_student"
    learning_graph.update_mastery("addition", True, 0.5, 1.0)
    learning_graph.update_mastery("subtraction", True, 0.5, 1.0)
    learning_graph.update_mastery("multiplication", False, 0.5, 1.0)
    
    print("📊 Initial Learning Graph State:")
    for concept in ["addition", "subtraction", "multiplication"]:
        mastery = learning_graph.get_mastery(concept)
        print(f"   {concept.title()}: {mastery:.2f}")
    
    # Analyze an image that reveals gaps
    image_data = create_sample_image_data(
        problem="What is 9 × 6?",
        student_answer="48",
        correct_answer="54"
    )
    
    print(f"\n📸 Analyzing multiplication exercise...")
    result = await agent.analyze_exercise_image(
        student_id=student_id,
        image_data=image_data,
        image_format="base64"
    )
    
    print(f"\n📊 Learning Graph State After Analysis:")
    for concept in ["addition", "subtraction", "multiplication"]:
        mastery = learning_graph.get_mastery(concept)
        print(f"   {concept.title()}: {mastery:.2f}")
    
    # Show identified gaps and recommendations
    if result.get('error_analysis'):
        missing_concepts = result['error_analysis'].get('missing_concepts', [])
        print(f"\n🎯 Newly identified gaps: {missing_concepts}")
    
    if result.get('exercise_recommendation'):
        rec = result['exercise_recommendation']
        print(f"💡 Next recommended exercise: {rec.get('problem', 'N/A')}")
        print(f"🎯 Target concept: {rec.get('target_concept', 'N/A')}")
    print()


async def main():
    """Run all demonstrations."""
    print("🚀 Visual Learning Analysis Feature Demo")
    print("=" * 60)
    print("This demo shows how AI agents can analyze exercise images")
    print("to understand student mistakes and build learning graphs.")
    print("=" * 60)
    print()
    
    await demo_single_image_analysis()
    await demo_progressive_learning()
    await demo_learning_graph_evolution()
    
    print("🎉 Demo Complete!")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("✅ Image upload and OCR text extraction")
    print("✅ Mathematical error analysis and classification")
    print("✅ Concept mapping from errors to knowledge gaps")
    print("✅ Dynamic learning graph updates")
    print("✅ Personalized exercise recommendations")
    print("✅ Progressive learning tracking across multiple images")
    print("✅ Learning trajectory analysis and predictions")


if __name__ == "__main__":
    asyncio.run(main()) 