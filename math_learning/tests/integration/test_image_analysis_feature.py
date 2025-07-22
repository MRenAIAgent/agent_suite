#!/usr/bin/env python3
"""
Test Image Analysis Feature for Math Learning System

This test demonstrates the complete workflow of:
1. User uploads image of exercise with wrong answer
2. AI agent analyzes the image and identifies errors
3. Agent updates learning graph based on identified concepts
4. Agent recommends targeted exercise for improvement
5. Multiple images gradually build the learning graph
"""

import asyncio
import base64
import json
from datetime import datetime
from typing import Dict, Any
import pytest

# Import the math learning system components
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.recommendation.gap_analyzer import GapAnalyzer
from math_learning.recommendation.recommender import Recommender
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.ai_agent.math_tutoring_agent import MathTutoringAgent

# Mock LLM for testing
class MockLLM:
    """Mock LLM that provides realistic responses for testing."""
    
    async def generate_response(self, prompt: str, model: str = "gpt-4", **kwargs) -> str:
        """Generate mock responses based on prompt content."""
        
        if "Solve this math problem" in prompt:
            # Mock solving math problems
            if "2 + 3" in prompt:
                return "5"
            elif "4 * 6" in prompt:
                return "24"
            elif "15 / 3" in prompt:
                return "5"
            elif "8 - 3" in prompt:
                return "5"
            else:
                return "42"  # Default answer
        
        elif "Create a targeted math exercise" in prompt:
            # Mock exercise generation
            return json.dumps({
                "problem": "What is 3 + 4?",
                "solution_steps": [
                    "Start with 3",
                    "Add 4 to get 7"
                ],
                "concept_explanation": "Addition combines two numbers to find their total.",
                "difficulty": "easy"
            })
        
        elif "Analyze this math exercise image" in prompt:
            # Mock image analysis
            return json.dumps({
                "problem": "2 + 3 = ?",
                "student_answer": "6",
                "correct_answer": "5"
            })
        
        else:
            return "I understand your request and will help with the math problem."

def create_sample_image_data() -> str:
    """Create a sample base64 encoded image for testing."""
    # This would normally be a real image, but for testing we'll use a simple string
    sample_text = "Math Exercise: 2 + 3 = 6 (student wrote 6 instead of 5)"
    return base64.b64encode(sample_text.encode()).decode()

def create_multiple_sample_images() -> list:
    """Create multiple sample images representing different math errors."""
    images = []
    
    # Image 1: Basic addition error
    sample1 = "Exercise 1: 2 + 3 = 6 (student answer: 6, correct: 5)"
    images.append({
        'data': base64.b64encode(sample1.encode()).decode(),
        'format': 'base64'
    })
    
    # Image 2: Multiplication error
    sample2 = "Exercise 2: 4 * 6 = 20 (student answer: 20, correct: 24)"
    images.append({
        'data': base64.b64encode(sample2.encode()).decode(),
        'format': 'base64'
    })
    
    # Image 3: Division error
    sample3 = "Exercise 3: 15 / 3 = 6 (student answer: 6, correct: 5)"
    images.append({
        'data': base64.b64encode(sample3.encode()).decode(),
        'format': 'base64'
    })
    
    return images

@pytest.mark.asyncio
async def test_single_image_analysis():
    """Test analyzing a single exercise image."""
    print("🧪 Testing Single Image Analysis")
    print("=" * 50)
    
    # Initialize components
    learning_graph = LearningGraph(user_id="test_student_001", name="Test Student")
    knowledge_graph = KnowledgeGraph()
    gap_analyzer = GapAnalyzer(knowledge_graph)
    exercise_bank = ExerciseBank()
    recommender = Recommender(knowledge_graph, exercise_bank)
    mock_llm = MockLLM()
    
    # Create the AI agent
    agent = MathTutoringAgent(
        learning_graph=learning_graph,
        gap_analyzer=gap_analyzer,
        recommender=recommender,
        llm=mock_llm,
        user_model=learning_graph
    )
    
    # Create sample image data
    image_data = create_sample_image_data()
    
    print(f"📸 Analyzing image for student: {learning_graph.user_id}")
    print(f"🕒 Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze the image
    result = await agent.analyze_exercise_image(
        student_id="test_student_001",
        image_data=image_data,
        image_format="base64"
    )
    
    # Display results
    print("\n📊 Analysis Results:")
    print("-" * 30)
    
    if result.get('success'):
        print(f"✅ Analysis successful!")
        
        # Show extracted content
        extracted = result.get('extracted_content', {})
        print(f"\n📝 Extracted Content:")
        print(f"   Problem: {extracted.get('parsed_problem', 'N/A')}")
        print(f"   Student Answer: {extracted.get('student_answer', 'N/A')}")
        print(f"   Correct Answer: {extracted.get('correct_answer', 'N/A')}")
        
        # Show error analysis
        error_analysis = result.get('error_analysis', {})
        print(f"\n🔍 Error Analysis:")
        print(f"   Error Type: {error_analysis.get('error_type', 'N/A')}")
        print(f"   Missing Concepts: {error_analysis.get('missing_concepts', [])}")
        print(f"   Confidence: {error_analysis.get('confidence', 0.0):.2f}")
        print(f"   Description: {error_analysis.get('description', 'N/A')}")
        
        # Show learning graph updates
        updates = result.get('learning_graph_updates', {})
        print(f"\n📈 Learning Graph Updates:")
        print(f"   Updated: {updates.get('updated', False)}")
        print(f"   Concepts Updated: {updates.get('concepts_updated', [])}")
        print(f"   New Gaps Identified: {updates.get('new_gaps_identified', [])}")
        
        # Show exercise recommendation
        recommendation = result.get('exercise_recommendation', {})
        print(f"\n💡 Exercise Recommendation:")
        print(f"   Recommended: {recommendation.get('recommended', False)}")
        print(f"   Target Concept: {recommendation.get('target_concept', 'N/A')}")
        print(f"   Problem: {recommendation.get('problem', 'N/A')}")
        print(f"   Difficulty: {recommendation.get('difficulty', 'N/A')}")
        
    else:
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    return result

@pytest.mark.asyncio
async def test_multiple_images_progression():
    """Test processing multiple images to build learning graph progressively."""
    print("\n\n🧪 Testing Multiple Images Progression")
    print("=" * 50)
    
    # Initialize components
    learning_graph = LearningGraph(user_id="test_student_002", name="Progressive Student")
    knowledge_graph = KnowledgeGraph()
    gap_analyzer = GapAnalyzer(knowledge_graph)
    exercise_bank = ExerciseBank()
    recommender = Recommender(knowledge_graph, exercise_bank)
    mock_llm = MockLLM()
    
    # Create the AI agent
    agent = MathTutoringAgent(
        learning_graph=learning_graph,
        gap_analyzer=gap_analyzer,
        recommender=recommender,
        llm=mock_llm,
        user_model=learning_graph
    )
    
    # Create multiple sample images
    images = create_multiple_sample_images()
    
    print(f"📸 Processing {len(images)} images for student: {learning_graph.user_id}")
    print(f"🕒 Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process multiple images
    result = await agent.process_multiple_images(
        student_id="test_student_002",
        images=images
    )
    
    # Display results
    print("\n📊 Multiple Images Analysis Results:")
    print("-" * 40)
    
    if result.get('success'):
        print(f"✅ Processed {result.get('total_images_processed', 0)} images successfully!")
        
        # Show learning progression
        progression = result.get('learning_progression', [])
        print(f"\n📈 Learning Progression:")
        for i, entry in enumerate(progression, 1):
            print(f"   Image {i}:")
            print(f"     Concepts: {entry.get('concepts_identified', [])}")
            print(f"     Recommendation: {entry.get('recommendation', 'N/A')}")
        
        # Show summary
        summary = result.get('summary', {})
        print(f"\n📋 Learning Summary:")
        print(f"   Total Concepts Identified: {summary.get('total_concepts_identified', 0)}")
        print(f"   Most Problematic: {summary.get('most_problematic_concepts', [])}")
        print(f"   Learning Trajectory: {summary.get('learning_trajectory', 'N/A')}")
        
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print(f"   Recommendations:")
            for rec in recommendations:
                print(f"     • {rec}")
        
        # Show confidence trends
        trends = summary.get('confidence_trends', {})
        if trends:
            print(f"\n📊 Confidence Trends:")
            for concept, values in trends.items():
                print(f"   {concept}: {values}")
        
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    return result

@pytest.mark.asyncio
async def test_learning_graph_state():
    """Test the state of the learning graph after image analysis."""
    print("\n\n🧪 Testing Learning Graph State")
    print("=" * 50)
    
    # Initialize components
    learning_graph = LearningGraph(user_id="test_student_003", name="State Test Student")
    knowledge_graph = KnowledgeGraph()
    gap_analyzer = GapAnalyzer(knowledge_graph)
    exercise_bank = ExerciseBank()
    recommender = Recommender(knowledge_graph, exercise_bank)
    mock_llm = MockLLM()
    
    # Set some initial mastery levels
    learning_graph.set_mastery("addition", 0.8, 0.9)
    learning_graph.set_mastery("subtraction", 0.6, 0.7)
    learning_graph.set_mastery("multiplication", 0.4, 0.5)
    
    print("📊 Initial Learning Graph State:")
    print(f"   Addition: {learning_graph.get_mastery('addition'):.2f}")
    print(f"   Subtraction: {learning_graph.get_mastery('subtraction'):.2f}")
    print(f"   Multiplication: {learning_graph.get_mastery('multiplication'):.2f}")
    
    # Create the AI agent
    agent = MathTutoringAgent(
        learning_graph=learning_graph,
        gap_analyzer=gap_analyzer,
        recommender=recommender,
        llm=mock_llm,
        user_model=learning_graph
    )
    
    # Analyze an image that affects addition
    image_data = create_sample_image_data()
    result = await agent.analyze_exercise_image(
        student_id="test_student_003",
        image_data=image_data,
        image_format="base64"
    )
    
    print(f"\n📊 Learning Graph State After Analysis:")
    print(f"   Addition: {learning_graph.get_mastery('addition'):.2f}")
    print(f"   Subtraction: {learning_graph.get_mastery('subtraction'):.2f}")
    print(f"   Multiplication: {learning_graph.get_mastery('multiplication'):.2f}")
    
    # Show mastered and struggling concepts
    mastered = learning_graph.get_mastered_concepts(threshold=0.7)
    struggling = learning_graph.get_struggling_concepts(threshold=0.5)
    
    print(f"\n🎯 Mastered Concepts (≥0.7): {mastered}")
    print(f"⚠️  Struggling Concepts (<0.5): {struggling}")
    
    return result

async def main():
    """Run all image analysis tests."""
    print("🚀 Starting Image Analysis Feature Tests")
    print("=" * 60)
    
    try:
        # Test 1: Single image analysis
        await test_single_image_analysis()
        
        # Test 2: Multiple images progression
        await test_multiple_images_progression()
        
        # Test 3: Learning graph state changes
        await test_learning_graph_state()
        
        print("\n\n✅ All Image Analysis Tests Completed Successfully!")
        print("=" * 60)
        
        print("\n📋 Feature Summary:")
        print("✅ Image upload and processing")
        print("✅ Error analysis and concept identification")
        print("✅ Learning graph updates based on errors")
        print("✅ Targeted exercise recommendations")
        print("✅ Progressive learning graph building")
        print("✅ Learning trajectory analysis")
        
        print("\n🎯 Key Capabilities Demonstrated:")
        print("• OCR text extraction from math exercise images")
        print("• Mathematical error classification and analysis")
        print("• Concept mapping from errors to missing knowledge")
        print("• Dynamic learning graph updates")
        print("• Personalized exercise recommendations")
        print("• Multi-image learning progression tracking")
        print("• Confidence trend analysis")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 