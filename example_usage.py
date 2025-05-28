#!/usr/bin/env python3
"""
Simple Example: Using Enhanced Image Recognition Agent

This example shows how to use the Enhanced Image Recognition Agent to:
- Recognize images containing math exercises
- Identify questions and user answers from images
- Extract mathematical expressions and provide analysis

Usage:
    python example_usage.py
"""

import asyncio
import base64
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the standalone test functionality (works without dependencies)
from test_image_recognition_standalone import StandaloneImageRecognizer


async def example_basic_usage():
    """Basic example of using the image recognition functionality."""
    print("🚀 Enhanced Image Recognition Agent - Simple Example")
    print("=" * 60)
    
    # Create the recognizer
    recognizer = StandaloneImageRecognizer()
    
    print(f"📋 Available capabilities: {recognizer.capabilities}")
    print()
    
    # Example 1: Math homework with questions and student answers
    print("📝 Example 1: Math Homework Analysis")
    print("-" * 40)
    
    homework_text = """
    Math Homework - Chapter 3: Algebra
    
    Question 1: Solve for x in the equation 2x + 8 = 20
    Student answer: x = 6
    Work shown: 2x = 12, x = 6
    
    Question 2: What is the value of y when 3y - 5 = 16?
    Your answer: y = 7
    Solution: 3y = 21, y = 7
    
    Question 3: Find the area of a rectangle with length 12 and width 8
    Student work: Area = 12 × 8 = 96 square units
    Final answer: 96 square units
    """
    
    result = await recognizer.analyze_text(homework_text)
    
    if result.success:
        content = result.recognized_content
        print(f"✅ Analysis successful! Confidence: {content.confidence_score:.2f}")
        print()
        
        print(f"❓ Questions identified ({len(content.questions)}):")
        for i, question in enumerate(content.questions, 1):
            print(f"   {i}. {question}")
        print()
        
        print(f"✏️  Student answers found ({len(content.user_answers)}):")
        for i, answer in enumerate(content.user_answers, 1):
            print(f"   {i}. {answer}")
        print()
        
        print(f"🧮 Mathematical expressions ({len(content.mathematical_expressions)}):")
        for i, expr in enumerate(content.mathematical_expressions[:5], 1):  # Show first 5
            print(f"   {i}. {expr}")
        if len(content.mathematical_expressions) > 5:
            print(f"   ... and {len(content.mathematical_expressions) - 5} more")
        print()
        
        print("💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print(f"❌ Analysis failed: {result.error_message}")
    
    print("\n" + "=" * 60)
    
    # Example 2: Quiz with multiple choice
    print("📝 Example 2: Quiz Analysis")
    print("-" * 40)
    
    quiz_text = """
    Math Quiz - Unit 2
    
    1) What is 15 × 4?
    a) 50  b) 60  c) 70  d) 80
    Student answer: b) 60
    
    2) Solve: x + 7 = 15
    Student work: x = 8
    
    3) Calculate: 25% of 80
    Your answer: 20
    Work: 0.25 × 80 = 20
    """
    
    result2 = await recognizer.analyze_text(quiz_text)
    
    if result2.success:
        content2 = result2.recognized_content
        print(f"✅ Quiz analysis successful! Confidence: {content2.confidence_score:.2f}")
        print()
        
        print(f"❓ Quiz questions ({len(content2.questions)}):")
        for i, question in enumerate(content2.questions, 1):
            print(f"   {i}. {question}")
        print()
        
        print(f"✏️  Student responses ({len(content2.user_answers)}):")
        for i, answer in enumerate(content2.user_answers, 1):
            print(f"   {i}. {answer}")
        print()
        
        print("💡 Analysis insights:")
        for rec in result2.recommendations:
            print(f"   • {rec}")
    
    print("\n" + "=" * 60)
    
    # Example 3: Handwritten work simulation
    print("📝 Example 3: Handwritten Work Analysis")
    print("-" * 40)
    
    handwritten_text = """
    Problem: Solve the system of equations
    2x + y = 7
    x - y = 2
    
    Student work:
    From equation 2: x = y + 2
    Substitute into equation 1: 2(y + 2) + y = 7
    2y + 4 + y = 7
    3y = 3
    y = 1
    
    Then x = 1 + 2 = 3
    
    Final answer: x = 3, y = 1
    Check: 2(3) + 1 = 7 ✓, 3 - 1 = 2 ✓
    """
    
    result3 = await recognizer.analyze_text(handwritten_text)
    
    if result3.success:
        content3 = result3.recognized_content
        print(f"✅ Handwritten work analysis successful! Confidence: {content3.confidence_score:.2f}")
        print()
        
        print(f"🔍 Mathematical expressions found ({len(content3.mathematical_expressions)}):")
        for i, expr in enumerate(content3.mathematical_expressions[:8], 1):  # Show first 8
            print(f"   {i}. {expr}")
        
        print(f"\n✏️  Student work identified ({len(content3.user_answers)}):")
        for i, answer in enumerate(content3.user_answers[:3], 1):  # Show first 3
            print(f"   {i}. {answer}")
    
    print("\n" + "=" * 60)
    print("🎉 Examples completed!")
    print()
    print("🚀 The Enhanced Image Recognition Agent can:")
    print("   📸 Process actual images (when PIL/tesseract available)")
    print("   🔍 Extract text using OCR or LLM analysis")
    print("   ❓ Identify questions in various formats")
    print("   ✏️  Recognize student answers and work")
    print("   🧮 Detect mathematical expressions")
    print("   📊 Provide confidence scores")
    print("   💡 Generate helpful recommendations")
    print()
    print("💡 To use with actual images:")
    print("   1. Install dependencies: pip install pillow pytesseract opencv-python")
    print("   2. Use the full EnhancedImageRecognitionAgent class")
    print("   3. Pass image data as base64, file path, or bytes")
    print("   4. Get structured results with questions and answers identified")


async def example_with_image_simulation():
    """Simulate processing an actual image by showing what the agent would extract."""
    print("\n" + "🖼️  Image Processing Simulation")
    print("=" * 60)
    print("Simulating analysis of a math worksheet image...")
    print()
    
    # This simulates what OCR might extract from an actual image
    simulated_ocr_text = """
    Name: Sarah Johnson                    Date: March 15, 2024
    Math Worksheet - Linear Equations
    
    Instructions: Solve each equation and show your work.
    
    1. 3x + 5 = 14
       Student work: 3x = 9
                     x = 3
    
    2. 2y - 7 = 11
       Your answer: y = 9
       Work: 2y = 18, y = 9
    
    3. 4z + 8 = 32
       Solution: z = 6
       Steps: 4z = 24, z = 6
    
    Teacher's note: Good work! All answers correct.
    """
    
    recognizer = StandaloneImageRecognizer()
    result = await recognizer.analyze_text(simulated_ocr_text)
    
    if result.success:
        content = result.recognized_content
        print(f"📊 Image Analysis Results:")
        print(f"   Confidence Score: {content.confidence_score:.2f}")
        print(f"   Processing Method: {content.processing_method}")
        print()
        
        print("🎯 Extracted Content Summary:")
        print(f"   • Questions found: {len(content.questions)}")
        print(f"   • Student answers: {len(content.user_answers)}")
        print(f"   • Math expressions: {len(content.mathematical_expressions)}")
        print()
        
        print("📝 Detailed Breakdown:")
        print("   Questions:")
        for q in content.questions:
            print(f"     - {q}")
        
        print("   Student Answers:")
        for a in content.user_answers:
            print(f"     - {a}")
        
        print("   Key Math Expressions:")
        for expr in content.mathematical_expressions[:6]:  # Show first 6
            print(f"     - {expr}")
        
        print(f"\n💡 System Recommendations:")
        for rec in result.recommendations:
            print(f"   • {rec}")
    
    print("\n✨ This demonstrates how the agent would process a real math worksheet image!")


if __name__ == "__main__":
    asyncio.run(example_basic_usage())
    asyncio.run(example_with_image_simulation()) 