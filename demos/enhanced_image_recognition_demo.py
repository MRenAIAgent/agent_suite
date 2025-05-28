"""
Enhanced Image Recognition Agent Demo

This demo shows how to use the Enhanced Image Recognition Agent to:
1. Recognize images containing math exercises
2. Identify questions and user answers from images
3. Extract mathematical expressions
4. Provide confidence scores and recommendations

The agent can handle various image types:
- Handwritten math work
- Printed exercises and worksheets
- Digital math content
- Quiz and homework images
"""

import asyncio
import base64
import json
from pathlib import Path
import sys

# Add the parent directory to the path to import the agent
sys.path.append(str(Path(__file__).parent.parent))

from agents.enhanced_image_recognition_agent import (
    EnhancedImageRecognitionAgent,
    recognize_math_image
)


class MockLLM:
    """Mock LLM for demonstration purposes."""
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a mock response based on the prompt."""
        if "analyze this image" in prompt.lower():
            return """
            Based on the image analysis, I can identify the following content:
            
            Questions:
            1. What is the value of x in the equation 2x + 5 = 15?
            2. Solve for y: 3y - 7 = 14
            3. Find the area of a rectangle with length 8 and width 5
            
            Student Answers:
            1. x = 5 (showing work: 2x = 10, x = 5)
            2. y = 7 (showing work: 3y = 21, y = 7)
            3. Area = 40 square units (showing work: 8 × 5 = 40)
            
            Mathematical Expressions:
            - 2x + 5 = 15
            - 2x = 10
            - x = 5
            - 3y - 7 = 14
            - 3y = 21
            - y = 7
            - Area = length × width = 8 × 5 = 40
            
            The student's work appears to be correct with clear step-by-step solutions.
            """
        return "Mock LLM response for demonstration"


def create_sample_base64_image() -> str:
    """Create a sample base64 image for demonstration."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Create a sample image with math content
        img = Image.new('RGB', (600, 400), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some sample math content
        text_content = [
            "Math Homework - Chapter 3",
            "",
            "1. What is 2x + 5 = 15?",
            "   Student answer: x = 5",
            "",
            "2. Solve: 3y - 7 = 14",
            "   Your work: y = 7",
            "",
            "3. Area of rectangle (8×5)?",
            "   Answer: 40 square units"
        ]
        
        y_position = 30
        for line in text_content:
            draw.text((20, y_position), line, fill='black')
            y_position += 25
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except ImportError:
        # Return a minimal base64 string if PIL not available
        print("⚠️  PIL not available, using minimal test image")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


async def demo_basic_usage():
    """Demonstrate basic usage of the Enhanced Image Recognition Agent."""
    print("🚀 Enhanced Image Recognition Agent - Basic Usage Demo")
    print("=" * 60)
    
    # Create the agent with mock LLM
    mock_llm = MockLLM()
    agent = EnhancedImageRecognitionAgent(llm=mock_llm)
    
    print(f"✅ Agent created: {agent.name}")
    print(f"📋 Description: {agent.description}")
    print(f"🔧 Capabilities: {agent.capabilities}")
    print()
    
    # Create sample image
    sample_image = create_sample_base64_image()
    print("📸 Created sample math homework image")
    print()
    
    # Analyze the image
    print("🔍 Analyzing image...")
    result = await agent.recognize_image(
        image_data=sample_image,
        image_format="base64",
        processing_options={
            'preprocessing_type': 'standard',
            'extract_handwriting': True
        }
    )
    
    # Display results
    print("📊 Analysis Results:")
    print("-" * 40)
    print(f"Success: {result.success}")
    
    if result.success and result.recognized_content:
        content = result.recognized_content
        print(f"Confidence Score: {content.confidence_score:.2f}")
        print(f"Processing Method: {content.processing_method}")
        print()
        
        print("❓ Questions Found:")
        for i, question in enumerate(content.questions, 1):
            print(f"  {i}. {question}")
        print()
        
        print("✏️  User Answers Found:")
        for i, answer in enumerate(content.user_answers, 1):
            print(f"  {i}. {answer}")
        print()
        
        print("✅ Correct Answers Found:")
        for i, answer in enumerate(content.correct_answers, 1):
            print(f"  {i}. {answer}")
        print()
        
        print("🧮 Mathematical Expressions:")
        for i, expr in enumerate(content.mathematical_expressions, 1):
            print(f"  {i}. {expr}")
        print()
        
        print("💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print(f"❌ Analysis failed: {result.error_message}")
    
    print()
    return result


async def demo_different_scenarios():
    """Demonstrate different usage scenarios."""
    print("🎯 Enhanced Image Recognition Agent - Scenario Demos")
    print("=" * 60)
    
    mock_llm = MockLLM()
    agent = EnhancedImageRecognitionAgent(llm=mock_llm)
    
    scenarios = [
        {
            'name': 'Math Homework',
            'text': """
            Math Homework - Chapter 5
            
            1. What is 15 + 27?
            Student answer: 42
            
            2. Solve for x: 3x - 9 = 15
            Your work: 3x = 24, x = 8
            
            3. Find the area of a circle with radius 5
            Answer: A = πr² = π(5)² = 25π
            """,
            'options': {'preprocessing_type': 'standard'}
        },
        {
            'name': 'Handwritten Work',
            'text': """
            Problem: 2x + 5 = 13
            
            Step 1: 2x = 13 - 5
            Step 2: 2x = 8
            Step 3: x = 4
            
            Check: 2(4) + 5 = 8 + 5 = 13 ✓
            """,
            'options': {'preprocessing_type': 'handwriting'}
        },
        {
            'name': 'Quiz Questions',
            'text': """
            Quiz: Algebra Basics
            
            1. What is the value of x in 2x + 3 = 11?
            a) 3  b) 4  c) 5  d) 6
            Student answer: b) 4
            
            2. Simplify: 3(x + 2) - 2x
            Student work: 3x + 6 - 2x = x + 6
            """,
            'options': {'preprocessing_type': 'standard'}
        }
    ]
    
    for scenario in scenarios:
        print(f"📝 Scenario: {scenario['name']}")
        print("-" * 30)
        
        # Simulate text extraction for this scenario
        content = await agent._parse_and_identify_content(scenario['text'], scenario['options'])
        
        print(f"Questions found: {len(content.questions)}")
        print(f"User answers found: {len(content.user_answers)}")
        print(f"Math expressions found: {len(content.mathematical_expressions)}")
        print(f"Confidence score: {content.confidence_score:.2f}")
        
        if content.questions:
            print("Sample question:", content.questions[0][:50] + "..." if len(content.questions[0]) > 50 else content.questions[0])
        
        print()


async def demo_standalone_function():
    """Demonstrate the standalone recognize_math_image function."""
    print("🔧 Standalone Function Demo")
    print("=" * 40)
    
    # Create sample image
    sample_image = create_sample_base64_image()
    
    # Use the standalone function
    result = await recognize_math_image(
        image_data=sample_image,
        image_format="base64",
        llm=MockLLM(),
        processing_options={'preprocessing_type': 'standard'}
    )
    
    print(f"Standalone function result: {result.success}")
    if result.recognized_content:
        print(f"Questions: {len(result.recognized_content.questions)}")
        print(f"Confidence: {result.recognized_content.confidence_score:.2f}")
    print()


async def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("⚠️  Error Handling Demo")
    print("=" * 30)
    
    agent = EnhancedImageRecognitionAgent(llm=MockLLM())
    
    # Test with invalid image data
    result = await agent.recognize_image("invalid_base64_data", "base64")
    
    print(f"Invalid data handling: {result.success}")
    print(f"Error message: {result.error_message}")
    print(f"Recommendations: {result.recommendations}")
    print()


def demo_data_structures():
    """Demonstrate the data structures used by the agent."""
    print("📊 Data Structures Demo")
    print("=" * 30)
    
    from agents.enhanced_image_recognition_agent import RecognizedContent, ImageAnalysisResult
    
    # Create sample recognized content
    content = RecognizedContent(
        questions=["What is 2+2?", "Solve x+3=7"],
        user_answers=["4", "x=4"],
        correct_answers=["4", "x=4"],
        mathematical_expressions=["2+2=4", "x+3=7", "x=4"],
        text_content="Sample math content",
        confidence_score=0.85,
        processing_method="ocr_with_llm",
        timestamp="2024-01-01T12:00:00"
    )
    
    # Create analysis result
    result = ImageAnalysisResult(
        success=True,
        recognized_content=content,
        error_message=None,
        processing_details={
            'image_format': 'base64',
            'processing_method': 'ocr_with_llm',
            'text_length': 50
        },
        recommendations=["Content recognized successfully", "High confidence score"]
    )
    
    # Show serialization
    print("RecognizedContent as dict:")
    print(json.dumps(content.to_dict(), indent=2))
    print()
    
    print("ImageAnalysisResult as dict:")
    print(json.dumps(result.to_dict(), indent=2))
    print()


async def main():
    """Run all demos."""
    print("🎉 Enhanced Image Recognition Agent - Complete Demo Suite")
    print("=" * 70)
    print()
    
    try:
        # Run all demos
        await demo_basic_usage()
        await demo_different_scenarios()
        await demo_standalone_function()
        await demo_error_handling()
        demo_data_structures()
        
        print("✅ All demos completed successfully!")
        print()
        print("🔍 Key Features Demonstrated:")
        print("  ✓ Image recognition and text extraction")
        print("  ✓ Question and answer identification")
        print("  ✓ Mathematical expression parsing")
        print("  ✓ Confidence scoring")
        print("  ✓ Error handling and recommendations")
        print("  ✓ Multiple processing scenarios")
        print("  ✓ Data structure serialization")
        print()
        print("🚀 The Enhanced Image Recognition Agent is ready to use!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the complete demo
    asyncio.run(main()) 