#!/usr/bin/env python3
"""
Algebra Learning Agent Demo

This demo shows how to use the AlgebraLearningAgent to:
1. Process uploaded images of math problems
2. Analyze student work and identify misconceptions
3. Build personal learning graphs
4. Provide personalized recommendations
5. Track learning progress over time

Usage:
    python demos/algebra_learning_agent_demo.py
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from agents.algebra_learning_agent import AlgebraLearningAgent, create_algebra_learning_agent
    print("✅ Successfully imported AlgebraLearningAgent")
except ImportError as e:
    print(f"❌ Failed to import AlgebraLearningAgent: {e}")
    sys.exit(1)


class MockLLM:
    """Mock LLM for demonstration purposes."""
    
    def __init__(self):
        self.name = "mock_gpt4"
        self.call_count = 0
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response based on the prompt content."""
        self.call_count += 1
        
        # Simulate different responses based on prompt content
        if "image_recognition" in prompt.lower():
            return """I can see this is a math worksheet with algebra problems. Let me analyze what I found:

**Problems Identified:**
1. Solve for x: 2x + 3 = 7
   - Student's answer: x = 5
   - Correct answer: x = 2
   
2. Simplify: 3x + 2x
   - Student's answer: 5x
   - This is correct!

3. Factor: x² - 4
   - Student's answer: (x-2)(x+2)
   - This is correct!

**Analysis:**
The student shows good understanding of combining like terms and factoring, but made an arithmetic error in solving the linear equation."""
        
        elif "concept_analysis" in prompt.lower():
            return """Based on the student's work, I can identify the following:

**Misconception Detected:**
- In problem 1 (2x + 3 = 7), the student likely made an arithmetic error
- They may have incorrectly calculated 7 - 3 = 5 instead of 4, or made an error in division

**Strengths:**
- Correctly combined like terms (3x + 2x = 5x)
- Successfully factored difference of squares (x² - 4)
- Shows understanding of algebraic manipulation

**Recommendation:**
Focus on careful arithmetic when solving linear equations. Practice more two-step equation problems."""
        
        elif "learning_graph" in prompt.lower():
            return """I've updated the student's learning profile:

**Progress Update:**
- Linear equations: Mastery level updated to 0.6 (was 0.4)
- Like terms: Mastery level 0.9 (strong)
- Factoring: Mastery level 0.8 (good)

**Recommendations:**
1. Practice 5 more linear equation problems focusing on arithmetic accuracy
2. Review order of operations
3. Continue with quadratic equations when linear equation mastery improves"""
        
        else:
            return f"""I understand you want me to help with algebra learning. I can:

1. **Analyze Images**: Upload photos of math problems and I'll identify questions and student answers
2. **Track Progress**: I'll build a personal learning graph showing your strengths and areas to improve
3. **Provide Feedback**: Get specific, encouraging feedback on your work
4. **Recommend Practice**: I'll suggest personalized exercises based on your learning gaps

What would you like to work on today?"""
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Alternative method name for compatibility."""
        return await self.agenerate(prompt, **kwargs)


def create_sample_image_data() -> str:
    """Create a sample base64 image for demonstration."""
    # This would normally be actual image data
    # For demo purposes, we'll use a placeholder
    sample_text = "Sample math worksheet: 1) 2x + 3 = 7, Student answer: x = 5"
    return base64.b64encode(sample_text.encode()).decode()


async def demo_basic_usage():
    """Demonstrate basic usage of the Algebra Learning Agent."""
    print("\n🎯 Demo 1: Basic Agent Creation and Usage")
    print("=" * 50)
    
    # Create mock LLM
    llm = MockLLM()
    
    # Create the algebra learning agent
    agent = await create_algebra_learning_agent(llm)
    
    print(f"✅ Created AlgebraLearningAgent with {len(agent.tools)} tools:")
    for i, tool in enumerate(agent.tools):
        print(f"   {i+1}. {tool.name}: {tool.description.strip()[:60]}...")
    
    return agent


async def demo_image_processing(agent: AlgebraLearningAgent):
    """Demonstrate image processing capabilities."""
    print("\n📸 Demo 2: Image Processing and Analysis")
    print("=" * 50)
    
    # Simulate uploading an image
    student_id = "demo_student_123"
    image_data = create_sample_image_data()
    
    print(f"📤 Processing image upload for student: {student_id}")
    print("🔍 Analyzing image content...")
    
    try:
        # Process the image
        response = await agent.process_image_upload(
            student_id=student_id,
            image_data=image_data,
            image_format="base64"
        )
        
        print("✅ Image processing completed!")
        print(f"📝 Agent Response:\n{response}")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        # Fallback to direct tool usage
        print("🔄 Trying direct tool usage...")
        
        # Use image recognition tool directly
        image_tool = agent.tools[0]  # ImageRecognitionTool
        result = await image_tool.execute(
            image_data=image_data,
            image_format="base64"
        )
        print(f"📊 Image Recognition Result: {json.dumps(result, indent=2)}")


async def demo_learning_progress(agent: AlgebraLearningAgent):
    """Demonstrate learning progress tracking."""
    print("\n📈 Demo 3: Learning Progress Tracking")
    print("=" * 50)
    
    student_id = "demo_student_123"
    
    # Get student dashboard
    print("📊 Generating student dashboard...")
    dashboard = await agent.get_student_dashboard(student_id)
    
    print("✅ Student Dashboard Generated:")
    print(json.dumps(dashboard, indent=2))
    
    # Simulate updating mastery
    print("\n🔄 Updating student mastery levels...")
    learning_tool = agent.tools[1]  # LearningGraphTool
    
    # Update mastery for different concepts
    concepts_to_update = [
        ("linear_equations", 0.6, {"problem_type": "two_step", "accuracy": 0.6}),
        ("like_terms", 0.9, {"problem_type": "combining", "accuracy": 0.9}),
        ("factoring", 0.8, {"problem_type": "difference_of_squares", "accuracy": 0.8})
    ]
    
    for concept, mastery, evidence in concepts_to_update:
        result = await learning_tool.execute(
            action="update_mastery",
            student_id=student_id,
            concept=concept,
            mastery_level=mastery,
            evidence=evidence
        )
        print(f"   ✅ {concept}: {result.get('message', 'Updated')}")
    
    # Get updated progress
    print("\n📊 Updated Progress:")
    progress = await learning_tool.execute(action="get_progress", student_id=student_id)
    print(json.dumps(progress, indent=2))


async def demo_personalized_recommendations(agent: AlgebraLearningAgent):
    """Demonstrate personalized exercise recommendations."""
    print("\n🎯 Demo 4: Personalized Recommendations")
    print("=" * 50)
    
    student_id = "demo_student_123"
    
    # Get knowledge gaps
    learning_tool = agent.tools[1]  # LearningGraphTool
    gaps = await learning_tool.execute(action="analyze_gaps", student_id=student_id)
    
    print("🔍 Knowledge Gaps Analysis:")
    print(json.dumps(gaps, indent=2))
    
    # Get general recommendations
    print("\n💡 General Recommendations:")
    recommendations = await learning_tool.execute(
        action="get_recommendations",
        student_id=student_id
    )
    print(json.dumps(recommendations, indent=2))
    
    # Get focused recommendations
    print("\n🎯 Focused Recommendations (Linear Equations):")
    focused_recs = await learning_tool.execute(
        action="get_recommendations",
        student_id=student_id,
        focus_area="linear_equations",
        difficulty="medium"
    )
    print(json.dumps(focused_recs, indent=2))


async def demo_practice_session(agent: AlgebraLearningAgent):
    """Demonstrate a practice session."""
    print("\n🏃‍♂️ Demo 5: Practice Session")
    print("=" * 50)
    
    student_id = "demo_student_123"
    
    print("🎯 Starting adaptive practice session...")
    try:
        response = await agent.practice_session(
            student_id=student_id,
            focus_concept="linear_equations",
            difficulty="adaptive"
        )
        
        print("✅ Practice Session Response:")
        print(response)
        
    except Exception as e:
        print(f"❌ Error in practice session: {e}")
        print("🔄 Using fallback approach...")
        
        # Fallback: use tools directly
        learning_tool = agent.tools[1]
        recommendations = await learning_tool.execute(
            action="get_recommendations",
            student_id=student_id,
            focus_area="linear_equations"
        )
        
        print("📚 Practice Recommendations:")
        for rec in recommendations.get("recommendations", []):
            print(f"   📝 {rec.get('type', 'Practice')}: {rec.get('concept', 'General')}")
            print(f"      Exercises: {', '.join(rec.get('exercises', []))}")
            print(f"      Time: {rec.get('estimated_time', 'N/A')}")


async def demo_concept_analysis(agent: AlgebraLearningAgent):
    """Demonstrate concept analysis capabilities."""
    print("\n🧠 Demo 6: Concept Analysis")
    print("=" * 50)
    
    concept_tool = agent.tools[2]  # ConceptAnalysisTool
    
    # Test misconception analysis
    print("🔍 Analyzing misconceptions...")
    misconception_result = await concept_tool.execute(
        analysis_type="misconception",
        student_work="x = 5",
        correct_answer="x = 2",
        concept_area="linear_equations"
    )
    print("📊 Misconception Analysis:")
    print(json.dumps(misconception_result, indent=2))
    
    # Test mastery analysis
    print("\n📈 Analyzing concept mastery...")
    mastery_result = await concept_tool.execute(
        analysis_type="concept_mastery",
        student_work="2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2",
        concept_area="linear_equations"
    )
    print("📊 Mastery Analysis:")
    print(json.dumps(mastery_result, indent=2))
    
    # Test error pattern analysis
    print("\n🔍 Analyzing error patterns...")
    error_result = await concept_tool.execute(
        analysis_type="error_pattern",
        student_work="x = 5 = 2 = wrong",
        concept_area="linear_equations"
    )
    print("📊 Error Pattern Analysis:")
    print(json.dumps(error_result, indent=2))


async def main():
    """Run all demos."""
    print("🚀 Algebra Learning Agent Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive capabilities of the")
    print("AlgebraLearningAgent for personalized math learning.")
    print("=" * 60)
    
    try:
        # Demo 1: Basic usage
        agent = await demo_basic_usage()
        
        # Demo 2: Image processing
        await demo_image_processing(agent)
        
        # Demo 3: Learning progress
        await demo_learning_progress(agent)
        
        # Demo 4: Recommendations
        await demo_personalized_recommendations(agent)
        
        # Demo 5: Practice session
        await demo_practice_session(agent)
        
        # Demo 6: Concept analysis
        await demo_concept_analysis(agent)
        
        print("\n🎉 Demo Complete!")
        print("=" * 60)
        print("The AlgebraLearningAgent successfully demonstrated:")
        print("✅ Image recognition and analysis")
        print("✅ Learning progress tracking")
        print("✅ Personalized recommendations")
        print("✅ Concept analysis and misconception detection")
        print("✅ Practice session management")
        print("✅ Student dashboard generation")
        
        # Show final student state
        print("\n📊 Final Student State:")
        dashboard = await agent.get_student_dashboard("demo_student_123")
        progress = dashboard.get("progress", {}).get("progress", {})
        print(f"   📈 Total Concepts: {progress.get('total_concepts', 0)}")
        print(f"   ✅ Mastered: {progress.get('mastered_concepts', 0)}")
        print(f"   🔄 In Progress: {progress.get('in_progress_concepts', 0)}")
        print(f"   ⚠️  Struggling: {progress.get('struggling_concepts', 0)}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 