#!/usr/bin/env python3
"""
Simple test for AlgebraLearningAgent without external dependencies.
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

# Add the agents directory to the path
sys.path.append(str(Path(__file__).parent / "agents"))

class MockLLM:
    """Mock LLM for testing."""
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        if "image_recognition" in prompt.lower():
            return """I can see a math worksheet with these problems:

1. Solve: 2x + 3 = 7
   Student answer: x = 5 (incorrect, should be x = 2)
   
2. Simplify: 3x + 2x  
   Student answer: 5x (correct!)
   
3. Factor: x² - 4
   Student answer: (x-2)(x+2) (correct!)

The student shows good understanding of combining like terms and factoring, but made an arithmetic error in the linear equation."""
        
        elif "learning_graph" in prompt.lower():
            return """Updated student progress:
- Linear equations: 60% mastery (needs practice with arithmetic)
- Like terms: 90% mastery (strong)
- Factoring: 80% mastery (good)

Recommendation: Practice more two-step equations focusing on careful arithmetic."""
        
        else:
            return "I'm ready to help with algebra learning! Upload an image or ask me about math concepts."

async def test_algebra_agent():
    """Test the AlgebraLearningAgent."""
    print("🧪 Testing AlgebraLearningAgent")
    print("=" * 40)
    
    try:
        # Import the agent
        from algebra_learning_agent import create_algebra_learning_agent
        print("✅ Successfully imported AlgebraLearningAgent")
        
        # Create mock LLM
        llm = MockLLM()
        
        # Create the agent
        agent = await create_algebra_learning_agent(llm)
        print(f"✅ Created agent with {len(agent.tools)} tools")
        
        # Test image processing
        print("\n📸 Testing image processing...")
        student_id = "test_student"
        sample_image = base64.b64encode(b"Sample math worksheet").decode()
        
        response = await agent.process_image_upload(
            student_id=student_id,
            image_data=sample_image
        )
        print(f"📝 Response: {response[:200]}...")
        
        # Test dashboard
        print("\n📊 Testing student dashboard...")
        dashboard = await agent.get_student_dashboard(student_id)
        print(f"📈 Dashboard keys: {list(dashboard.keys())}")
        
        # Test practice session
        print("\n🏃‍♂️ Testing practice session...")
        practice_response = await agent.practice_session(
            student_id=student_id,
            focus_concept="linear_equations"
        )
        print(f"🎯 Practice response: {practice_response[:200]}...")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_algebra_agent()) 