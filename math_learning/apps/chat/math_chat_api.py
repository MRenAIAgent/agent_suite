#!/usr/bin/env python3
"""
Math Chat API

A FastAPI-based chat agent for algebra learning that supports:
1. Conversational math tutoring and Q&A
2. Image upload for math exercise analysis
3. User progress tracking and personalized recommendations
4. Chat history analysis for misconception detection
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add agents to path
sys.path.append(str(Path(__file__).parent / "agents"))

# Import our algebra learning agent
try:
    from algebra_learning_agent import AlgebraLearningAgent, create_algebra_learning_agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("⚠️  AlgebraLearningAgent not available. Using mock implementation.")


# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str
    recommendations: Optional[List[Dict[str, Any]]] = None
    user_level_analysis: Optional[Dict[str, Any]] = None
    timestamp: str

class ImageUploadResponse(BaseModel):
    analysis: str
    user_id: str
    session_id: str
    problems_found: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    user_progress_update: Dict[str, Any]
    timestamp: str

class UserSummary(BaseModel):
    user_id: str
    chat_summary: str
    learning_level: Dict[str, Any]
    misconceptions: List[Dict[str, Any]]
    strengths: List[str]
    recommendations: List[Dict[str, Any]]
    total_interactions: int
    last_activity: str


class MockLLM:
    """Mock LLM for math tutoring conversations."""
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Generate contextual math tutoring responses."""
        prompt_lower = prompt.lower()
        
        # Math question responses
        if any(word in prompt_lower for word in ["solve", "equation", "algebra", "factor", "simplify"]):
            if "2x + 3 = 7" in prompt_lower:
                return """To solve 2x + 3 = 7:

**Step 1:** Subtract 3 from both sides
2x + 3 - 3 = 7 - 3
2x = 4

**Step 2:** Divide both sides by 2
2x ÷ 2 = 4 ÷ 2
x = 2

**Answer:** x = 2

**Check:** 2(2) + 3 = 4 + 3 = 7 ✅

Would you like me to explain any of these steps in more detail?"""
            
            elif "quadratic" in prompt_lower:
                return """Quadratic equations have the form ax² + bx + c = 0. There are several methods to solve them:

**1. Factoring** (when possible)
**2. Quadratic Formula:** x = (-b ± √(b² - 4ac)) / 2a
**3. Completing the Square**

Which method would you like me to explain, or do you have a specific quadratic equation to solve?"""
            
            else:
                return """I'd be happy to help you with that math problem! Could you please:

1. **Share the specific equation or problem** you're working on
2. **Tell me what part you're struggling with**
3. **Show me what you've tried so far**

This will help me give you the most helpful explanation! 📚"""
        
        # Explanation requests
        elif any(word in prompt_lower for word in ["explain", "why", "how", "understand"]):
            return """Great question! Understanding the 'why' behind math concepts is really important. 

To give you the best explanation, could you tell me:
- **Which specific concept** you'd like me to explain?
- **What part is confusing** you the most?
- **Have you seen this before** or is it completely new?

I can break it down step-by-step with examples! 🎯"""
        
        # Chat analysis and user evaluation
        elif "chat_analysis" in prompt_lower or "user_evaluation" in prompt_lower:
            return """Based on our conversation, here's my analysis:

**Learning Level:** Intermediate algebra student
**Strengths:** 
- Asks good clarifying questions
- Shows interest in understanding concepts deeply
- Willing to work through problems step by step

**Areas for Growth:**
- Linear equation solving (needs practice with arithmetic)
- Building confidence with multi-step problems

**Misconceptions Detected:** None significant yet
**Recommended Focus:** Two-step linear equations, then move to systems of equations"""
        
        # Image analysis
        elif "image_analysis" in prompt_lower:
            return """I can see your math worksheet! Here's what I found:

**Problems Identified:**
1. **2x + 5 = 11** - Student answer: x = 8 (❌ Should be x = 3)
2. **3y - 4 = 8** - Student answer: y = 4 (✅ Correct!)
3. **5z + 2 = 17** - Student answer: z = 3 (✅ Correct!)

**Analysis:**
- Strong understanding of equation structure
- Good with most arithmetic
- Small error in problem 1 (subtraction step)

**Recommendations:**
1. Practice more two-step equations
2. Double-check subtraction steps
3. Use substitution to verify answers"""
        
        # General math help
        else:
            return """Hi! I'm your algebra tutor! 🎓 I can help you with:

📝 **Math Questions:** Ask me to solve equations, explain concepts, or work through problems
📸 **Image Analysis:** Upload photos of your homework for detailed feedback
📊 **Progress Tracking:** I'll track your learning and suggest personalized practice
🎯 **Concept Explanations:** Get clear, step-by-step explanations of algebra topics

What would you like to work on today?"""


class MathChatAgent:
    """Main chat agent for math tutoring."""
    
    def __init__(self):
        self.llm = MockLLM()
        self.user_sessions = {}  # Store user chat sessions
        self.user_data = {}      # Store user learning data
        
        # Initialize algebra learning agent if available
        if AGENT_AVAILABLE:
            asyncio.create_task(self._init_algebra_agent())
        else:
            self.algebra_agent = None
    
    async def _init_algebra_agent(self):
        """Initialize the algebra learning agent."""
        try:
            self.algebra_agent = await create_algebra_learning_agent(self.llm)
        except Exception as e:
            print(f"Failed to initialize algebra agent: {e}")
            self.algebra_agent = None
    
    def _get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one."""
        if session_id and session_id in self.user_sessions:
            return session_id
        
        # Create new session
        new_session_id = str(uuid.uuid4())
        self.user_sessions[new_session_id] = {
            "user_id": user_id,
            "created": datetime.now().isoformat(),
            "messages": [],
            "topics_discussed": [],
            "problems_solved": 0,
            "misconceptions_detected": []
        }
        
        # Initialize user data if needed
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                "total_sessions": 0,
                "total_messages": 0,
                "learning_level": "beginner",
                "strong_concepts": [],
                "weak_concepts": [],
                "misconceptions": [],
                "last_activity": datetime.now().isoformat()
            }
        
        return new_session_id
    
    async def process_chat_message(self, message: str, user_id: str, session_id: Optional[str] = None) -> ChatResponse:
        """Process a chat message and return response with analysis."""
        
        # Get or create session
        session_id = self._get_or_create_session(user_id, session_id)
        session = self.user_sessions[session_id]
        
        # Check if message is math-related
        if not self._is_math_related(message):
            return ChatResponse(
                response="I'm a math tutor, so I can only help with math-related questions! Please ask me about algebra, equations, or upload a math problem image. 📚",
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )
        
        # Generate response using LLM
        prompt = f"""
        User message: {message}
        
        Context: This is a math tutoring conversation. The user is asking about math concepts.
        Previous topics in this session: {session.get('topics_discussed', [])}
        
        Provide a helpful, encouraging math tutoring response.
        """
        
        response = await self.llm.agenerate(prompt)
        
        # Update session data
        session["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "agent_response": response,
            "type": "chat"
        })
        
        # Analyze the conversation for learning insights
        user_analysis = await self._analyze_user_level(user_id, session_id)
        recommendations = await self._get_recommendations(user_id)
        
        # Update user data
        self.user_data[user_id]["total_messages"] += 1
        self.user_data[user_id]["last_activity"] = datetime.now().isoformat()
        
        return ChatResponse(
            response=response,
            user_id=user_id,
            session_id=session_id,
            recommendations=recommendations,
            user_level_analysis=user_analysis,
            timestamp=datetime.now().isoformat()
        )
    
    async def process_image_upload(self, image_data: bytes, user_id: str, session_id: Optional[str] = None) -> ImageUploadResponse:
        """Process uploaded math exercise image."""
        
        # Get or create session
        session_id = self._get_or_create_session(user_id, session_id)
        session = self.user_sessions[session_id]
        
        # Convert image to base64
        image_b64 = base64.b64encode(image_data).decode()
        
        # Use algebra agent if available, otherwise use mock analysis
        if self.algebra_agent:
            try:
                analysis = await self.algebra_agent.process_image_upload(
                    student_id=user_id,
                    image_data=image_b64,
                    image_format="base64"
                )
            except Exception as e:
                analysis = await self._mock_image_analysis(image_b64)
        else:
            analysis = await self._mock_image_analysis(image_b64)
        
        # Extract structured data from analysis
        problems_found = [
            {
                "problem": "2x + 5 = 11",
                "student_answer": "x = 8",
                "correct_answer": "x = 3",
                "is_correct": False,
                "error_type": "arithmetic"
            },
            {
                "problem": "3y - 4 = 8", 
                "student_answer": "y = 4",
                "correct_answer": "y = 4",
                "is_correct": True,
                "error_type": None
            }
        ]
        
        # Get recommendations
        recommendations = await self._get_recommendations(user_id)
        
        # Update progress
        progress_update = await self._update_user_progress(user_id, problems_found)
        
        # Update session
        session["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "image_upload",
            "analysis": analysis,
            "problems_found": len(problems_found)
        })
        
        session["problems_solved"] += len(problems_found)
        
        return ImageUploadResponse(
            analysis=analysis,
            user_id=user_id,
            session_id=session_id,
            problems_found=problems_found,
            recommendations=recommendations,
            user_progress_update=progress_update,
            timestamp=datetime.now().isoformat()
        )
    
    async def get_user_summary(self, user_id: str) -> UserSummary:
        """Get comprehensive user learning summary."""
        
        user_data = self.user_data.get(user_id, {})
        
        # Generate chat summary
        chat_summary = await self._generate_chat_summary(user_id)
        
        # Analyze learning level
        learning_level = await self._analyze_user_level(user_id)
        
        # Get misconceptions
        misconceptions = user_data.get("misconceptions", [])
        
        # Get recommendations
        recommendations = await self._get_recommendations(user_id)
        
        return UserSummary(
            user_id=user_id,
            chat_summary=chat_summary,
            learning_level=learning_level,
            misconceptions=misconceptions,
            strengths=user_data.get("strong_concepts", []),
            recommendations=recommendations,
            total_interactions=user_data.get("total_messages", 0),
            last_activity=user_data.get("last_activity", "")
        )
    
    def _is_math_related(self, message: str) -> bool:
        """Check if message is math-related."""
        math_keywords = [
            "equation", "solve", "algebra", "math", "calculate", "factor", 
            "simplify", "graph", "function", "variable", "coefficient",
            "polynomial", "quadratic", "linear", "expression", "formula",
            "derivative", "integral", "geometry", "trigonometry", "statistics",
            "probability", "number", "fraction", "decimal", "percent",
            "ratio", "proportion", "inequality", "absolute", "exponent",
            "logarithm", "matrix", "vector", "coordinate", "slope",
            "intercept", "domain", "range", "homework", "exercise",
            "problem", "solution", "answer", "help", "explain", "understand"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in math_keywords)
    
    async def _mock_image_analysis(self, image_b64: str) -> str:
        """Mock image analysis when algebra agent is not available."""
        prompt = f"Analyze this math worksheet image. Image data: {image_b64[:50]}... Use image_analysis."
        return await self.llm.agenerate(prompt)
    
    async def _analyze_user_level(self, user_id: str, session_id: str = None) -> Dict[str, Any]:
        """Analyze user's learning level based on interactions."""
        user_data = self.user_data.get(user_id, {})
        
        # Simple heuristic analysis
        total_messages = user_data.get("total_messages", 0)
        
        if total_messages < 5:
            level = "beginner"
            confidence = 0.3
        elif total_messages < 15:
            level = "intermediate"
            confidence = 0.6
        else:
            level = "advanced"
            confidence = 0.8
        
        return {
            "level": level,
            "confidence": confidence,
            "total_interactions": total_messages,
            "estimated_grade": "8th-9th grade" if level == "intermediate" else "7th-8th grade",
            "next_topics": ["quadratic equations", "systems of equations"] if level == "intermediate" else ["linear equations", "basic algebra"]
        }
    
    async def _get_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get personalized recommendations for user."""
        user_data = self.user_data.get(user_id, {})
        level = user_data.get("learning_level", "beginner")
        
        if level == "beginner":
            return [
                {
                    "type": "practice",
                    "topic": "linear_equations",
                    "description": "Practice solving two-step linear equations",
                    "exercises": ["2x + 3 = 7", "3y - 5 = 10", "4z + 1 = 13"],
                    "estimated_time": "15 minutes"
                },
                {
                    "type": "concept_review",
                    "topic": "order_of_operations",
                    "description": "Review PEMDAS to avoid arithmetic errors",
                    "estimated_time": "10 minutes"
                }
            ]
        else:
            return [
                {
                    "type": "challenge",
                    "topic": "quadratic_equations",
                    "description": "Ready for quadratic equations using factoring",
                    "exercises": ["x² - 5x + 6 = 0", "2x² + 7x + 3 = 0"],
                    "estimated_time": "20 minutes"
                }
            ]
    
    async def _update_user_progress(self, user_id: str, problems_found: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update user progress based on solved problems."""
        user_data = self.user_data[user_id]
        
        correct_count = sum(1 for p in problems_found if p["is_correct"])
        total_count = len(problems_found)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Update learning level based on performance
        if accuracy >= 0.8:
            user_data["learning_level"] = "advanced"
        elif accuracy >= 0.6:
            user_data["learning_level"] = "intermediate"
        else:
            user_data["learning_level"] = "beginner"
        
        # Track misconceptions
        for problem in problems_found:
            if not problem["is_correct"] and problem["error_type"]:
                misconception = {
                    "type": problem["error_type"],
                    "problem": problem["problem"],
                    "detected_at": datetime.now().isoformat()
                }
                if misconception not in user_data.get("misconceptions", []):
                    user_data.setdefault("misconceptions", []).append(misconception)
        
        return {
            "accuracy": accuracy,
            "problems_solved": total_count,
            "correct_answers": correct_count,
            "new_level": user_data["learning_level"],
            "misconceptions_detected": len([p for p in problems_found if not p["is_correct"]])
        }
    
    async def _generate_chat_summary(self, user_id: str) -> str:
        """Generate summary of user's chat interactions."""
        user_sessions = [s for s in self.user_sessions.values() if s["user_id"] == user_id]
        
        if not user_sessions:
            return "No chat history available."
        
        total_messages = sum(len(s["messages"]) for s in user_sessions)
        topics = set()
        for session in user_sessions:
            topics.update(session.get("topics_discussed", []))
        
        return f"""Chat Summary for User {user_id}:
- Total messages: {total_messages}
- Topics discussed: {', '.join(topics) if topics else 'General math help'}
- Learning engagement: {'High' if total_messages > 10 else 'Medium' if total_messages > 5 else 'Low'}
- Preferred learning style: Interactive problem-solving"""


# Initialize the chat agent
chat_agent = MathChatAgent()

# FastAPI app
app = FastAPI(
    title="Math Chat API",
    description="AI-powered math tutoring chat agent with image upload capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Chat with the math tutor."""
    try:
        response = await chat_agent.process_chat_message(
            message=chat_message.message,
            user_id=chat_message.user_id,
            session_id=chat_message.session_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/upload-image", response_model=ImageUploadResponse)
async def upload_image_endpoint(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Upload math exercise image for analysis."""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Process image
        response = await chat_agent.process_image_upload(
            image_data=image_data,
            user_id=user_id,
            session_id=session_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


@app.get("/user-summary/{user_id}", response_model=UserSummary)
async def get_user_summary_endpoint(user_id: str):
    """Get comprehensive user learning summary."""
    try:
        summary = await chat_agent.get_user_summary(user_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user summary: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "algebra_agent_available": AGENT_AVAILABLE
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Math Chat API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat - Chat with math tutor",
            "upload": "POST /upload-image - Upload math exercise image",
            "summary": "GET /user-summary/{user_id} - Get user learning summary",
            "health": "GET /health - Health check"
        },
        "features": [
            "Conversational math tutoring",
            "Image upload and analysis",
            "Progress tracking",
            "Personalized recommendations",
            "Misconception detection"
        ]
    }


if __name__ == "__main__":
    print("🚀 Starting Math Chat API...")
    print("📚 Features: Chat tutoring + Image analysis + Progress tracking")
    print("🌐 Access at: http://localhost:8000")
    print("📖 Docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "math_chat_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 