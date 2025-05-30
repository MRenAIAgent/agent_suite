# 🎓 Math Chat API

A FastAPI-based conversational algebra tutor that combines chat functionality with image analysis for personalized math learning.

## ✨ Features

### 1. **Conversational Math Tutoring**
- Ask math questions and get detailed explanations
- Step-by-step problem solving guidance
- Concept explanations for algebra topics
- Only responds to math-related queries (filters out non-math content)

### 2. **Image Analysis**
- Upload photos of math worksheets or homework
- Automatic problem recognition and analysis
- Student answer evaluation with feedback
- Error pattern identification

### 3. **Personalized Learning**
- User progress tracking across sessions
- Learning level assessment based on chat history
- Misconception detection and analysis
- Personalized exercise recommendations
- Adaptive difficulty adjustment

### 4. **Session Management**
- Persistent chat sessions with context
- User data storage and retrieval
- Progress summaries and dashboards
- Learning analytics

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements_api.txt
```

### 1. Start the API Server
```bash
python math_chat_api.py
```

The server will start on `http://localhost:8000`

### 2. Open the Web Interface
Open `math_chat_frontend.html` in your browser for a beautiful chat interface.

### 3. Or Use the API Directly
Access the interactive API docs at `http://localhost:8000/docs`

## 📡 API Endpoints

### Chat Endpoint
```http
POST /chat
Content-Type: application/json

{
    "message": "How do I solve 2x + 3 = 7?",
    "user_id": "student_001",
    "session_id": null
}
```

**Response:**
```json
{
    "response": "To solve 2x + 3 = 7, follow these steps...",
    "session_id": "session_abc123",
    "recommendations": [
        {
            "type": "practice",
            "description": "Practice more linear equations",
            "estimated_time": "15 minutes"
        }
    ],
    "user_level_analysis": {
        "level": "beginner",
        "confidence": 0.8,
        "strengths": ["basic arithmetic"],
        "areas_for_improvement": ["equation solving"]
    }
}
```

### Image Upload Endpoint
```http
POST /upload-image
Content-Type: multipart/form-data

file: [image file]
user_id: student_001
session_id: session_abc123
```

**Response:**
```json
{
    "analysis": "I can see 3 algebra problems in your worksheet...",
    "problems_found": [
        {
            "problem": "2x + 3 = 7",
            "student_answer": "x = 5",
            "correct_answer": "x = 2",
            "is_correct": false,
            "feedback": "Check your arithmetic when isolating x"
        }
    ],
    "session_id": "session_abc123",
    "recommendations": [...],
    "user_progress_update": {
        "problems_solved": 3,
        "correct_answers": 2,
        "accuracy": 0.67
    }
}
```

### User Summary Endpoint
```http
GET /user-summary/{user_id}
```

**Response:**
```json
{
    "user_id": "student_001",
    "chat_summary": "Student shows strong understanding of basic algebra...",
    "learning_level": "intermediate",
    "strengths": ["linear equations", "basic factoring"],
    "areas_for_improvement": ["quadratic equations", "word problems"],
    "recommendations": [...],
    "total_interactions": 15,
    "accuracy_trend": [0.6, 0.7, 0.75, 0.8],
    "session_count": 3
}
```

## 🧪 Testing

### Run the Test Suite
```bash
python test_math_chat_api.py
```

This will test:
- ✅ Chat functionality with various math questions
- ✅ Non-math message filtering
- ✅ Image upload and analysis
- ✅ User progress tracking
- ✅ API health checks

### Manual Testing with Frontend
1. Open `math_chat_frontend.html` in your browser
2. Enter a user ID (e.g., "student_001")
3. Try these test cases:

**Chat Tests:**
- "Hi! I need help with algebra"
- "How do I solve 2x + 3 = 7?"
- "Can you explain quadratic equations?"
- "What's the weather like?" (should be filtered out)

**Image Upload Tests:**
- Create a simple text file with math problems
- Upload it using the 📸 Upload button
- See the analysis and feedback

## 🏗️ Architecture

### Core Components

1. **MathChatAgent** - Main conversational agent
   - Extends the AlgebraLearningAgent
   - Handles chat context and math filtering
   - Provides personalized responses

2. **FastAPI Application** - Web server
   - RESTful API endpoints
   - File upload handling
   - CORS support for web frontend

3. **User Data Management** - Progress tracking
   - Session management
   - Learning analytics
   - Recommendation engine

4. **Image Analysis** - Visual problem recognition
   - OCR and LLM-based text extraction
   - Problem identification
   - Answer evaluation

### Data Flow

```
User Input → Math Filter → Agent Processing → Response Generation
     ↓              ↓              ↓              ↓
Chat/Image → Context Analysis → Learning Update → Recommendations
```

## 🎯 Use Cases

### For Students
- **Homework Help**: Upload photos of math problems for instant feedback
- **Concept Learning**: Ask questions about algebra topics
- **Practice Tracking**: Monitor progress and get personalized recommendations
- **Error Analysis**: Understand mistakes and improve problem-solving

### For Educators
- **Student Assessment**: Track individual student progress
- **Misconception Detection**: Identify common error patterns
- **Adaptive Learning**: Provide personalized exercise recommendations
- **Learning Analytics**: Analyze student interaction data

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Configure API settings
export MATH_CHAT_HOST=0.0.0.0
export MATH_CHAT_PORT=8000
export MATH_CHAT_DEBUG=true
```

### Customization
- **Math Topics**: Modify the agent prompts to focus on specific math areas
- **Difficulty Levels**: Adjust the learning level assessment criteria
- **Recommendations**: Customize the exercise recommendation logic
- **UI Themes**: Modify the frontend CSS for different visual styles

## 📊 Example Interactions

### Conversation Flow
```
Student: "Hi! I need help with algebra"
Agent: "Hello! I'm here to help with algebra. What specific topic or problem would you like to work on?"

Student: "How do I solve 2x + 3 = 7?"
Agent: "Great question! To solve 2x + 3 = 7, follow these steps:
1. Subtract 3 from both sides: 2x = 4
2. Divide both sides by 2: x = 2
Would you like to practice with similar problems?"

Student: [uploads image of worksheet]
Agent: "I can see 3 problems in your worksheet:
1. 2x + 5 = 11 → Your answer: x = 4 ❌ (Correct: x = 3)
2. 3y - 4 = 8 → Your answer: y = 4 ✅ Correct!
3. 5z + 2 = 17 → Your answer: z = 3 ✅ Correct!

You're doing well with basic solving! Let's work on double-checking arithmetic."
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

---

**Ready to start learning algebra? 🎓**

1. `python math_chat_api.py` - Start the server
2. Open `math_chat_frontend.html` - Use the web interface
3. Start chatting and uploading math problems!

For questions or support, please check the API documentation at `/docs` when the server is running. 