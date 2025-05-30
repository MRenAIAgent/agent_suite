#!/usr/bin/env python3
"""
Simple Math Chat API - Standalone Version

A basic HTTP server for math tutoring that works without external dependencies.
"""

import json
import http.server
import socketserver
import urllib.parse
from datetime import datetime
import uuid
import base64

# Simple HTTP server for math chat
class MathChatHandler(http.server.BaseHTTPRequestHandler):
    
    # Store user sessions and data
    user_sessions = {}
    user_data = {}
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self.send_json_response({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "message": "Simple Math Chat API is running"
            })
        elif self.path == '/':
            self.send_json_response({
                "message": "Simple Math Chat API",
                "version": "1.0.0",
                "endpoints": {
                    "chat": "POST /chat - Chat with math tutor",
                    "health": "GET /health - Health check"
                }
            })
        elif self.path.startswith('/user-summary/'):
            user_id = self.path.split('/')[-1]
            summary = self.get_user_summary(user_id)
            self.send_json_response(summary)
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/chat':
            self.handle_chat()
        else:
            self.send_error(404, "Not Found")
    
    def handle_chat(self):
        """Handle chat messages."""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            message = data.get('message', '')
            user_id = data.get('user_id', '')
            session_id = data.get('session_id')
            
            if not message or not user_id:
                self.send_error(400, "Missing message or user_id")
                return
            
            # Check if message is math-related
            if not self.is_math_related(message):
                response = {
                    "response": "I'm a math tutor, so I can only help with math-related questions! Please ask me about algebra, equations, or math problems. 📚",
                    "user_id": user_id,
                    "session_id": session_id or str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat()
                }
                self.send_json_response(response)
                return
            
            # Generate math tutoring response
            session_id = self.get_or_create_session(user_id, session_id)
            math_response = self.generate_math_response(message)
            recommendations = self.get_recommendations(user_id)
            
            # Update session
            self.update_session(session_id, message, math_response)
            
            response = {
                "response": math_response,
                "user_id": user_id,
                "session_id": session_id,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
    
    def is_math_related(self, message):
        """Check if message is math-related."""
        math_keywords = [
            "equation", "solve", "algebra", "math", "calculate", "factor", 
            "simplify", "graph", "function", "variable", "coefficient",
            "polynomial", "quadratic", "linear", "expression", "formula",
            "number", "fraction", "decimal", "percent", "homework", 
            "exercise", "problem", "solution", "answer", "help", 
            "explain", "understand", "x", "y", "z", "=", "+", "-", "*", "/"
        ]
        
        # Exclude non-math topics
        non_math_keywords = [
            "weather", "temperature", "rain", "sunny", "cloudy", "forecast",
            "news", "politics", "sports", "music", "movie", "food", "cooking",
            "travel", "vacation", "job", "work", "business", "health", "doctor"
        ]
        
        message_lower = message.lower()
        
        # If it contains non-math keywords, reject it
        if any(keyword in message_lower for keyword in non_math_keywords):
            return False
            
        # Check for math keywords
        return any(keyword in message_lower for keyword in math_keywords)
    
    def generate_math_response(self, message):
        """Generate a math tutoring response."""
        message_lower = message.lower()
        
        # Specific equation solving
        if "2x + 3 = 7" in message_lower:
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
        
        elif "quadratic" in message_lower:
            return """Quadratic equations have the form ax² + bx + c = 0. There are several methods to solve them:

**1. Factoring** (when possible)
**2. Quadratic Formula:** x = (-b ± √(b² - 4ac)) / 2a
**3. Completing the Square**

Which method would you like me to explain, or do you have a specific quadratic equation to solve?"""
        
        elif any(word in message_lower for word in ["solve", "equation"]):
            return """I'd be happy to help you solve that equation! 

To give you the best help, please:
1. **Share the specific equation** you're working on
2. **Tell me what step you're stuck on**
3. **Show me what you've tried so far**

For example: "How do I solve 3x - 5 = 10?"

This way I can give you step-by-step guidance! 📝"""
        
        elif any(word in message_lower for word in ["explain", "understand", "help"]):
            return """Great question! I love helping students understand math concepts.

Could you tell me:
- **Which specific topic** you'd like me to explain?
- **What part is confusing** you the most?
- **Have you seen this before** or is it completely new?

Some topics I can help with:
• Linear equations (2x + 3 = 7)
• Quadratic equations (x² - 5x + 6 = 0)
• Factoring and simplifying
• Graphing functions
• Word problems

What would you like to explore? 🎯"""
        
        else:
            return """Hi! I'm your algebra tutor! 🎓 

I can help you with:
📝 **Solving equations** - Linear, quadratic, and more
📊 **Graphing functions** - Understanding slopes and intercepts  
🔢 **Simplifying expressions** - Factoring and combining like terms
📖 **Explaining concepts** - Breaking down complex topics
🎯 **Practice problems** - Step-by-step guidance

What math topic would you like to work on today?"""
    
    def get_or_create_session(self, user_id, session_id=None):
        """Get existing session or create new one."""
        if session_id and session_id in self.user_sessions:
            return session_id
        
        new_session_id = str(uuid.uuid4())
        self.user_sessions[new_session_id] = {
            "user_id": user_id,
            "created": datetime.now().isoformat(),
            "messages": []
        }
        
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                "total_sessions": 0,
                "total_messages": 0,
                "learning_level": "beginner",
                "last_activity": datetime.now().isoformat()
            }
        
        return new_session_id
    
    def update_session(self, session_id, user_message, agent_response):
        """Update session with new message."""
        if session_id in self.user_sessions:
            session = self.user_sessions[session_id]
            session["messages"].append({
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "agent_response": agent_response
            })
            
            user_id = session["user_id"]
            if user_id in self.user_data:
                self.user_data[user_id]["total_messages"] += 1
                self.user_data[user_id]["last_activity"] = datetime.now().isoformat()
    
    def get_recommendations(self, user_id):
        """Get personalized recommendations."""
        user_data = self.user_data.get(user_id, {})
        total_messages = user_data.get("total_messages", 0)
        
        if total_messages < 5:
            return [
                {
                    "type": "practice",
                    "topic": "linear_equations",
                    "description": "Practice solving two-step linear equations",
                    "estimated_time": "15 minutes"
                }
            ]
        else:
            return [
                {
                    "type": "challenge",
                    "topic": "quadratic_equations", 
                    "description": "Ready for quadratic equations using factoring",
                    "estimated_time": "20 minutes"
                }
            ]
    
    def get_user_summary(self, user_id):
        """Get user learning summary."""
        user_data = self.user_data.get(user_id, {})
        user_sessions = [s for s in self.user_sessions.values() if s["user_id"] == user_id]
        
        total_messages = user_data.get("total_messages", 0)
        
        return {
            "user_id": user_id,
            "chat_summary": f"User has sent {total_messages} messages across {len(user_sessions)} sessions",
            "learning_level": user_data.get("learning_level", "beginner"),
            "total_interactions": total_messages,
            "last_activity": user_data.get("last_activity", ""),
            "recommendations": self.get_recommendations(user_id)
        }
    
    def send_json_response(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def main():
    """Start the simple math chat server."""
    PORT = 8000
    
    print("🚀 Starting Simple Math Chat API...")
    print(f"🌐 Server running at: http://localhost:{PORT}")
    print("📚 Features: Math tutoring chat with basic recommendations")
    print("🔗 Test with: curl http://localhost:8000/health")
    print("\n💡 Try these endpoints:")
    print("  GET  /health - Health check")
    print("  POST /chat - Send math questions")
    print("  GET  /user-summary/{user_id} - Get user progress")
    print("\nPress Ctrl+C to stop the server")
    
    with socketserver.TCPServer(("", PORT), MathChatHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Shutting down server...")

if __name__ == "__main__":
    main() 