#!/usr/bin/env python3
"""
Demo script for Simple Math Chat API

Shows the capabilities of the math tutoring system.
"""

import json
import urllib.request
import urllib.parse
import time

def demo_chat():
    """Demonstrate the math chat capabilities."""
    base_url = "http://localhost:8000"
    user_id = "demo_student"
    
    print("🎓 Math Chat API Demo")
    print("=" * 50)
    print("🌐 Server: http://localhost:8000")
    print("👤 User: demo_student")
    print()
    
    # Test questions to demonstrate capabilities
    test_questions = [
        "How do I solve 2x + 3 = 7?",
        "Can you explain quadratic equations?", 
        "What's the weather like?",  # This should be rejected
        "Help me factor x² - 5x + 6",
        "I need help with my algebra homework"
    ]
    
    session_id = None
    
    for i, question in enumerate(test_questions, 1):
        print(f"📝 Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Prepare request data
            chat_data = {
                "message": question,
                "user_id": user_id
            }
            
            if session_id:
                chat_data["session_id"] = session_id
            
            # Send request
            req = urllib.request.Request(
                f"{base_url}/chat",
                data=json.dumps(chat_data).encode(),
                headers={'Content-Type': 'application/json'}
            )
            
            response = urllib.request.urlopen(req)
            data = json.loads(response.read().decode())
            
            # Update session ID for continuity
            session_id = data.get('session_id')
            
            # Display response
            print(f"🤖 Response:")
            print(data['response'])
            
            # Show recommendations if any
            if data.get('recommendations'):
                print(f"\n💡 Recommendations:")
                for rec in data['recommendations']:
                    print(f"   📚 {rec['description']} ({rec['estimated_time']})")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
        
        # Small delay between questions
        time.sleep(1)
    
    # Show user summary
    print("📊 User Summary")
    print("-" * 40)
    try:
        response = urllib.request.urlopen(f"{base_url}/user-summary/{user_id}")
        data = json.loads(response.read().decode())
        
        print(f"👤 User ID: {data['user_id']}")
        print(f"📈 Summary: {data['chat_summary']}")
        print(f"🎯 Level: {data['learning_level']}")
        print(f"💬 Total Interactions: {data['total_interactions']}")
        print(f"⏰ Last Activity: {data['last_activity']}")
        
        if data.get('recommendations'):
            print(f"\n🎯 Current Recommendations:")
            for rec in data['recommendations']:
                print(f"   📚 {rec['description']} ({rec['estimated_time']})")
        
    except Exception as e:
        print(f"❌ Error getting user summary: {e}")
    
    print("\n🎉 Demo completed!")
    print("\n💡 Try the web interface:")
    print("   Open simple_math_chat_frontend.html in your browser")
    print("   Or visit: http://localhost:8000 for API info")

def main():
    """Run the demo."""
    print("🚀 Starting Math Chat API Demo...")
    print("📋 Make sure the server is running: python simple_math_chat.py")
    print()
    
    # Check if server is running
    try:
        response = urllib.request.urlopen("http://localhost:8000/health")
        health_data = json.loads(response.read().decode())
        print(f"✅ Server is healthy: {health_data['message']}")
        print()
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        print("Please start the server first: python simple_math_chat.py")
        return
    
    demo_chat()

if __name__ == "__main__":
    main() 