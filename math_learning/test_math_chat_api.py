#!/usr/bin/env python3
"""
Test client for Math Chat API

Demonstrates the chat and image upload functionality.
"""

import asyncio
import json
import requests
import base64
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_chat_functionality():
    """Test the chat endpoint with various math questions."""
    
    print("🧪 Testing Chat Functionality")
    print("=" * 50)
    
    # Test user
    user_id = "test_student_123"
    
    # Test cases
    test_messages = [
        "Hi! I need help with algebra",
        "How do I solve 2x + 3 = 7?",
        "Can you explain quadratic equations?",
        "What's the difference between linear and quadratic functions?",
        "I'm confused about factoring polynomials"
    ]
    
    session_id = None
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n📝 Test {i}: {message}")
        
        # Send chat message
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "message": message,
                "user_id": user_id,
                "session_id": session_id
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            session_id = data["session_id"]  # Keep session for continuity
            
            print(f"✅ Response: {data['response'][:100]}...")
            
            if data.get("recommendations"):
                print(f"📚 Recommendations: {len(data['recommendations'])} items")
            
            if data.get("user_level_analysis"):
                level_info = data["user_level_analysis"]
                print(f"📊 Level: {level_info.get('level', 'unknown')} (confidence: {level_info.get('confidence', 0):.1f})")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    
    return user_id, session_id

def test_non_math_message():
    """Test that non-math messages are rejected."""
    
    print("\n🚫 Testing Non-Math Message Filtering")
    print("=" * 50)
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "What's the weather like today?",
            "user_id": "test_user",
            "session_id": None
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Correctly filtered: {data['response']}")
    else:
        print(f"❌ Error: {response.status_code}")

def test_image_upload():
    """Test image upload functionality."""
    
    print("\n📸 Testing Image Upload")
    print("=" * 50)
    
    # Create a simple test image (base64 encoded text)
    test_image_content = "Math worksheet: 2x+5=11, 3y-4=8, 5z+2=17"
    test_image_bytes = test_image_content.encode()
    
    # Create a temporary image file
    temp_image_path = Path("temp_math_worksheet.txt")
    temp_image_path.write_bytes(test_image_bytes)
    
    try:
        # Upload the image
        with open(temp_image_path, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/upload-image",
                files={"file": ("worksheet.txt", f, "image/png")},
                data={
                    "user_id": "test_student_123",
                    "session_id": None
                }
            )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis: {data['analysis'][:150]}...")
            print(f"📊 Problems found: {len(data['problems_found'])}")
            print(f"📈 Progress update: {data['user_progress_update']}")
            
            for i, problem in enumerate(data['problems_found'], 1):
                status = "✅" if problem['is_correct'] else "❌"
                print(f"   {i}. {problem['problem']} → {problem['student_answer']} {status}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    
    finally:
        # Clean up
        if temp_image_path.exists():
            temp_image_path.unlink()

def test_user_summary():
    """Test user summary endpoint."""
    
    print("\n📊 Testing User Summary")
    print("=" * 50)
    
    user_id = "test_student_123"
    
    response = requests.get(f"{BASE_URL}/user-summary/{user_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ User: {data['user_id']}")
        print(f"📝 Chat Summary: {data['chat_summary']}")
        print(f"📊 Learning Level: {data['learning_level']}")
        print(f"💪 Strengths: {data['strengths']}")
        print(f"🎯 Recommendations: {len(data['recommendations'])} items")
        print(f"📈 Total Interactions: {data['total_interactions']}")
    else:
        print(f"❌ Error: {response.status_code}")

def test_health_check():
    """Test health check endpoint."""
    
    print("\n🏥 Testing Health Check")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"🤖 Algebra Agent Available: {data['algebra_agent_available']}")
        print(f"⏰ Timestamp: {data['timestamp']}")
    else:
        print(f"❌ Error: {response.status_code}")

def main():
    """Run all tests."""
    
    print("🚀 Math Chat API Test Suite")
    print("=" * 60)
    
    try:
        # Test health check first
        test_health_check()
        
        # Test chat functionality
        user_id, session_id = test_chat_functionality()
        
        # Test non-math filtering
        test_non_math_message()
        
        # Test image upload
        test_image_upload()
        
        # Test user summary
        test_user_summary()
        
        print("\n🎉 All tests completed!")
        print("\n💡 To start the API server, run:")
        print("   python math_chat_api.py")
        print("\n🌐 Then access the interactive docs at:")
        print("   http://localhost:8000/docs")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running!")
        print("   Start it with: python math_chat_api.py")
    except Exception as e:
        print(f"❌ Test Error: {e}")

if __name__ == "__main__":
    main() 