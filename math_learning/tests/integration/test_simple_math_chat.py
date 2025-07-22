#!/usr/bin/env python3
"""
Test script for Simple Math Chat API

Tests the basic functionality of the standalone math chat server.
"""

import json
import urllib.request
import urllib.parse
import time

def test_api():
    """Test the Simple Math Chat API."""
    base_url = "http://localhost:8000"
    
    print("🧪 Simple Math Chat API Test Suite")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n🏥 Testing Health Check")
    print("-" * 30)
    try:
        response = urllib.request.urlopen(f"{base_url}/health")
        data = json.loads(response.read().decode())
        print("✅ Health check passed!")
        print(f"   Status: {data['status']}")
        print(f"   Message: {data['message']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Math Question
    print("\n📝 Testing Math Question")
    print("-" * 30)
    try:
        chat_data = {
            "message": "How do I solve 2x + 3 = 7?",
            "user_id": "test_student"
        }
        
        req = urllib.request.Request(
            f"{base_url}/chat",
            data=json.dumps(chat_data).encode(),
            headers={'Content-Type': 'application/json'}
        )
        
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode())
        
        print("✅ Math question answered!")
        print(f"   Question: {chat_data['message']}")
        print(f"   Response: {data['response'][:100]}...")
        print(f"   Session ID: {data['session_id']}")
        print(f"   Recommendations: {len(data['recommendations'])} items")
        
        session_id = data['session_id']
        
    except Exception as e:
        print(f"❌ Math question test failed: {e}")
        return
    
    # Test 3: Non-Math Question (should be rejected)
    print("\n🚫 Testing Non-Math Question")
    print("-" * 30)
    try:
        chat_data = {
            "message": "What's the weather like today?",
            "user_id": "test_student"
        }
        
        req = urllib.request.Request(
            f"{base_url}/chat",
            data=json.dumps(chat_data).encode(),
            headers={'Content-Type': 'application/json'}
        )
        
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode())
        
        if "math tutor" in data['response'].lower():
            print("✅ Non-math question properly rejected!")
            print(f"   Response: {data['response']}")
        else:
            print("❌ Non-math question was not rejected")
            
    except Exception as e:
        print(f"❌ Non-math question test failed: {e}")
        return
    
    # Test 4: Follow-up Math Question
    print("\n🔄 Testing Follow-up Question")
    print("-" * 30)
    try:
        chat_data = {
            "message": "Can you explain quadratic equations?",
            "user_id": "test_student",
            "session_id": session_id
        }
        
        req = urllib.request.Request(
            f"{base_url}/chat",
            data=json.dumps(chat_data).encode(),
            headers={'Content-Type': 'application/json'}
        )
        
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode())
        
        print("✅ Follow-up question answered!")
        print(f"   Question: {chat_data['message']}")
        print(f"   Response: {data['response'][:100]}...")
        print(f"   Same session: {data['session_id'] == session_id}")
        
    except Exception as e:
        print(f"❌ Follow-up question test failed: {e}")
        return
    
    # Test 5: User Summary
    print("\n📊 Testing User Summary")
    print("-" * 30)
    try:
        response = urllib.request.urlopen(f"{base_url}/user-summary/test_student")
        data = json.loads(response.read().decode())
        
        print("✅ User summary retrieved!")
        print(f"   User ID: {data['user_id']}")
        print(f"   Summary: {data['chat_summary']}")
        print(f"   Level: {data['learning_level']}")
        print(f"   Interactions: {data['total_interactions']}")
        print(f"   Recommendations: {len(data['recommendations'])} items")
        
    except Exception as e:
        print(f"❌ User summary test failed: {e}")
        return
    
    print("\n🎉 All tests passed! The Simple Math Chat API is working correctly.")
    print("\n💡 Try these examples:")
    print("   • 'How do I solve x + 5 = 12?'")
    print("   • 'Explain factoring'")
    print("   • 'What is a quadratic equation?'")
    print("   • 'Help me with algebra homework'")

def main():
    """Run the test suite."""
    print("Starting Simple Math Chat API tests...")
    print("Make sure the server is running: python simple_math_chat.py")
    print()
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    test_api()

if __name__ == "__main__":
    main() 