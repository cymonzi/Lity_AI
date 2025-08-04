#!/usr/bin/env python3
"""
Test script for the local model server
"""

import requests
import json

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:5000")
        print("✅ Health Check:", response.json())
        return True
    except Exception as e:
        print("❌ Health Check Failed:", e)
        return False

def test_chat(message):
    """Test the chat endpoint"""
    try:
        response = requests.post(
            "http://localhost:5000/chat/",
            headers={"Content-Type": "application/json"},
            json={"text": message}
        )
        result = response.json()
        print(f"💬 User: {message}")
        print(f"🤖 Bot: {result.get('reply', 'No response')}")
        return True
    except Exception as e:
        print(f"❌ Chat Test Failed: {e}")
        return False

def test_status():
    """Test the status endpoint"""
    try:
        response = requests.get("http://localhost:5000/status")
        print("📊 Status:", response.json())
        return True
    except Exception as e:
        print("❌ Status Check Failed:", e)
        return False

if __name__ == "__main__":
    print("🧪 Testing Lity AI Local Model Server")
    print("=" * 50)
    
    # Test endpoints
    health_ok = test_health()
    status_ok = test_status()
    
    if health_ok and status_ok:
        print("\n💬 Testing Chat Functionality:")
        print("-" * 30)
        
        # Test various questions
        test_messages = [
            "How do I start budgeting?",
            "What is saving?",
            "Tell me about investing",
            "What is Litywise?",
            "How does Nfunayo work?"
        ]
        
        for msg in test_messages:
            test_chat(msg)
            print()
    
    print("🎉 Testing complete!")
