#!/usr/bin/env python3
"""
Simple test script to debug Google Gemini Embedding API
"""

import os
import requests
from dotenv import load_dotenv

def test_gemini_embedding():
    """Test Google Gemini embedding API with detailed debugging."""
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return
    
    print(f"🔍 Testing Google Gemini API...")
    print(f"Using API key ending in: ...{api_key[-10:]}")
    
    # Test the exact format from user's curl example
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    
    data = {
        "model": "models/gemini-embedding-001",
        "content": {
            "parts": [{"text": "What is the meaning of life?"}]
        }
    }
    
    try:
        print(f"📡 Making request to: {url}")
        print(f"📋 Headers: {headers}")
        print(f"📦 Data: {data}")
        
        response = requests.post(url, headers=headers, json=data)
        
        print(f"🔄 Response status: {response.status_code}")
        print(f"📄 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response: {result}")
            
            if 'embedding' in result and 'values' in result['embedding']:
                embedding_length = len(result['embedding']['values'])
                print(f"🎯 Embedding dimension: {embedding_length}")
            else:
                print(f"⚠️ Unexpected response format")
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"💥 Exception: {e}")

if __name__ == "__main__":
    test_gemini_embedding() 