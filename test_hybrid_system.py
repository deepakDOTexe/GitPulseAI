#!/usr/bin/env python3
"""
Test script for the Hybrid RAG System (Hugging Face + Google Gemini)

This script tests the full pipeline:
1. Hugging Face sentence-transformers for embeddings (free, local)
2. Google Gemini for LLM generation (generous free tier)
3. End-to-end RAG functionality

Usage: python test_hybrid_system.py
"""

import os
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.hybrid_rag_system import create_hybrid_rag_system
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_system_initialization():
    """Test system initialization."""
    print("🔧 Testing Hybrid RAG System Initialization...")
    
    # Check API key
    if not config.GEMINI_API_KEY:
        print("❌ Error: Google Gemini API key not configured!")
        print("Please set GEMINI_API_KEY in your environment or .env file.")
        print("Get your free API key from: https://makersuite.google.com/app/apikey")
        return None
    
    print("✅ Gemini API key configured")
    
    # Initialize system
    try:
        rag_system = create_hybrid_rag_system(
            gemini_api_key=config.GEMINI_API_KEY,
            embedding_model="all-MiniLM-L6-v2",
            gemini_model="gemini-1.5-flash"
        )
        
        if rag_system.is_initialized:
            print("✅ Hybrid RAG system initialized successfully!")
            return rag_system
        else:
            print(f"❌ Failed to initialize system: {rag_system.initialization_error}")
            return None
            
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return None

def test_system_health(rag_system):
    """Test system health."""
    print("\n🏥 Testing System Health...")
    
    try:
        health = rag_system.health_check()
        print(f"Overall Status: {'✅ Healthy' if health['status'] == 'healthy' else '❌ Unhealthy'}")
        
        for check in health['checks']:
            status_icon = "✅" if check['status'] == 'passed' else "❌"
            print(f"{status_icon} {check['name']}: {check['message']}")
        
        return health['status'] == 'healthy'
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_system_stats(rag_system):
    """Test system statistics."""
    print("\n📊 Testing System Statistics...")
    
    try:
        stats = rag_system.get_system_stats()
        
        print(f"System Type: {stats.get('system_type', 'Unknown')}")
        print(f"Initialized: {'✅' if stats.get('initialized') else '❌'}")
        
        # Configuration info
        config_info = stats.get('config', {})
        print(f"Embedding Model: {config_info.get('embedding_model', 'Unknown')}")
        print(f"LLM Model: {config_info.get('llm_model', 'Unknown')}")
        print(f"Free Tier: {'✅' if config_info.get('free_tier') else '❌'}")
        
        # Data processor stats
        if 'data_processor' in stats:
            data_stats = stats['data_processor']
            print(f"Documents: {data_stats.get('total_documents', 0)}")
            print(f"Chunks: {data_stats.get('total_chunks', 0)}")
        
        # Vector store stats
        if 'vector_store' in stats:
            vector_stats = stats['vector_store']
            print(f"Embeddings: {vector_stats.get('total_documents', 0)}")
            print(f"Embedding Dimension: {vector_stats.get('embedding_dimension', 0)}")
        
        # LLM stats
        if 'llm' in stats:
            llm_stats = stats['llm']
            print(f"LLM Available: {'✅' if llm_stats.get('available') else '❌'}")
            print(f"Rate Limit: {llm_stats.get('rate_limit', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        return False

def test_queries(rag_system):
    """Test query processing."""
    print("\n💬 Testing Query Processing...")
    
    test_queries = [
        "What are GitLab's core values?",
        "How does GitLab handle remote work?",
        "What is GitLab's engineering process?",
        "How does GitLab approach diversity and inclusion?"
    ]
    
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        
        try:
            response = rag_system.query(query)
            
            if response.get('answer'):
                print(f"✅ Response generated ({len(response['answer'])} chars)")
                print(f"Sources found: {len(response.get('sources', []))}")
                
                # Show first 200 chars of response
                answer_preview = response['answer'][:200] + "..." if len(response['answer']) > 200 else response['answer']
                print(f"Answer preview: {answer_preview}")
                
                success_count += 1
            else:
                print("❌ No answer generated")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    print(f"\n📈 Query Success Rate: {success_count}/{len(test_queries)} ({success_count/len(test_queries)*100:.1f}%)")
    return success_count == len(test_queries)

def main():
    """Main test function."""
    print("🚀 GitLab Handbook Assistant - Hybrid RAG System Test")
    print("=" * 60)
    print("Stack: Hugging Face Embeddings + Google Gemini LLM")
    print("Cost: 100% Free (Gemini has generous free tier)")
    print("=" * 60)
    
    # Test initialization
    rag_system = test_system_initialization()
    if not rag_system:
        print("\n❌ Initialization failed. Cannot continue with tests.")
        return False
    
    # Test system health
    if not test_system_health(rag_system):
        print("\n❌ System health check failed.")
        return False
    
    # Test system stats
    if not test_system_stats(rag_system):
        print("\n❌ System stats test failed.")
        return False
    
    # Test queries
    if not test_queries(rag_system):
        print("\n❌ Query tests failed.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Hybrid RAG system is working correctly.")
    print("✅ Ready to run: streamlit run app.py")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 