#!/usr/bin/env python3
"""
Quick test to verify the TF-IDF fallback works without API calls
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.simple_vector_store import SimpleVectorStore
from src.data_processor import GitLabDataProcessor

def test_tfidf_only():
    """Test only the TF-IDF vector store functionality."""
    print("🔧 Testing TF-IDF Vector Store (No API calls)...")
    
    # Load and process data
    processor = GitLabDataProcessor()
    if not processor.load_data():
        print("❌ Failed to load data")
        return False
    
    chunks = processor.process_documents()
    print(f"✅ Processed {len(chunks)} chunks")
    
    # Test simple vector store
    store = SimpleVectorStore()
    successful_adds = store.add_documents(chunks)
    print(f"✅ Added {successful_adds} documents to TF-IDF store")
    
    # Test searches
    test_queries = [
        "values",
        "remote work",
        "engineering",
        "diversity"
    ]
    
    for query in test_queries:
        results = store.search(query, 3)
        print(f"Query '{query}': {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.get('title', 'No title')} (score: {result.get('similarity_score', 0):.3f})")
    
    return True

if __name__ == "__main__":
    success = test_tfidf_only()
    print(f"\n{'✅ TF-IDF test passed' if success else '❌ TF-IDF test failed'}") 