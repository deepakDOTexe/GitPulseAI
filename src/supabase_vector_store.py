"""
Supabase Vector Store for GitPulseAI

This module provides a vector store using Supabase (PostgreSQL + pgvector)
for cloud deployment with excellent performance and cost efficiency.
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from supabase import create_client, Client
import streamlit as st
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

class SupabaseVectorStore:
    """Vector store using Supabase PostgreSQL with pgvector."""
    
    def __init__(
        self, 
        supabase_url: Optional[str] = None, 
        supabase_key: Optional[str] = None,
        table_name: str = "gitlab_documents",
        similarity_threshold: float = 0.3,
        max_results: int = 5
    ):
        """
        Initialize Supabase vector store.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon key
            table_name: Name of the documents table
            similarity_threshold: Minimum similarity score for results
            max_results: Maximum number of results to return
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.table_name = table_name
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and Key are required")
        
        # Initialize Supabase client
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Ensure table exists
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the documents table exists with proper schema."""
        try:
            # Test table access
            result = self.client.table(self.table_name).select("count").limit(1).execute()
            return True
        except Exception as e:
            print(f"Warning: Could not verify table exists: {e}")
            return False
    
    @st.cache_data(ttl=3600)
    def _get_embedding(_self, text: str, api_key: str) -> Optional[List[float]]:
        """
        Generate embeddings using Google Gemini API.
        
        Args:
            text: Text to generate embeddings for
            api_key: Google Gemini API key
            
        Returns:
            List of embedding values, or None if failed
        """
        try:
            # Use Google Gemini REST API for embeddings (official format)
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            
            data = {
                "model": "models/gemini-embedding-001",
                "content": {
                    "parts": [{"text": text}]
                },
                "taskType": "RETRIEVAL_QUERY",
                "outputDimensionality": 1536  # Match existing Supabase schema
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            if 'embedding' in result and 'values' in result['embedding']:
                return result['embedding']['values']
            else:
                print("❌ Unexpected response format from Google Gemini API")
                return None
        
        except Exception as e:
            print(f"❌ Failed to generate embedding: {str(e)}")
            return None
    
    def add_document(self, document: Dict[str, Any], gemini_api_key: Optional[str] = None) -> bool:
        """
        Add a single document to the vector store.
        
        Args:
            document: Document to add
            gemini_api_key: Google Gemini API key for generating embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare document data
            doc_data = {
                "document_id": document.get("id", f"doc_{hash(str(document))}"),
                "title": document.get("title", ""),
                "content": document.get("content", ""),
                "url": document.get("url", ""),
                "section": document.get("section", ""),
                "metadata": document.get("metadata", {}),
                "created_at": "now()",
                "updated_at": "now()"
            }
            
            # Generate embedding if API key provided
            if gemini_api_key and doc_data["content"]:
                embedding = self._get_embedding(doc_data["content"], gemini_api_key)
                if embedding:
                    doc_data["embedding"] = embedding
            
            # Insert into Supabase
            result = self.client.table(self.table_name).upsert(doc_data).execute()
            
            return bool(result.data)
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]], gemini_api_key: Optional[str] = None) -> int:
        """
        Add multiple documents to the vector store.
        
        Args:
            documents: List of documents to add
            gemini_api_key: Google Gemini API key for generating embeddings
            
        Returns:
            Number of documents successfully added
        """
        successful_adds = 0
        
        for doc in documents:
            if self.add_document(doc, gemini_api_key):
                successful_adds += 1
                # Rate limiting
                time.sleep(0.1)
        
        return successful_adds
    
    def search(self, query: str, gemini_api_key: Optional[str] = None, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            gemini_api_key: Google Gemini API key for generating query embedding (optional)
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant documents
        """
        max_results = max_results or self.max_results
        
        # Try vector search if API key is provided
        if gemini_api_key:
            try:
                return self._vector_search(query, gemini_api_key, max_results)
            except Exception as e:
                print(f"Vector search failed, falling back to keyword search: {e}")
        
        # Fallback to keyword search
        return self._keyword_search(query, max_results)
    
    def _vector_search(self, query: str, gemini_api_key: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            gemini_api_key: Google Gemini API key
            max_results: Maximum number of results
            
        Returns:
            List of relevant documents
        """
        # Generate query embedding
        query_embedding = self._get_embedding(query, gemini_api_key)
        if not query_embedding:
            raise Exception("Failed to generate query embedding")
        
        # Perform vector search using Supabase RPC function
        try:
            result = self.client.rpc(
                'match_gitlab_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': self.similarity_threshold,
                    'match_count': max_results
                }
            ).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"RPC vector search failed: {e}")
            # If RPC fails, try direct similarity search
            return self._direct_vector_search(query_embedding, max_results)
    
    def _direct_vector_search(self, query_embedding: List[float], max_results: int) -> List[Dict[str, Any]]:
        """Direct vector search as fallback."""
        # This is a simplified approach - in practice, you'd want to use pgvector operations
        # For now, return empty list as we'll rely on keyword search
        return []
    
    def _keyword_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search using PostgreSQL full-text search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of relevant documents
        """
        try:
            # Use PostgreSQL full-text search
            result = self.client.table(self.table_name).select(
                "document_id, title, content, url, section, metadata"
            ).text_search('content', query).execute()
            
            # Manually limit results if needed
            data = result.data if result.data else []
            return data[:max_results] if len(data) > max_results else data
            
        except Exception as e:
            print(f"Keyword search failed: {e}")
            
            # Fallback to simple ILIKE search
            try:
                result = self.client.table(self.table_name).select(
                    "document_id, title, content, url, section, metadata"
                ).ilike('content', f'%{query}%').limit(max_results).execute()
                
                return result.data if result.data else []
                
            except Exception as e2:
                print(f"Fallback search also failed: {e2}")
                return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Get total document count
            all_docs = self.client.table(self.table_name).select("*").execute()
            total_count = len(all_docs.data) if all_docs.data else 0
            
            # Count documents with embeddings
            docs_with_embeddings = [doc for doc in all_docs.data if doc.get('embedding')] if all_docs.data else []
            embedding_count = len(docs_with_embeddings)
            
            return {
                "total_documents": total_count,
                "documents_with_embeddings": embedding_count,
                "embedding_coverage": (embedding_count / total_count * 100) if total_count > 0 else 0
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"error": str(e)}

@st.cache_resource
def create_supabase_vector_store(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    **kwargs
) -> SupabaseVectorStore:
    """Create and cache a Supabase vector store instance."""
    return SupabaseVectorStore(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        **kwargs
    )

def create_supabase_rag_system(
    supabase_url: str,
    supabase_key: str,
    gemini_api_key: Optional[str] = None,
    **kwargs
):
    """Create a Supabase-based RAG system."""
    from src.supabase_rag_system import SupabaseRAGSystem
    
    return SupabaseRAGSystem(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        gemini_api_key=gemini_api_key,
        **kwargs
    ) 