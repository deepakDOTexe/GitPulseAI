"""
Supabase Vector Store for GitPulseAI

This module provides a vector store using Supabase (PostgreSQL + pgvector)
for cloud deployment with excellent performance and cost efficiency.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from supabase import create_client, Client
import streamlit as st
from datetime import datetime

logger = logging.getLogger(__name__)

class SupabaseVectorStore:
    """
    Vector store using Supabase (PostgreSQL + pgvector).
    
    Features:
    - PostgreSQL with pgvector extension
    - Integrated document and vector storage
    - Excellent performance and cost efficiency
    - Streamlit Cloud compatible
    """
    
    def __init__(self,
                 supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None,
                 table_name: str = "gitlab_documents",
                 similarity_threshold: float = 0.3,
                 max_results: int = 5):
        """
        Initialize Supabase vector store.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/service role key
            table_name: Name of the table to store documents
            similarity_threshold: Minimum similarity for search results
            max_results: Maximum results to return
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.table_name = table_name
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key are required")
        
        # Initialize Supabase client
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Check connection and table setup
        self._ensure_table_exists()
        
        logger.info(f"Initialized SupabaseVectorStore with table: {self.table_name}")
    
    def _ensure_table_exists(self):
        """Ensure the documents table exists with proper schema."""
        try:
            # Test connection by running a simple query
            result = self.client.table(self.table_name).select("count", count="exact").limit(1).execute()
            logger.info(f"Connected to Supabase table: {self.table_name}")
            
        except Exception as e:
            logger.warning(f"Table may not exist or needs setup: {e}")
            logger.info("Please run the setup SQL commands in your Supabase database")
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def _get_embedding(_self, text: str, api_key: str) -> Optional[List[float]]:
        """
        Generate embedding using OpenAI API with caching.
        
        This is cached by Streamlit for efficiency.
        """
        try:
            import openai
            
            openai.api_key = api_key
            
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text.replace("\n", " ")
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def add_document(self, document: Dict[str, Any], openai_api_key: Optional[str] = None) -> bool:
        """
        Add a document with its embedding to Supabase.
        
        Args:
            document: Document dict with id, title, content, etc.
            openai_api_key: OpenAI API key for generating embeddings
            
        Returns:
            bool: True if added successfully
        """
        try:
            doc_id = document.get('id')
            content = document.get('content', '')
            
            if not doc_id or not content.strip():
                logger.warning(f"Skipping document with missing id or content: {doc_id}")
                return False
            
            # Generate embedding if OpenAI key is provided
            embedding = None
            if openai_api_key:
                embedding = self._get_embedding(content, openai_api_key)
                if not embedding:
                    logger.warning(f"Failed to generate embedding for document: {doc_id}")
            
            # Prepare document for insertion
            doc_data = {
                'document_id': doc_id,
                'title': document.get('title', ''),
                'content': content,
                'url': document.get('url', ''),
                'section': document.get('section', ''),
                'keywords': json.dumps(document.get('keywords', [])),
                'metadata': json.dumps({
                    'source': document.get('source', ''),
                    'timestamp': datetime.now().isoformat(),
                    'chunk_index': document.get('chunk_index', 0)
                }),
                'embedding': embedding
            }
            
            # Insert into Supabase
            result = self.client.table(self.table_name).insert(doc_data).execute()
            
            if result.data:
                logger.debug(f"Successfully added document: {doc_id}")
                return True
            else:
                logger.error(f"Failed to insert document: {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding document {document.get('id', 'unknown')}: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]], openai_api_key: Optional[str] = None) -> int:
        """
        Add multiple documents to Supabase.
        
        Args:
            documents: List of document dicts
            openai_api_key: OpenAI API key for generating embeddings
            
        Returns:
            int: Number of documents successfully added
        """
        successful_adds = 0
        batch_size = 100  # Process in batches for better performance
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_data = []
            
            for doc in batch:
                doc_id = doc.get('id')
                content = doc.get('content', '')
                
                if not doc_id or not content.strip():
                    continue
                
                # Generate embedding if API key provided
                embedding = None
                if openai_api_key:
                    embedding = self._get_embedding(content, openai_api_key)
                
                doc_data = {
                    'document_id': doc_id,
                    'title': doc.get('title', ''),
                    'content': content,
                    'url': doc.get('url', ''),
                    'section': doc.get('section', ''),
                    'keywords': json.dumps(doc.get('keywords', [])),
                    'metadata': json.dumps({
                        'source': doc.get('source', ''),
                        'timestamp': datetime.now().isoformat(),
                        'chunk_index': doc.get('chunk_index', 0)
                    }),
                    'embedding': embedding
                }
                batch_data.append(doc_data)
            
            try:
                result = self.client.table(self.table_name).insert(batch_data).execute()
                successful_adds += len(result.data) if result.data else 0
                
                logger.info(f"Batch {i//batch_size + 1}: Added {len(batch_data)} documents")
                
            except Exception as e:
                logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
        
        logger.info(f"Successfully added {successful_adds}/{len(documents)} documents to Supabase")
        return successful_adds
    
    def search(self, query: str, openai_api_key: Optional[str] = None, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity or keyword search.
        
        Args:
            query: Search query
            openai_api_key: OpenAI API key for generating query embedding
            max_results: Maximum results to return
            
        Returns:
            List[Dict]: Similar documents with similarity scores
        """
        max_results = max_results or self.max_results
        
        # Try vector search first if OpenAI key is available
        if openai_api_key:
            vector_results = self._vector_search(query, openai_api_key, max_results)
            if vector_results:
                return vector_results
        
        # Fallback to keyword search
        return self._keyword_search(query, max_results)
    
    def _vector_search(self, query: str, openai_api_key: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Perform vector-based similarity search using pgvector.
        
        Args:
            query: Search query
            openai_api_key: OpenAI API key
            max_results: Maximum results to return
            
        Returns:
            List[Dict]: Similar documents with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query, openai_api_key)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Use RPC function to perform similarity search
            result = self.client.rpc(
                'match_gitlab_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': self.similarity_threshold,
                    'match_count': max_results
                }
            ).execute()
            
            if not result.data:
                return []
            
            # Format results
            documents = []
            for row in result.data:
                doc = {
                    'id': row['document_id'],
                    'title': row['title'],
                    'content': row['content'],
                    'url': row['url'],
                    'section': row['section'],
                    'keywords': json.loads(row['keywords']) if row['keywords'] else [],
                    'similarity_score': float(row['similarity'])
                }
                documents.append(doc)
            
            logger.debug(f"Vector search found {len(documents)} similar documents")
            return documents
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _keyword_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based search using PostgreSQL full-text search.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List[Dict]: Documents matching keywords
        """
        try:
            # Use PostgreSQL full-text search
            result = self.client.table(self.table_name).select(
                'document_id, title, content, url, section, keywords'
            ).text_search(
                'content', query, type='websearch'
            ).limit(max_results).execute()
            
            if not result.data:
                return []
            
            # Format results
            documents = []
            for row in result.data:
                doc = {
                    'id': row['document_id'],
                    'title': row['title'],
                    'content': row['content'],
                    'url': row['url'],
                    'section': row['section'],
                    'keywords': json.loads(row['keywords']) if row['keywords'] else [],
                    'similarity_score': 0.8  # Default score for keyword matches
                }
                documents.append(doc)
            
            logger.debug(f"Keyword search found {len(documents)} matching documents")
            return documents
            
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Get document count
            count_result = self.client.table(self.table_name).select("count", count="exact").execute()
            total_docs = count_result.count if count_result.count is not None else 0
            
            # Get documents with embeddings count
            embedding_result = self.client.table(self.table_name).select("count", count="exact").not_.is_("embedding", "null").execute()
            docs_with_embeddings = embedding_result.count if embedding_result.count is not None else 0
            
            return {
                'total_documents': total_docs,
                'documents_with_embeddings': docs_with_embeddings,
                'embedding_model': 'text-embedding-ada-002',
                'embedding_dimension': 1536,
                'similarity_threshold': self.similarity_threshold,
                'max_results': self.max_results,
                'table_name': self.table_name,
                'status': 'connected'
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_documents': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def is_available(self) -> bool:
        """Check if Supabase is available and accessible."""
        try:
            result = self.client.table(self.table_name).select("count", count="exact").limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase not available: {e}")
            return False
    
    def clear(self):
        """Clear all documents from the table."""
        try:
            result = self.client.table(self.table_name).delete().neq('document_id', 'impossible_id').execute()
            logger.info("Cleared all documents from Supabase")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")


# Factory functions for Streamlit integration

@st.cache_resource
def create_supabase_vector_store(supabase_url: str, 
                               supabase_key: str,
                               table_name: str = "gitlab_documents",
                               similarity_threshold: float = 0.3,
                               max_results: int = 5) -> SupabaseVectorStore:
    """
    Create a cached SupabaseVectorStore instance.
    
    This function is cached by Streamlit's resource cache.
    """
    return SupabaseVectorStore(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        table_name=table_name,
        similarity_threshold=similarity_threshold,
        max_results=max_results
    )


def create_supabase_rag_system(supabase_url: str,
                             supabase_key: str,
                             openai_api_key: Optional[str] = None,
                             table_name: str = "gitlab_documents") -> Optional[SupabaseVectorStore]:
    """
    Create a complete Supabase RAG system.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase anon/service role key
        openai_api_key: OpenAI API key for embeddings
        table_name: Database table name
        
    Returns:
        SupabaseVectorStore: Ready-to-use vector store or None if failed
    """
    try:
        store = create_supabase_vector_store(supabase_url, supabase_key, table_name)
        
        if store.is_available():
            return store
        else:
            st.error("Failed to connect to Supabase")
            return None
            
    except Exception as e:
        st.error(f"Error creating Supabase RAG system: {e}")
        logger.exception("Detailed error creating Supabase RAG system")
        return None 