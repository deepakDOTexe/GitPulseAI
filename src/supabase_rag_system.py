"""
Supabase RAG System for GitPulseAI

Complete RAG system using Supabase (PostgreSQL + pgvector) for document storage
and Google Gemini for both LLM generation and embeddings.

This is the main integration point for Streamlit Cloud deployment.
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.supabase_vector_store import SupabaseVectorStore, create_supabase_vector_store
from src.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)

class SupabaseRAGSystem:
    """
    Complete RAG system using Supabase + Google Gemini.
    
    Features:
    - Supabase PostgreSQL + pgvector for document storage
    - Google Gemini embeddings for semantic search
    - Google Gemini LLM for response generation
    - Hybrid search (vector + full-text)
    - Streamlit Cloud optimized
    """
    
    def __init__(self, 
                 supabase_url: str,
                 supabase_key: str,
                 gemini_api_key: str,
                 table_name: str = "gitlab_documents",
                 gemini_model: str = "gemini-1.5-flash",
                 similarity_threshold: float = 0.3,
                 max_results: int = 5):
        """
        Initialize the Supabase RAG system.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/service role key
            gemini_api_key: Google Gemini API key (for both LLM and embeddings)
            table_name: Database table name
            gemini_model: Gemini model name
            similarity_threshold: Minimum similarity for search results
            max_results: Maximum search results
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.gemini_api_key = gemini_api_key
        self.table_name = table_name
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # Initialize components
        self.vector_store: Optional[SupabaseVectorStore] = None
        self.llm = GeminiLLM(self.gemini_api_key, gemini_model)
        
        # System state
        self.is_initialized = False
        self.initialization_error = None
        
        logger.info("Initialized SupabaseRAGSystem with Google Gemini")
    
    def initialize(self) -> bool:
        """
        Initialize the RAG system components.
        
        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized:
            return True
        
        try:
            logger.info("Initializing Supabase RAG system...")
            
            # Check Gemini availability
            if not self.llm.is_available():
                raise Exception("Google Gemini not available. Please check your API key.")
            
            # Initialize vector store
            self.vector_store = create_supabase_vector_store(
                supabase_url=self.supabase_url,
                supabase_key=self.supabase_key,
                table_name=self.table_name,
                similarity_threshold=self.similarity_threshold,
                max_results=self.max_results
            )
            
            if not self.vector_store:
                raise Exception("Failed to initialize Supabase vector store")
            
            # Test basic connectivity
            try:
                stats = self.vector_store.get_stats()
                logger.info(f"Connected to Supabase with {stats.get('total_documents', 0)} documents")
            except Exception as e:
                logger.warning(f"Could not get stats but vector store created: {e}")
            
            self.is_initialized = True
            self.initialization_error = None
            
            logger.info("Supabase RAG system initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize Supabase RAG system: {e}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    def _search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant documents using Supabase with Google Gemini embeddings."""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Use Google Gemini embeddings for search
            results = self.vector_store.search(
                query=query, 
                gemini_api_key=self.gemini_api_key,  # Use Gemini instead of OpenAI
                max_results=self.max_results
            )
            
            logger.debug(f"Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into context for LLM."""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Unknown')
            content = doc.get('content', '')
            url = doc.get('url', '')
            
            # Truncate content to reasonable length
            max_chars = 1000
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            context_part = f"Document {i}: {title}\nSource: {url}\nContent: {content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def query(self, user_query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            user_query: User's question
            conversation_history: Previous conversation context
            
        Returns:
            Dict containing response, sources, and metadata
        """
        if not self.is_initialized:
            if not self.initialize():
                return {
                    "status": "error",
                    "message": f"System not initialized: {self.initialization_error}",
                    "response": "I'm sorry, but I'm not properly initialized. Please check the system configuration.",
                    "sources": [],
                    "timestamp": datetime.now().isoformat()
                }
        
        try:
            logger.info(f"Processing query: {user_query[:100]}...")
            
            # Search for relevant documents
            relevant_docs = self._search_documents(user_query)
            
            # Handle case where no relevant documents are found
            if not relevant_docs:
                available_topics = [
                    "GitLab's core values and culture",
                    "Remote work practices and policies", 
                    "Engineering practices and workflows",
                    "Security guidelines and procedures",
                    "Anti-harassment and inclusion policies",
                    "Hiring and onboarding processes",
                    "Leadership and management practices"
                ]
                suggestions = "\n".join([f"• {topic}" for topic in available_topics])
                
                return {
                    "status": "success",
                    "response": f"""I couldn't find specific information about your question in the current GitLab documentation. 

However, I can help you with these GitLab topics:

{suggestions}

Could you try asking about one of these available topics instead? Or rephrase your question to be more specific about GitLab policies, processes, or culture.""",
                    "sources": [],
                    "warning": "No relevant documents found - showing available topics",
                    "timestamp": datetime.now().isoformat(),
                    "system": "Supabase + Google Gemini"
                }
            
            # Format context for LLM
            context = self._format_context(relevant_docs)
            
            # Generate response using Gemini
            llm_response = self.llm.generate_response(user_query, context, conversation_history or [])
            
            # Extract source information
            sources = []
            for doc in relevant_docs:
                source = {
                    "title": doc.get('title', 'Unknown'),
                    "url": doc.get('url', ''),
                    "section": doc.get('section', ''),
                    "similarity_score": doc.get('similarity_score', 0.0)
                }
                sources.append(source)
            
            return {
                "status": "success",
                "response": llm_response,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "num_sources": len(sources),
                "system": "Supabase + Google Gemini"
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            
            return {
                "status": "error",
                "message": error_msg,
                "response": "I'm sorry, I encountered an error while processing your question. Please try again or rephrase your question.",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the system."""
        checks = []
        
        # Check initialization
        checks.append({
            "name": "System Initialization",
            "status": "passed" if self.is_initialized else "failed",
            "message": self.initialization_error if not self.is_initialized else "OK"
        })
        
        # Check Gemini LLM
        llm_available = self.llm.is_available()
        checks.append({
            "name": "Gemini LLM",
            "status": "passed" if llm_available else "failed",
            "message": "Available" if llm_available else "Not available"
        })
        
        # Check Supabase vector store
        vector_available = self.vector_store is not None
        if vector_available and self.vector_store:
            try:
                stats = self.vector_store.get_stats()
                doc_count = stats.get('total_documents', 0)
                checks.append({
                    "name": "Supabase Vector Store",
                    "status": "passed",
                    "message": f"Connected with {doc_count} documents"
                })
            except Exception as e:
                checks.append({
                    "name": "Supabase Vector Store", 
                    "status": "failed",
                    "message": f"Connection error: {e}"
                })
        else:
            checks.append({
                "name": "Supabase Vector Store",
                "status": "failed",
                "message": "Not initialized"
            })
        
        # Overall status
        all_passed = all(check["status"] == "passed" for check in checks)
        
        return {
            "status": "healthy" if all_passed else "unhealthy",
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system."""
        info = {
            "system_type": "Supabase RAG",
            "initialized": self.is_initialized,
            "llm_model": self.llm.get_model_info().get('model_name', 'unknown'),
            "deployment_mode": "Supabase PostgreSQL + pgvector",
            "embedding_provider": "Google Gemini"
        }
        
        if self.vector_store:
            try:
                vector_stats = self.vector_store.get_stats()
                info.update({
                    "documents": vector_stats.get('total_documents', 0),
                    "documents_with_embeddings": vector_stats.get('documents_with_embeddings', 0),
                    "table_name": self.table_name
                })
            except Exception as e:
                info["stats_error"] = str(e)
        
        return info


# Factory function for easy integration with Streamlit
@st.cache_resource
def create_supabase_rag_system(supabase_url: str,
                             supabase_key: str,
                             gemini_api_key: str,
                             table_name: str = "gitlab_documents") -> SupabaseRAGSystem:
    """
    Create a cached SupabaseRAGSystem instance.
    
    This function is cached by Streamlit for fast reloads.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase anon/service role key
        gemini_api_key: Google Gemini API key (for both LLM and embeddings)
        table_name: Database table name
        
    Returns:
        SupabaseRAGSystem: Ready-to-use RAG system
    """
    return SupabaseRAGSystem(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        gemini_api_key=gemini_api_key,
        table_name=table_name
    ) 