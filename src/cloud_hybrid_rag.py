"""
Cloud-Optimized Hybrid RAG System for Streamlit Community Cloud

This module implements a RAG system optimized for Streamlit Cloud:
- Uses precomputed embeddings (no model downloads)
- Fast startup with Streamlit caching
- Minimal memory footprint
- Google Gemini LLM (generous free tier)
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.config import config
from src.cloud_vector_store import CloudVectorStore, create_cloud_rag_system
from src.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)

class CloudHybridRAGSystem:
    """
    Cloud-optimized hybrid RAG system for fast Streamlit deployment.
    
    Features:
    - Precomputed embeddings (no model downloads)
    - Streamlit caching for fast startup
    - Google Gemini LLM
    - Keyword fallback search
    """
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 precomputed_data_file: str = "data/precomputed_gitlab_comprehensive_handbook.json",
                 gemini_model: str = "gemini-1.5-flash",
                 similarity_threshold: float = 0.3,
                 max_results: int = 5):
        """
        Initialize the cloud hybrid RAG system.
        
        Args:
            gemini_api_key: Google Gemini API key
            precomputed_data_file: Path to precomputed embeddings file
            gemini_model: Google Gemini model name
            similarity_threshold: Minimum similarity for search results
            max_results: Maximum search results
        """
        self.gemini_api_key = gemini_api_key or config.GEMINI_API_KEY
        self.precomputed_data_file = precomputed_data_file
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # Initialize components
        self.vector_store: Optional[CloudVectorStore] = None
        self.llm = GeminiLLM(self.gemini_api_key, gemini_model)
        
        # System state
        self.is_initialized = False
        self.initialization_error = None
        
        logger.info("Initialized CloudHybridRAGSystem")
    
    @st.cache_resource
    def _initialize_components(_self):
        """Initialize system components with Streamlit caching."""
        try:
            logger.info("Initializing cloud hybrid RAG system components...")
            
            # Check Gemini availability
            if not _self.llm.is_available():
                raise Exception("Google Gemini not available. Please check your API key.")
            
            # Initialize vector store with precomputed embeddings
            vector_store = create_cloud_rag_system(
                data_file=_self.precomputed_data_file,
                similarity_threshold=_self.similarity_threshold,
                max_results=_self.max_results
            )
            
            if not vector_store:
                raise Exception("Failed to initialize cloud vector store")
            
            logger.info("Cloud hybrid RAG system components initialized successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def initialize(self) -> bool:
        """
        Initialize the cloud RAG system.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            if self.is_initialized:
                return True
            
            logger.info("Initializing Cloud Hybrid RAG system...")
            
            # Use cached initialization
            self.vector_store = self._initialize_components()
            
            # Log system statistics
            if self.vector_store:
                self._log_system_stats()
                self.is_initialized = True
                self.initialization_error = None
                return True
            else:
                raise Exception("Vector store initialization failed")
                
        except Exception as e:
            error_msg = f"Failed to initialize Cloud Hybrid RAG system: {e}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    def _log_system_stats(self):
        """Log system statistics after initialization."""
        if not self.vector_store:
            return
        
        vector_stats = self.vector_store.get_stats()
        llm_info = self.llm.get_model_info()
        
        logger.info("Cloud Hybrid RAG system initialized successfully:")
        logger.info(f"  - Documents: {vector_stats.get('total_documents', 0)}")
        logger.info(f"  - Embedding model: {vector_stats.get('embedding_model', 'unknown')}")
        logger.info(f"  - Embedding dimension: {vector_stats.get('embedding_dimension', 0)}")
        logger.info(f"  - Characters: {vector_stats.get('total_characters', 0):,}")
        logger.info(f"  - LLM: {llm_info.get('model_name', 'unknown')}")
        logger.info(f"  - Mode: Cloud-optimized (precomputed embeddings)")
    
    def _search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Use keyword-based search (since we don't have embedding model in cloud)
            results = self.vector_store.search(query, max_results=self.max_results)
            
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
            Dict containing answer, sources, and metadata
        """
        if not self.is_initialized:
            if not self.initialize():
                return {
                    "answer": "I'm sorry, but I'm not properly initialized. Please check the system configuration.",
                    "sources": [],
                    "error": self.initialization_error,
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
                    "Anti-harassment policies",
                    "Hiring and onboarding processes",
                    "Leadership and management practices"
                ]
                suggestions = "\n".join([f"• {topic}" for topic in available_topics])
                
                return {
                    "answer": f"""I couldn't find specific information about your question in the current GitLab documentation. 

However, I can help you with these GitLab topics:

{suggestions}

Could you try asking about one of these available topics instead? Or rephrase your question to be more specific about GitLab policies, processes, or culture.""",
                    "sources": [],
                    "warning": "No relevant documents found - showing available topics",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Format context for LLM
            context = self._format_context(relevant_docs)
            
            # Generate response using Gemini
            response = self.llm.generate_response(user_query, context, conversation_history or [])
            
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
                "answer": response,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "num_sources": len(sources),
                "system": "Cloud (Precomputed + Gemini)"
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            
            return {
                "answer": "I'm sorry, I encountered an error while processing your question. Please try again or rephrase your question.",
                "sources": [],
                "error": error_msg,
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
        
        # Check vector store
        vector_available = self.vector_store and self.vector_store.is_available()
        checks.append({
            "name": "Vector Store",
            "status": "passed" if vector_available else "failed",
            "message": "Available" if vector_available else "Not available"
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
            "system_type": "Cloud Hybrid RAG",
            "initialized": self.is_initialized,
            "llm_model": self.llm.get_model_info().get('model_name', 'unknown'),
            "deployment_mode": "Streamlit Cloud Optimized"
        }
        
        if self.vector_store:
            vector_stats = self.vector_store.get_stats()
            info.update({
                "documents": vector_stats.get('total_documents', 0),
                "embedding_model": vector_stats.get('embedding_model', 'unknown'),
                "embedding_dimension": vector_stats.get('embedding_dimension', 0)
            })
        
        return info


# Factory function for easy integration with Streamlit
@st.cache_resource
def create_cached_cloud_rag_system(gemini_api_key: Optional[str] = None,
                                 precomputed_data_file: str = "data/precomputed_gitlab_comprehensive_handbook.json") -> CloudHybridRAGSystem:
    """
    Create a cached CloudHybridRAGSystem instance.
    
    This function is cached by Streamlit for fast reloads.
    """
    return CloudHybridRAGSystem(
        gemini_api_key=gemini_api_key,
        precomputed_data_file=precomputed_data_file
    ) 