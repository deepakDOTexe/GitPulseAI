#!/usr/bin/env python3
"""
Simplified RAG System: TF-IDF + Google Gemini Only

This removes the Hugging Face complexity and just uses:
- TF-IDF for embeddings (completely offline)
- Google Gemini for LLM (generous free tier)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.config import config
from src.data_processor import GitLabDataProcessor
from src.simple_vector_store import SimpleVectorStore
from src.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    """
    Simplified RAG system using only TF-IDF + Gemini.
    No Hugging Face dependencies, no SSL issues.
    """
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 data_file: Optional[str] = None,
                 gemini_model: str = "gemini-1.5-flash"):
        """
        Initialize the simple RAG system.
        
        Args:
            gemini_api_key: Google Gemini API key
            data_file: Path to GitLab data file
            gemini_model: Google Gemini model name
        """
        self.gemini_api_key = gemini_api_key or config.GEMINI_API_KEY
        
        # Initialize components
        self.data_processor = GitLabDataProcessor(data_file)
        self.vector_store = SimpleVectorStore()
        self.llm = GeminiLLM(self.gemini_api_key, gemini_model)
        
        # Configuration
        self.max_context_length = config.MAX_CONTEXT_LENGTH
        
        # System state
        self.is_initialized = False
        self.initialization_error = None
        
        logger.info("Initialized SimpleRAGSystem (TF-IDF + Gemini)")
    
    def initialize(self, force_reload: bool = False) -> bool:
        """
        Initialize the RAG system by loading data and building vector store.
        
        Args:
            force_reload: Force reloading of data even if already initialized
            
        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized and not force_reload:
            logger.info("Simple RAG system already initialized")
            return True
        
        try:
            logger.info("Initializing Simple RAG system...")
            
            # Check Gemini availability
            if not self.llm.is_available():
                raise Exception("Google Gemini not available. Please check your API key.")
            
            # Load and process data
            logger.info("Loading GitLab documentation data...")
            if not self.data_processor.load_data():
                raise Exception("Failed to load GitLab data")
            
            processed_chunks = self.data_processor.process_documents()
            if not processed_chunks:
                raise Exception("No documents processed successfully")
            
            # Build TF-IDF vector store
            logger.info(f"Building TF-IDF vector store with {len(processed_chunks)} chunks...")
            successful_adds = self.vector_store.add_documents(processed_chunks)
            
            if successful_adds == 0:
                raise Exception("Failed to add any documents to vector store")
            
            # Log system statistics
            self._log_system_stats()
            
            self.is_initialized = True
            self.initialization_error = None
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize Simple RAG system: {e}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    def _log_system_stats(self):
        """Log system statistics after initialization."""
        data_stats = self.data_processor.get_document_stats()
        vector_stats = self.vector_store.get_stats()
        llm_info = self.llm.get_model_info()
        
        logger.info(f"Simple RAG system initialized successfully:")
        logger.info(f"  - Documents: {data_stats.get('total_documents', 0)}")
        logger.info(f"  - Chunks: {data_stats.get('total_chunks', 0)}")
        logger.info(f"  - Vector store: {vector_stats.get('total_documents', 0)} embeddings")
        logger.info(f"  - Embedding model: {vector_stats.get('embedding_model', 'unknown')}")
        logger.info(f"  - Embedding dimension: {vector_stats.get('embedding_dimension', 0)}")
        logger.info(f"  - LLM: {llm_info.get('model_name', 'unknown')}")
        logger.info(f"  - Rate limit: {llm_info.get('rate_limit', 'unknown')}")
    
    def query(self, user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query and return a response.
        
        Args:
            user_query: User's question
            conversation_history: Previous conversation turns
            
        Returns:
            Dict: Response with answer, sources, and metadata
        """
        if not self.is_initialized:
            return {
                "answer": "❌ System not initialized. Please check the logs for errors.",
                "sources": [],
                "error": self.initialization_error
            }
        
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Search for relevant documents using TF-IDF
            relevant_docs = self.vector_store.search(user_query)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the GitLab documentation for your question. Could you try rephrasing your query or asking about a different topic?",
                    "sources": [],
                    "query": user_query
                }
            
            # Format context and generate response using Gemini
            context = self._format_context(relevant_docs)
            response = self.llm.generate_response(user_query, context, conversation_history or [])
            
            # Format sources
            sources = self._format_sources(relevant_docs)
            
            return {
                "answer": response,
                "sources": sources,
                "query": user_query,
                "num_sources": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"❌ Error processing your question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context for the LLM."""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Unknown Document')
            content = doc.get('content', '')
            similarity = doc.get('similarity_score', 0)
            
            context_parts.append(f"Document {i}: {title} (Relevance: {similarity:.3f})\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for the response."""
        sources = []
        for doc in documents:
            source = {
                "title": doc.get('title', 'Unknown Document'),
                "url": doc.get('url', ''),
                "similarity_score": doc.get('similarity_score', 0),
                "section": doc.get('section', ''),
                "document_id": doc.get('id', '')
            }
            sources.append(source)
        return sources
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "system_type": "Simple (TF-IDF + Gemini)",
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "embedding_approach": "TF-IDF (offline)",
            "llm_approach": "Google Gemini (free tier)"
        }
        
        if self.is_initialized:
            stats["data_processor"] = self.data_processor.get_document_stats()
            stats["vector_store"] = self.vector_store.get_stats()
            stats["llm"] = self.llm.get_model_info()
        
        return stats

def create_simple_rag_system(gemini_api_key: Optional[str] = None,
                            data_file: Optional[str] = None,
                            gemini_model: str = "gemini-1.5-flash") -> SimpleRAGSystem:
    """
    Convenience function to create and initialize a simple RAG system.
    
    Args:
        gemini_api_key: Google Gemini API key
        data_file: Path to GitLab data file
        gemini_model: Google Gemini model name
        
    Returns:
        SimpleRAGSystem: Initialized simple RAG system
    """
    rag_system = SimpleRAGSystem(gemini_api_key, data_file, gemini_model)
    
    if rag_system.initialize():
        logger.info("Simple RAG system created and initialized successfully")
    else:
        logger.error("Failed to initialize simple RAG system")
    
    return rag_system

# Export main classes and functions
__all__ = [
    'SimpleRAGSystem',
    'create_simple_rag_system'
] 