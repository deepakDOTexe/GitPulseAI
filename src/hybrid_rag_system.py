"""
Hybrid RAG System for GitLab Handbook Assistant

This module implements a RAG system using:
- Hugging Face sentence-transformers for embeddings (free, local)
- Google Gemini for LLM generation (generous free tier)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st

from src.config import config
from src.data_processor import GitLabDataProcessor, load_and_process_gitlab_data
from src.local_vector_store import LocalVectorStore, create_local_vector_store_from_documents
from src.simple_vector_store import SimpleVectorStore, create_simple_vector_store_from_documents
from src.gemini_llm import GeminiLLM, create_gemini_llm

logger = logging.getLogger(__name__)

class HybridRAGSystem:
    """
    Hybrid RAG system combining:
    - Hugging Face embeddings (free, local) with TF-IDF fallback
    - Google Gemini LLM (generous free tier)
    - Performance optimizations with caching
    """
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 data_file: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 gemini_model: str = "gemini-1.5-flash",
                 safety_level: str = "standard",
                 use_simple_fallback: bool = True):
        """
        Initialize the hybrid RAG system.
        
        Args:
            gemini_api_key: Google Gemini API key
            data_file: Path to GitLab data file
            embedding_model: Hugging Face embedding model name
            gemini_model: Google Gemini model name
            safety_level: Safety level to use - "standard", "reduced", or "minimal"
            use_simple_fallback: Whether to use simple vector store as fallback
        """
        self.gemini_api_key = gemini_api_key or config.GEMINI_API_KEY
        self.use_simple_fallback = use_simple_fallback
        
        # Initialize components
        self.data_processor = GitLabDataProcessor(data_file)
        self.vector_store = LocalVectorStore(embedding_model)
        self.simple_vector_store = SimpleVectorStore() if use_simple_fallback else None
        self.llm = GeminiLLM(self.gemini_api_key, gemini_model, safety_level)
        
        # Track which vector store is being used
        self.using_simple_store = False
        
        # Configuration
        self.max_context_length = config.MAX_CONTEXT_LENGTH
        
        # System state
        self.is_initialized = False
        self.initialization_error = None
        
        # Performance improvements
        self._query_cache = {}  # Simple query cache
        self._max_cache_size = 50
        
        logger.info("Initialized HybridRAGSystem (HuggingFace + Gemini with TF-IDF fallback)")
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _cached_search(_self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Cached document search to improve performance."""
        try:
            if _self.using_simple_store and _self.simple_vector_store:
                return _self.simple_vector_store.search(query, max_results)
            elif _self.vector_store:
                return _self.vector_store.search(query, max_results)
            else:
                return []
        except Exception as e:
            logger.error(f"Error in cached search: {e}")
            return []
    
    def _get_cache_key(self, query: str, conversation_history: List[Dict]) -> str:
        """Generate cache key for query results."""
        history_str = str([msg.get("content", "")[:50] for msg in conversation_history[-3:]])
        return f"{query[:100]}_{hash(history_str)}"
    
    def _manage_cache(self):
        """Simple cache management to prevent memory bloat."""
        if len(self._query_cache) > self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._query_cache.keys())[:10]
            for key in oldest_keys:
                del self._query_cache[key]
    
    def initialize(self, force_reload: bool = False) -> bool:
        """
        Initialize the RAG system by loading data and building vector store.
        
        Args:
            force_reload: Force reloading of data even if already initialized
            
        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized and not force_reload:
            logger.info("Hybrid RAG system already initialized")
            return True
        
        try:
            logger.info("Initializing Hybrid RAG system...")
            
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
            
            # Try to use Hugging Face embeddings first
            logger.info(f"Attempting to build vector store with Hugging Face embeddings...")
            if self.vector_store.is_available():
                successful_adds = self.vector_store.add_documents(processed_chunks)
                if successful_adds > 0:
                    logger.info(f"Successfully initialized with Hugging Face embeddings ({successful_adds} documents)")
                    self.using_simple_store = False
                else:
                    logger.warning("Failed to add documents to Hugging Face vector store")
                    if self.use_simple_fallback:
                        logger.info("Falling back to simple TF-IDF vector store...")
                        self._initialize_simple_store(processed_chunks)
                    else:
                        raise Exception("Failed to add documents to vector store")
            else:
                logger.warning("Hugging Face model not available")
                if self.use_simple_fallback:
                    logger.info("Using simple TF-IDF vector store as fallback...")
                    self._initialize_simple_store(processed_chunks)
                else:
                    raise Exception("Hugging Face model not available and no fallback enabled")
            
            # Log system statistics
            self._log_system_stats()
            
            self.is_initialized = True
            self.initialization_error = None
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize Hybrid RAG system: {e}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    def _initialize_simple_store(self, processed_chunks: List[Dict[str, Any]]):
        """Initialize the simple vector store with processed chunks."""
        if not self.simple_vector_store:
            self.simple_vector_store = SimpleVectorStore()
        
        successful_adds = self.simple_vector_store.add_documents(processed_chunks)
        if successful_adds > 0:
            self.using_simple_store = True
            logger.info(f"Successfully initialized with TF-IDF vector store ({successful_adds} documents)")
        else:
            raise Exception("Failed to add documents to simple vector store")
    
    def _log_system_stats(self):
        """Log system statistics after initialization."""
        data_stats = self.data_processor.get_document_stats()
        
        if self.using_simple_store and self.simple_vector_store:
            vector_stats = self.simple_vector_store.get_stats()
            logger.info(f"Hybrid RAG system initialized successfully (TF-IDF fallback mode):")
        else:
            vector_stats = self.vector_store.get_stats()
            logger.info(f"Hybrid RAG system initialized successfully (Hugging Face mode):")
        
        llm_info = self.llm.get_model_info()
        
        logger.info(f"  - Documents: {data_stats.get('total_documents', 0)}")
        logger.info(f"  - Chunks: {data_stats.get('total_chunks', 0)}")
        logger.info(f"  - Vector store: {vector_stats.get('total_documents', 0)} embeddings")
        logger.info(f"  - Embedding model: {vector_stats.get('embedding_model', 'unknown')}")
        logger.info(f"  - Embedding dimension: {vector_stats.get('embedding_dimension', 0)}")
        logger.info(f"  - LLM: {llm_info.get('model_name', 'unknown')}")
        logger.info(f"  - Rate limit: {llm_info.get('rate_limit', 'unknown')}")
        logger.info(f"  - Using fallback: {'Yes' if self.using_simple_store else 'No'}")
    
    def _search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using the appropriate vector store.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Relevant documents with scores
        """
        if self.using_simple_store and self.simple_vector_store:
            return self.simple_vector_store.search(query, max_results)
        else:
            return self.vector_store.search(query, max_results)
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the LLM.
        
        Args:
            documents: Retrieved documents with metadata
            
        Returns:
            str: Formatted context string
        """
        if not documents:
            return "No relevant documentation found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Unknown Document')
            section = doc.get('section', '')
            content = doc.get('content', '')
            
            # Create clean context without URLs or technical metadata
            if section and section != title:
                context_header = f"{title} - {section}"
            else:
                context_header = title
            
            context_part = f"{context_header}:\n{content}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format source information for display.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List[Dict]: Formatted source information
        """
        sources = []
        
        for doc in documents:
            source = {
                "title": doc.get('title', 'Unknown Document'),
                "url": doc.get('url', ''),
                "section": doc.get('section', ''),
                "similarity_score": doc.get('similarity_score', 0)
            }
            
            # Add relevance indicator
            score = source["similarity_score"]
            if score > 0.8:
                source["relevance"] = "High"
            elif score > 0.6:
                source["relevance"] = "Medium"
            else:
                source["relevance"] = "Low"
            
            sources.append(source)
        
        return sources
    
    def query(self, user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response with sources.
        
        Args:
            user_query: User's question or query
            conversation_history: Previous conversation turns
            
        Returns:
            Dict: Response with answer, sources, and metadata
        """
        if not self.is_initialized:
            return {
                "response": "I'm sorry, but the system is not properly initialized. Please try again later.",
                "sources": [],
                "error": self.initialization_error or "System not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Search for relevant documents using the appropriate vector store
            relevant_docs = self._search_documents(user_query)
            used_fallback = False
            fallback_context = ""
            
            # If no results, try broader search terms for technical questions
            if not relevant_docs and any(term in user_query.lower() for term in 
                ['development', 'environment', 'setup', 'install', 'configure', 'technical', 'code', 'programming']):
                logger.info("No results for technical query, trying broader engineering search...")
                fallback_queries = [
                    "engineering practices",
                    "development workflow", 
                    "technical guidelines",
                    "security practices"
                ]
                
                for fallback_query in fallback_queries:
                    relevant_docs = self._search_documents(fallback_query)
                    if relevant_docs:
                        used_fallback = True
                        fallback_context = f"Note: I couldn't find specific information about '{user_query}', so I'm providing related information from GitLab's {fallback_query} documentation."
                        logger.info(f"Found results with fallback query: {fallback_query}")
                        break
            
            if not relevant_docs:
                # Try to provide helpful suggestions based on available topics
                available_topics = [
                    "GitLab's core values and culture",
                    "Remote work practices and policies", 
                    "Engineering practices and workflows",
                    "Security guidelines and procedures",
                    "Performance reviews and career development",
                    "Hiring processes and practices",
                    "Diversity, inclusion and belonging initiatives"
                ]
                
                suggestions = "\n".join([f"• {topic}" for topic in available_topics])
                
                return {
                    "response": f"""I couldn't find specific information about that topic in what I have access to right now. 

However, I'd love to help you with these GitLab topics:

{suggestions}

You could also try asking about:
• "What are GitLab's engineering practices?"
• "How does GitLab handle code reviews?"  
• "What security guidelines does GitLab follow?"

What would you like to know about?""",
                    "sources": [],
                    "warning": "No relevant documents found - showing available topics",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Format context and generate response using Gemini
            context = self._format_context(relevant_docs)
            
            # Add fallback context if used
            if used_fallback:
                context = f"{fallback_context}\n\n{context}"
            
            response = self.llm.generate_response(user_query, context, conversation_history or [])
            
            # Format sources
            sources = self._format_sources(relevant_docs)
            
            logger.info(f"Query processed successfully, {len(sources)} sources found")
            
            result = {
                "response": response,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "num_sources": len(sources),
                "system": "HuggingFace + Gemini"
            }
            
            # Add warning if fallback was used
            if used_fallback:
                result["warning"] = "Response based on related content rather than exact match"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I'm having a bit of trouble right now, but I'll be back up and running soon. Please try asking your question again!",
                "sources": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict: System statistics and health information
        """
        stats = {
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "system_type": "Hybrid (HuggingFace + Gemini with TF-IDF fallback)",
            "config": {
                "embedding_model": self.vector_store.model_name if hasattr(self.vector_store, 'model_name') else 'unknown',
                "llm_model": self.llm.model_name,
                "max_context_length": self.max_context_length,
                "free_tier": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if self.is_initialized:
            stats["data_processor"] = self.data_processor.get_document_stats()
            if self.using_simple_store and self.simple_vector_store:
                stats["vector_store"] = self.simple_vector_store.get_stats()
            else:
                stats["vector_store"] = self.vector_store.get_stats()
            stats["llm"] = self.llm.get_model_info()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the hybrid RAG system.
        
        Returns:
            Dict: Health check results
        """
        health = {
            "status": "healthy",
            "checks": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Check initialization
        if not self.is_initialized:
            health["status"] = "unhealthy"
            health["checks"].append({
                "name": "initialization",
                "status": "failed",
                "message": self.initialization_error or "System not initialized"
            })
            return health
        
        # Check data processor
        try:
            data_stats = self.data_processor.get_document_stats()
            if data_stats.get("total_chunks", 0) > 0:
                health["checks"].append({
                    "name": "data_processor",
                    "status": "passed",
                    "message": f"Processed {data_stats.get('total_chunks', 0)} chunks"
                })
            else:
                health["status"] = "unhealthy"
                health["checks"].append({
                    "name": "data_processor",
                    "status": "failed",
                    "message": "No processed chunks available"
                })
        except Exception as e:
            health["status"] = "unhealthy"
            health["checks"].append({
                "name": "data_processor",
                "status": "failed",
                "message": str(e)
            })
        
        # Check local vector store
        try:
            if self.using_simple_store and self.simple_vector_store:
                vector_stats = self.simple_vector_store.get_stats()
                health["checks"].append({
                    "name": "local_vector_store",
                    "status": "passed",
                    "message": f"Stored {vector_stats.get('total_documents', 0)} documents locally (TF-IDF)"
                })
            else:
                vector_stats = self.vector_store.get_stats()
                health["checks"].append({
                    "name": "local_vector_store",
                    "status": "passed",
                    "message": f"Stored {vector_stats.get('total_documents', 0)} documents locally (Hugging Face)"
                })
            if vector_stats.get("total_documents", 0) > 0:
                health["checks"].append({
                    "name": "local_vector_store",
                    "status": "passed",
                    "message": f"Stored {vector_stats.get('total_documents', 0)} documents locally"
                })
            else:
                health["status"] = "unhealthy"
                health["checks"].append({
                    "name": "local_vector_store",
                    "status": "failed",
                    "message": "No documents in local vector store"
                })
        except Exception as e:
            health["status"] = "unhealthy"
            health["checks"].append({
                "name": "local_vector_store",
                "status": "failed",
                "message": str(e)
            })
        
        # Check Gemini LLM
        if self.llm.is_available():
            health["checks"].append({
                "name": "gemini_llm",
                "status": "passed",
                "message": f"Gemini {self.llm.model_name} available"
            })
        else:
            health["status"] = "unhealthy"
            health["checks"].append({
                "name": "gemini_llm",
                "status": "failed",
                "message": "Gemini LLM not available"
            })
        
        return health

def create_hybrid_rag_system(gemini_api_key: Optional[str] = None,
                           data_file: Optional[str] = None,
                           embedding_model: str = "all-MiniLM-L6-v2",
                           gemini_model: str = "gemini-1.5-flash",
                           safety_level: str = "standard",
                           use_simple_fallback: bool = True) -> HybridRAGSystem:
    """
    Convenience function to create and initialize a hybrid RAG system.
    
    Args:
        gemini_api_key: Google Gemini API key
        data_file: Path to GitLab data file
        embedding_model: Hugging Face embedding model name
        gemini_model: Google Gemini model name
        safety_level: Safety level to use - "standard", "reduced", or "minimal"
        use_simple_fallback: Whether to use simple vector store as fallback
        
    Returns:
        HybridRAGSystem: Initialized hybrid RAG system
    """
    rag_system = HybridRAGSystem(gemini_api_key, data_file, embedding_model, gemini_model, safety_level, use_simple_fallback)
    
    if rag_system.initialize():
        logger.info("Hybrid RAG system created and initialized successfully")
    else:
        logger.error(f"Failed to initialize hybrid RAG system: {rag_system.initialization_error}")
    
    return rag_system

# Export main classes and functions
__all__ = [
    'HybridRAGSystem',
    'create_hybrid_rag_system'
] 