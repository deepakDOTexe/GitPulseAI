"""
RAG System for GitLab Handbook Assistant

This module implements the core Retrieval-Augmented Generation system,
combining document retrieval with OpenAI's language models for intelligent responses.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

from openai import OpenAI
from src.config import config
from src.data_processor import GitLabDataProcessor, load_and_process_gitlab_data
from src.vector_store import SimpleVectorStore, create_vector_store_from_documents

logger = logging.getLogger(__name__)

class GitLabRAGSystem:
    """
    Main RAG system for GitLab Handbook Assistant.
    
    Integrates document processing, vector storage, and language model generation
    to provide intelligent, context-aware responses to user queries.
    """
    
    def __init__(self, api_key: Optional[str] = None, data_file: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            api_key: OpenAI API key. Uses config if None.
            data_file: Path to GitLab data file. Uses config if None.
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize components
        self.data_processor = GitLabDataProcessor(data_file)
        self.vector_store = SimpleVectorStore(api_key)
        
        # Configuration
        self.llm_model = config.LLM_MODEL
        self.max_context_length = config.MAX_CONTEXT_LENGTH
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        
        # System state
        self.is_initialized = False
        self.initialization_error = None
        
        # System prompt for GitLab context
        self.system_prompt = """You are a helpful assistant for GitLab's Handbook and Direction pages. 
You have access to GitLab's internal documentation about their company culture, values, processes, 
and policies. Your role is to:

1. Provide accurate, helpful information based on the GitLab documentation
2. Always cite your sources when providing information
3. Be conversational and friendly while maintaining professionalism
4. If you don't know something, admit it rather than guessing
5. Focus on GitLab-specific context and practices
6. Help users understand GitLab's "build in public" philosophy and culture

When responding:
- Use the provided context from GitLab documentation
- Include relevant source citations
- Be specific and actionable when possible
- Maintain GitLab's friendly, inclusive tone"""
        
        logger.info("Initialized GitLabRAGSystem")
    
    def initialize(self, force_reload: bool = False) -> bool:
        """
        Initialize the RAG system by loading data and building vector store.
        
        Args:
            force_reload: Force reloading of data even if already initialized
            
        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized and not force_reload:
            logger.info("RAG system already initialized")
            return True
        
        try:
            logger.info("Initializing RAG system...")
            
            # Load and process data
            logger.info("Loading GitLab documentation data...")
            if not self.data_processor.load_data():
                raise Exception("Failed to load GitLab data")
            
            processed_chunks = self.data_processor.process_documents()
            if not processed_chunks:
                raise Exception("No documents processed successfully")
            
            # Build vector store
            logger.info(f"Building vector store with {len(processed_chunks)} chunks...")
            successful_adds = self.vector_store.add_documents(processed_chunks, fallback_mode=True)
            
            if successful_adds == 0:
                raise Exception("Failed to add any documents to vector store")
            elif successful_adds < len(processed_chunks):
                logger.warning(f"Only {successful_adds}/{len(processed_chunks)} documents added successfully")
                logger.info("Some documents may be using fallback mode (keyword search only)")
            
            # Log system statistics
            data_stats = self.data_processor.get_document_stats()
            vector_stats = self.vector_store.get_stats()
            
            logger.info(f"RAG system initialized successfully:")
            logger.info(f"  - Documents: {data_stats.get('total_documents', 0)}")
            logger.info(f"  - Chunks: {data_stats.get('total_chunks', 0)}")
            logger.info(f"  - Vector store: {vector_stats.get('total_documents', 0)} embeddings")
            logger.info(f"  - Embedding dimension: {vector_stats.get('embedding_dimension', 0)}")
            
            self.is_initialized = True
            self.initialization_error = None
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize RAG system: {e}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    def _search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using both vector and keyword search.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Relevant documents with scores
        """
        # Use vector search as primary method
        vector_results = self.vector_store.search(query, max_results)
        
        # Fallback to keyword search if vector search fails or returns few results
        if len(vector_results) < max_results // 2:
            keyword_results = self.data_processor.search_chunks_by_keywords(
                query, max_results - len(vector_results)
            )
            
            # Combine results, avoiding duplicates
            existing_ids = {doc['id'] for doc in vector_results}
            for doc in keyword_results:
                if doc['id'] not in existing_ids:
                    vector_results.append(doc)
        
        return vector_results[:max_results]
    
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
            url = doc.get('url', '')
            
            # Format document reference
            doc_header = f"Document {i}: {title}"
            if section:
                doc_header += f" (Section: {section})"
            
            context_part = f"{doc_header}\n"
            if url:
                context_part += f"Source: {url}\n"
            context_part += f"Content: {content}\n"
            
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate response using OpenAI's language model.
        
        Args:
            query: User query
            context: Retrieved document context
            conversation_history: Previous conversation turns
            
        Returns:
            str: Generated response
        """
        try:
            # Build conversation messages
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                for turn in conversation_history[-5:]:  # Keep last 5 turns
                    messages.append({"role": "user", "content": turn.get("user", "")})
                    messages.append({"role": "assistant", "content": turn.get("assistant", "")})
            
            # Add current query with context
            user_message = f"""Based on the following GitLab documentation, please answer the user's question:

CONTEXT:
{context}

USER QUESTION: {query}

Please provide a helpful response based on the GitLab documentation provided above. Include relevant source citations where appropriate."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
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
                "similarity_score": doc.get('similarity_score', doc.get('relevance_score', 0))
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
    
    def query(self, user_query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
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
                "answer": "I'm sorry, but the system is not properly initialized. Please try again later.",
                "sources": [],
                "error": self.initialization_error or "System not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Search for relevant documents
            relevant_docs = self._search_documents(user_query)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the GitLab documentation for your question. Could you try rephrasing your query or asking about a different topic?",
                    "sources": [],
                    "warning": "No relevant documents found",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Format context and generate response
            context = self._format_context(relevant_docs)
            response = self._generate_response(user_query, context, conversation_history)
            
            # Format sources
            sources = self._format_sources(relevant_docs)
            
            logger.info(f"Query processed successfully, {len(sources)} sources found")
            
            return {
                "answer": response,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your query: {str(e)}",
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
            "config": {
                "llm_model": self.llm_model,
                "embedding_model": self.vector_store.embedding_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_context_length": self.max_context_length
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if self.is_initialized:
            stats["data_processor"] = self.data_processor.get_document_stats()
            stats["vector_store"] = self.vector_store.get_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the RAG system.
        
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
        
        # Check vector store
        try:
            vector_stats = self.vector_store.get_stats()
            if vector_stats.get("total_documents", 0) > 0:
                health["checks"].append({
                    "name": "vector_store",
                    "status": "passed",
                    "message": f"Stored {vector_stats.get('total_documents', 0)} documents"
                })
            else:
                health["status"] = "unhealthy"
                health["checks"].append({
                    "name": "vector_store",
                    "status": "failed",
                    "message": "No documents in vector store"
                })
        except Exception as e:
            health["status"] = "unhealthy"
            health["checks"].append({
                "name": "vector_store",
                "status": "failed",
                "message": str(e)
            })
        
        # Test OpenAI API connection
        try:
            test_response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            health["checks"].append({
                "name": "openai_api",
                "status": "passed",
                "message": "API connection successful"
            })
        except Exception as e:
            health["status"] = "unhealthy"
            health["checks"].append({
                "name": "openai_api",
                "status": "failed",
                "message": str(e)
            })
        
        return health

def create_rag_system(api_key: Optional[str] = None, data_file: Optional[str] = None) -> GitLabRAGSystem:
    """
    Convenience function to create and initialize a RAG system.
    
    Args:
        api_key: OpenAI API key
        data_file: Path to GitLab data file
        
    Returns:
        GitLabRAGSystem: Initialized RAG system
    """
    rag_system = GitLabRAGSystem(api_key, data_file)
    
    if rag_system.initialize():
        logger.info("RAG system created and initialized successfully")
    else:
        logger.error(f"Failed to initialize RAG system: {rag_system.initialization_error}")
    
    return rag_system

# Export main classes and functions
__all__ = [
    'GitLabRAGSystem',
    'create_rag_system'
] 