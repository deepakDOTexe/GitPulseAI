"""
GitLab Handbook Assistant - Streamlit App

This is a Streamlit-based chat interface for the GitLab Handbook Assistant.
It uses a hybrid RAG system with Hugging Face embeddings and Google Gemini LLM.
"""

import streamlit as st
import logging
from typing import Dict, Any, List
from datetime import datetime
import traceback
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our hybrid RAG system
from src.hybrid_rag_system import create_hybrid_rag_system, HybridRAGSystem
from src.config import config

# Page configuration
st.set_page_config(
    page_title="GitLab Handbook Assistant",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        color: #FC6D26;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        word-wrap: break-word;
    }
    
    .user-message {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
    }
    
    .assistant-message {
        background-color: #f0f8e8;
        border-left: 4px solid #2ca02c;
    }
    
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    
    .system-status {
        font-size: 0.9rem;
        color: #666;
        margin-top: 1rem;
    }
    
    .error-message {
        background-color: #ffe6e6;
        border-left: 4px solid #ff4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .success-message {
        background-color: #e6ffe6;
        border-left: 4px solid #00cc00;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .warning-message {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'initialization_error' not in st.session_state:
    st.session_state.initialization_error = None

def initialize_rag_system() -> bool:
    """
    Initialize the hybrid RAG system with Hugging Face and Gemini.
    
    Returns:
        bool: True if initialization successful
    """
    try:
        logger.info("Initializing hybrid RAG system...")
        
        # Check for required API key
        if not config.GEMINI_API_KEY:
            st.session_state.initialization_error = "Google Gemini API key not configured. Please set GEMINI_API_KEY in your environment."
            return False
        
        # Create hybrid RAG system
        rag_system = create_hybrid_rag_system(
            gemini_api_key=config.GEMINI_API_KEY,
            data_file=None,  # Use default data
            embedding_model="all-MiniLM-L6-v2",  # Free Hugging Face model
            gemini_model="gemini-1.5-flash"  # Fast Gemini model
        )
        
        if not rag_system.is_initialized:
            st.session_state.initialization_error = rag_system.initialization_error
            return False
        
        st.session_state.rag_system = rag_system
        st.session_state.system_initialized = True
        st.session_state.initialization_error = None
        
        logger.info("Hybrid RAG system initialized successfully")
        return True
        
    except Exception as e:
        error_msg = f"Failed to initialize hybrid RAG system: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.session_state.initialization_error = error_msg
        return False

def get_response(query: str) -> Dict[str, Any]:
    """
    Get response from the hybrid RAG system.
    
    Args:
        query: User query
        
    Returns:
        Dict: Response with answer, sources, and metadata
    """
    if not st.session_state.system_initialized or not st.session_state.rag_system:
        return {
            "answer": "System not initialized. Please check the system status.",
            "sources": [],
            "error": "System not initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Convert conversation history to the expected format
        conversation_history = []
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                conversation_history.append({"user": msg["content"]})
            elif msg["role"] == "assistant":
                conversation_history.append({"assistant": msg["content"]})
        
        # Get response from hybrid RAG system
        response = st.session_state.rag_system.query(query, conversation_history)
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting response: {e}")
        logger.error(traceback.format_exc())
        return {
            "answer": f"I apologize, but I encountered an error while processing your query: {str(e)}",
            "sources": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def display_sources(sources: List[Dict[str, Any]]):
    """
    Display source information in the sidebar.
    
    Args:
        sources: List of source documents
    """
    if not sources:
        return
    
    st.sidebar.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        with st.sidebar.expander(f"Source {i}: {source.get('title', 'Unknown')}"):
            st.write(f"**Relevance:** {source.get('relevance', 'Unknown')}")
            st.write(f"**Score:** {source.get('similarity_score', 0):.3f}")
            
            if source.get('section'):
                st.write(f"**Section:** {source['section']}")
            
            if source.get('url'):
                st.write(f"**URL:** {source['url']}")

def display_system_stats():
    """Display system statistics in the sidebar."""
    if not st.session_state.rag_system:
        return
    
    try:
        stats = st.session_state.rag_system.get_system_stats()
        
        st.sidebar.subheader("🔧 System Status")
        
        # System type
        st.sidebar.write(f"**System:** {stats.get('system_type', 'Unknown')}")
        
        # Configuration
        config_info = stats.get('config', {})
        st.sidebar.write(f"**Embedding Model:** {config_info.get('embedding_model', 'Unknown')}")
        st.sidebar.write(f"**LLM Model:** {config_info.get('llm_model', 'Unknown')}")
        st.sidebar.write(f"**Free Tier:** ✅ Yes")
        
        # Data stats
        if stats.get('data_processor'):
            data_stats = stats['data_processor']
            st.sidebar.write(f"**Documents:** {data_stats.get('total_documents', 0)}")
            st.sidebar.write(f"**Chunks:** {data_stats.get('total_chunks', 0)}")
        
        # Vector store stats
        if stats.get('vector_store'):
            vector_stats = stats['vector_store']
            st.sidebar.write(f"**Embeddings:** {vector_stats.get('total_documents', 0)}")
            st.sidebar.write(f"**Dimension:** {vector_stats.get('embedding_dimension', 0)}")
        
        # LLM stats
        if stats.get('llm'):
            llm_stats = stats['llm']
            st.sidebar.write(f"**Rate Limit:** {llm_stats.get('rate_limit', 'Unknown')}")
            st.sidebar.write(f"**Available:** {'✅' if llm_stats.get('available') else '❌'}")
        
    except Exception as e:
        st.sidebar.error(f"Error getting system stats: {e}")

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🦊 GitLab Handbook Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize system if not already done
    if not st.session_state.system_initialized:
        with st.spinner("Initializing system with Hugging Face embeddings and Google Gemini..."):
            if initialize_rag_system():
                st.success("✅ System initialized successfully with Hugging Face + Gemini!")
                st.balloons()
            else:
                st.error(f"❌ Failed to initialize system: {st.session_state.initialization_error}")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Info")
        
        # System status
        if st.session_state.system_initialized:
            st.success("✅ System Ready")
            st.write("**Stack:** Hugging Face + Gemini")
            st.write("**Cost:** 100% Free*")
            st.write("*Gemini has generous free tier")
        else:
            st.error("❌ System Not Ready")
            if st.session_state.initialization_error:
                st.error(st.session_state.initialization_error)
        
        # Display system stats
        if st.session_state.system_initialized:
            display_system_stats()
        
        # Configuration info
        st.header("⚙️ Configuration")
        
        # API key status
        if config.GEMINI_API_KEY:
            st.success("✅ Gemini API key configured")
        else:
            st.error("❌ Gemini API key missing")
            st.write("Set GEMINI_API_KEY in your environment")
        
        # Reinitialize button
        if st.button("🔄 Reinitialize System"):
            st.session_state.system_initialized = False
            st.session_state.rag_system = None
            st.session_state.conversation_history = []
            st.rerun()
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Health check
        if st.session_state.system_initialized and st.button("🏥 Health Check"):
            try:
                health = st.session_state.rag_system.health_check()
                if health['status'] == 'healthy':
                    st.success("✅ System is healthy")
                else:
                    st.error("❌ System has issues")
                    for check in health['checks']:
                        if check['status'] == 'failed':
                            st.error(f"❌ {check['name']}: {check['message']}")
            except Exception as e:
                st.error(f"❌ Health check failed: {e}")
    
    # Main chat interface
    st.subheader("💬 Chat with the GitLab Handbook")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">👤 **You:** {message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f'<div class="chat-message assistant-message">🤖 **Assistant:** {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask about GitLab's handbook, culture, processes, or policies...")
    
    if user_input:
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show user message
        st.markdown(f'<div class="chat-message user-message">👤 **You:** {user_input}</div>', unsafe_allow_html=True)
        
        # Get response
        with st.spinner("Thinking..."):
            response = get_response(user_input)
        
        # Display response
        answer = response.get("answer", "I apologize, but I couldn't generate a response.")
        st.markdown(f'<div class="chat-message assistant-message">🤖 **Assistant:** {answer}</div>', unsafe_allow_html=True)
        
        # Add assistant response to history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display sources
        sources = response.get("sources", [])
        if sources:
            display_sources(sources)
        
        # Show system info
        if response.get("system"):
            st.markdown(f'<div class="system-status">Powered by {response["system"]} • {len(sources)} sources • {response.get("timestamp", "")}</div>', unsafe_allow_html=True)
        
        # Show errors or warnings
        if response.get("error"):
            st.error(f"Error: {response['error']}")
        elif response.get("warning"):
            st.warning(f"Warning: {response['warning']}")
    
    # Example questions
    st.subheader("💡 Example Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**About GitLab:**")
        if st.button("What are GitLab's core values?"):
            st.session_state.conversation_history.append({
                "role": "user",
                "content": "What are GitLab's core values?",
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()
        
        if st.button("How does GitLab handle remote work?"):
            st.session_state.conversation_history.append({
                "role": "user",
                "content": "How does GitLab handle remote work?",
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()
    
    with col2:
        st.write("**Processes & Culture:**")
        if st.button("What is GitLab's engineering process?"):
            st.session_state.conversation_history.append({
                "role": "user",
                "content": "What is GitLab's engineering process?",
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()
        
        if st.button("How does GitLab approach diversity and inclusion?"):
            st.session_state.conversation_history.append({
                "role": "user",
                "content": "How does GitLab approach diversity and inclusion?",
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🦊 **GitLab Handbook Assistant** • "
        "Powered by Hugging Face Embeddings + Google Gemini • "
        "100% Free Solution"
    )

if __name__ == "__main__":
    main() 