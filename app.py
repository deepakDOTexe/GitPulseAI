"""
GitLab Handbook Assistant - Streamlit App

This is a Streamlit-based chat interface for the GitLab Handbook Assistant.
It uses a hybrid RAG system with Hugging Face embeddings and Google Gemini LLM.
"""

import os
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# 1. App title
st.set_page_config(
    page_title="🔍💬 GitPulseAI Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detect deployment mode
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"

@st.cache_resource
def initialize_rag_system():
    """Initialize the appropriate RAG system based on environment."""
    try:
        if USE_SUPABASE:
            # Use Supabase cloud RAG system
            from src.supabase_rag_system import create_supabase_rag_system
            
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY") 
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            
            if not supabase_url or not supabase_key or not gemini_api_key:
                st.error("❌ Missing required environment variables: SUPABASE_URL, SUPABASE_KEY, or GEMINI_API_KEY")
                return None
            
            return create_supabase_rag_system(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                gemini_api_key=gemini_api_key
            )
        else:
            # Use local hybrid RAG system
            from src.hybrid_rag_system import HybridRAGSystem
            return HybridRAGSystem()
            
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        return None

# Initialize RAG system
rag_system = initialize_rag_system()
if not rag_system:
    st.error("Failed to initialize the RAG system. Please check your configuration.")
    st.stop()

# Sidebar
with st.sidebar:
    st.title('🔍💬 GitPulseAI')
    st.caption("GitLab Handbook Assistant")
    
    # Status with metrics
    status_text = "🌩️ Cloud Mode" if USE_SUPABASE else "💻 Local Mode"
    st.success(f'{status_text} - System Ready!', icon='✅')
    
    # System metrics
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.query_count)
    with col2:
        st.metric("Mode", "Cloud" if USE_SUPABASE else "Local")
    
    st.divider()
    
    # Example questions in categories
    st.subheader('💡 Example Questions')
    
    with st.expander("🎯 Core Values & Culture"):
        questions_values = [
            "What are GitLab's core values?",
            "How does GitLab approach collaboration?",
            "What are GitLab's diversity initiatives?"
        ]
        for i, question in enumerate(questions_values):
            if st.button(question, key=f"val_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.query_count += 1
                st.rerun()
    
    with st.expander("🏠 Remote Work & Policies"):
        questions_work = [
            "How does GitLab handle remote work?",
            "What is GitLab's anti-harassment policy?",
            "What are the company's results-focused principles?"
        ]
        for i, question in enumerate(questions_work):
            if st.button(question, key=f"work_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.query_count += 1
                st.rerun()
    
    st.divider()
    
    # Clear chat history button
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you explore GitLab's handbook today?"}]
        st.session_state.query_count = 0
        if 'last_sources' in st.session_state:
            del st.session_state.last_sources
    st.button('🗑️ Clear Chat History', on_click=clear_chat_history, use_container_width=True)
    
    # Last response sources
    if 'last_sources' in st.session_state and st.session_state.last_sources:
        st.divider()
        st.subheader('📚 Sources from Last Response')
        for i, source in enumerate(st.session_state.last_sources[:3]):  # Show top 3
            with st.expander(f"📄 {source.get('title', 'Unknown')[:30]}..."):
                if source.get('url'):
                    st.write(f"🔗 [View Source]({source['url']})")
                if source.get('similarity_score'):
                    st.write(f"📊 Relevance: {source['similarity_score']:.1%}")

# 2. Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you explore GitLab's handbook today?"}]

# Welcome banner (only show when no chat history except initial message)
if len(st.session_state.messages) == 1:
    st.info(
        "👋 **Welcome to GitPulseAI!** Ask me anything about GitLab's handbook, policies, values, and culture. "
        "Use the example questions in the sidebar to get started, or type your own question below.",
        icon="💬"
    )

# 2. Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Add response feedback for assistant messages (except first welcome)
        if message["role"] == "assistant" and i > 0:
            col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
            with col1:
                if st.button("👍", key=f"thumb_up_{i}", help="Helpful response"):
                    st.success("Thanks for your feedback!", icon="✅")
            with col2:
                if st.button("👎", key=f"thumb_down_{i}", help="Not helpful"):
                    st.info("Thanks for your feedback! We'll improve.", icon="📝")
            with col3:
                if st.button("📋", key=f"copy_{i}", help="Copy response"):
                    st.code(message["content"], language=None)

# 3. Create the LLM response generation function
def generate_gitpulse_response(prompt_input):
    """Generate response using GitPulseAI RAG system."""
    try:
        # Check if RAG system is available
        if not rag_system:
            return {"response": "❌ RAG system not available", "sources": []}
        
        # Initialize RAG system if needed
        if not rag_system.initialize():
            return {"response": "❌ Failed to initialize RAG system", "sources": []}
        
        # Get response from RAG system
        response_data = rag_system.query(prompt_input, st.session_state.messages[:-1])
        
        if response_data.get("status") == "success":
            return {
                "response": response_data.get("response", "I couldn't generate a response."),
                "sources": response_data.get("sources", [])
            }
        else:
            return {
                "response": f"❌ {response_data.get('message', 'Something went wrong.')}",
                "sources": []
            }
            
    except Exception as e:
        return {
            "response": f"❌ Sorry, I encountered an error: {str(e)}",
            "sources": []
        }

# 4. Accept prompt input
if prompt := st.chat_input("Ask about GitLab's handbook, policies, values, and culture...", key="main_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.query_count += 1
    with st.chat_message("user"):
        st.write(prompt)

# 5. Generate a new LLM response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("🤔 Thinking..."):
        response_data = generate_gitpulse_response(st.session_state.messages[-1]["content"])
    
    # Store sources for sidebar display
    if response_data["sources"]:
        st.session_state.last_sources = response_data["sources"]
    
    # Add assistant response to session state
    message = {"role": "assistant", "content": response_data["response"]}
    st.session_state.messages.append(message)
    st.rerun() 