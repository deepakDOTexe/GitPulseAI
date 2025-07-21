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

# Page configuration
st.set_page_config(
    page_title="GitPulseAI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, minimal CSS following a design system
st.markdown("""
<style>
    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean typography */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: #ecfdf5;
        color: #059669;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        border: 1px solid #d1fae5;
    }
    
    /* Chat containers */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    .message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 0.75rem;
        line-height: 1.6;
    }
    
    .user-message {
        background: #f3f4f6;
        border-left: 3px solid #3b82f6;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #10b981;
    }
    
    .message-role {
        font-weight: 600;
        font-size: 0.875rem;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    /* Input styling */
    .stTextInput input {
        border-radius: 0.5rem !important;
        border: 1px solid #d1d5db !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Button styling */
    .stButton button {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    
    .stButton button:hover {
        background: #2563eb !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        padding: 1rem;
    }
    
    .example-btn {
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.5rem;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        color: #374151;
        font-size: 0.875rem;
        text-align: left;
        cursor: pointer;
    }
    
    .example-btn:hover {
        background: #f3f4f6;
        border-color: #d1d5db;
    }
    
    /* Welcome message */
    .welcome-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .welcome-text {
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

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

def main():
    # Header
    st.markdown('<div class="main-title">GitPulseAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">GitLab Handbook Assistant</div>', unsafe_allow_html=True)

    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    if not rag_system:
        st.error("Failed to initialize the RAG system. Please check your configuration.")
        st.stop()

    # Status indicator
    status_text = "🌩️ Cloud Mode: Supabase + Google Gemini" if USE_SUPABASE else "💻 Local Mode: Hybrid embeddings + Google Gemini"
    st.markdown(f'<div class="status-badge">{status_text}</div>', unsafe_allow_html=True)

    # Sidebar with example questions
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### 💡 Example Questions")
        
        example_questions = [
            "What are GitLab's core values?",
            "How does GitLab handle remote work?",
            "What is GitLab's anti-harassment policy?",
            "Tell me about GitLab's diversity and inclusion initiatives",
            "How does GitLab approach collaboration?",
            "What are the company's results-focused principles?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"ex_{i}", help="Click to ask this question"):
                st.session_state.selected_question = question
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize selected question
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""

    # Welcome message (only show when no chat history)
    if not st.session_state.chat_history:
        st.markdown('''
        <div class="welcome-message">
            <div class="welcome-title">👋 Welcome to GitPulseAI!</div>
            <div class="welcome-text">
                I'm here to help you explore GitLab's handbook and policies.<br><br>
                <strong>Ask me about:</strong> Company values • Remote work • Policies • Culture • Best practices<br><br>
                <em>Powered by Google Gemini and Supabase</em>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'''
            <div class="message user-message">
                <div class="message-role">👤 You</div>
                {message["content"]}
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="message assistant-message">
                <div class="message-role">🤖 GitPulseAI</div>
                {message["content"]}
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Input section
    st.markdown("---")
    
    # Handle selected question from sidebar
    user_input = st.session_state.selected_question if st.session_state.selected_question else ""
    if st.session_state.selected_question:
        st.session_state.selected_question = ""  # Clear it
    
    # Text input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input(
            "Ask about GitLab's handbook:",
            value=user_input,
            placeholder="e.g., What are GitLab's core values?",
            key="question_input"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)

    # Process question
    if (ask_button and user_question.strip()) or user_input.strip():
        question_to_process = user_question.strip() or user_input.strip()
        
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user", 
            "content": question_to_process
        })
        
        # Show thinking indicator
        with st.spinner("🤔 Thinking..."):
            try:
                # Initialize RAG system if needed
                if not rag_system.initialize():
                    st.error("❌ Failed to initialize RAG system")
                    return
                
                # Get response from RAG system
                response_data = rag_system.query(question_to_process, st.session_state.chat_history[:-1])
                
                if response_data.get("status") == "success":
                    response = response_data.get("response", "I couldn't generate a response.")
                else:
                    response = f"❌ {response_data.get('message', 'Something went wrong.')}"
                
                # Add assistant response to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Rerun to display new messages
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.rerun()

if __name__ == "__main__":
    main() 