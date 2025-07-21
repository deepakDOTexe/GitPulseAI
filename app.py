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

# Simple, clean CSS
st.markdown("""
<style>
    /* Clean, minimal styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Header */
    .header {
        padding: 1rem 0 2rem 0;
        text-align: center;
        border-bottom: 1px solid #e6e6e6;
        margin-bottom: 2rem;
    }
    
    .title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }
    
    .status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #dcfce7;
        color: #166534;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #e5e7eb;
    }
    
    .user-message {
        background-color: #eff6ff;
        border-left-color: #3b82f6;
    }
    
    .assistant-message {
        background-color: #f9fafb;
        border-left-color: #10b981;
    }
    
    .message-role {
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        color: #374151;
    }
    
    .message-content {
        color: #1f2937;
        line-height: 1.6;
    }
    
    /* Sidebar */
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .sidebar-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    
    /* Example buttons */
    .stButton > button {
        width: 100%;
        text-align: left;
        background-color: #f8fafc;
        color: #374151;
        border: 1px solid #d1d5db;
        border-radius: 0.375rem;
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    
    .stButton > button:hover {
        background-color: #f1f5f9;
        border-color: #9ca3af;
    }
    
    /* Input area */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 1px solid #d1d5db;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }
    
    /* Primary button */
    button[kind="primary"] {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    button[kind="primary"]:hover {
        background-color: #2563eb;
    }
    
    /* Welcome message */
    .welcome {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .welcome h3 {
        color: #1e40af;
        margin-bottom: 1rem;
    }
    
    .welcome p {
        color: #1f2937;
        margin-bottom: 0;
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
    # Simple header
    status_text = "🌩️ Cloud Mode" if USE_SUPABASE else "💻 Local Mode"
    st.markdown(f'''
    <div class="header">
        <div class="title">GitPulseAI</div>
        <div class="subtitle">GitLab Handbook Assistant</div>
        <div class="status">{status_text}</div>
    </div>
    ''', unsafe_allow_html=True)

    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    if not rag_system:
        st.error("Failed to initialize the RAG system. Please check your configuration.")
        st.stop()

    # Sidebar with example questions
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">💡 Example Questions</div>', unsafe_allow_html=True)
        
        example_questions = [
            "What are GitLab's core values?",
            "How does GitLab handle remote work?",
            "What is GitLab's anti-harassment policy?",
            "Tell me about GitLab's diversity initiatives",
            "How does GitLab approach collaboration?",
            "What are the company's results-focused principles?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"ex_{i}"):
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
        <div class="welcome">
            <h3>👋 Welcome to GitPulseAI!</h3>
            <p>Ask me anything about GitLab's handbook, policies, values, and culture.<br>
            Click an example question from the sidebar or type your own question below.</p>
        </div>
        ''', unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'''
            <div class="chat-message user-message">
                <div class="message-role">👤 You</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="chat-message assistant-message">
                <div class="message-role">🤖 GitPulseAI</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)

    # Input area
    st.markdown("---")
    
    # Handle selected question from sidebar
    user_input = st.session_state.selected_question if st.session_state.selected_question else ""
    if st.session_state.selected_question:
        st.session_state.selected_question = ""  # Clear it
    
    # Text input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input(
            "Ask about GitLab:",
            value=user_input,
            placeholder="e.g., What are GitLab's core values?",
            key="question_input"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary")

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