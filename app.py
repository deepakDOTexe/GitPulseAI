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

# Custom CSS for better visibility and design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #7b1fa2;
    }
    .example-question {
        background-color: #f5f5f5;
        padding: 0.8rem;
        border-radius: 0.3rem;
        cursor: pointer;
        margin: 0.3rem 0;
        border: 1px solid #ddd;
        color: #333 !important;
    }
    .example-question:hover {
        background-color: #e0e0e0;
        border-color: #1976d2;
    }
    .footer {
        margin-top: 2rem;
        padding: 1rem;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
    }
    .system-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .status-cloud {
        background-color: #e8f5e8;
        color: #2e7d2e;
        border-left: 4px solid #4caf50;
    }
    .status-local {
        background-color: #fff3e0;
        color: #e65100;
        border-left: 4px solid #ff9800;
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
            
            if not supabase_url or not supabase_key:
                st.error("❌ Supabase credentials not found in environment variables")
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
    # Main header
    st.markdown('<div class="main-header">GitPulseAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">GitLab Handbook Assistant</div>', unsafe_allow_html=True)

    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    if not rag_system:
        st.error("Failed to initialize the RAG system. Please check your configuration.")
        st.stop()

    # System status indicator
    if USE_SUPABASE:
        st.markdown('''
        <div class="system-status status-cloud">
            🌩️ <strong>Cloud Mode:</strong> Using Supabase + Google Gemini embeddings
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="system-status status-local">
            💻 <strong>Local Mode:</strong> Using local embeddings + Google Gemini LLM
        </div>
        ''', unsafe_allow_html=True)

    # Sidebar with example questions
    with st.sidebar:
        st.header("📋 Example Questions")
        
        example_questions = [
            "What are GitLab's core values?",
            "How does GitLab handle remote work?",
            "What is GitLab's anti-harassment policy?",
            "Tell me about GitLab's diversity and inclusion initiatives",
            "How does GitLab approach collaboration?",
            "What are the company's results-focused principles?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"💡", key=f"ex_{i}", help=question):
                st.session_state.user_input = question
                st.rerun()
        
        # Display each question text below buttons
        for question in example_questions:
            st.markdown(f'<div class="example-question">{question}</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize user input
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Welcome message
    if not st.session_state.chat_history:
        welcome_msg = f"""
        👋 **Welcome to GitPulseAI!** 
        
        I'm here to help you explore GitLab's handbook and policies. You can ask me about:
        - Company values and culture
        - Remote work practices  
        - Policies and procedures
        - Diversity & inclusion initiatives
        - And much more!
        
        **System Status:** {'🌩️ Cloud-powered' if USE_SUPABASE else '💻 Local mode'} with Google Gemini
        
        Try asking a question or click one of the example questions in the sidebar!
        """
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": welcome_msg
        })

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'''
            <div class="chat-message user-message">
                <strong>🧑‍💻</strong> {message["content"]}
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="chat-message assistant-message">
                <strong>🤖</strong> {message["content"]}
            </div>
            ''', unsafe_allow_html=True)

    # User input
    user_question = st.text_input(
        "Ask about GitLab's handbook:",
        value=st.session_state.user_input,
        placeholder="e.g., What are GitLab's core values?",
        key="question_input"
    )
    
    # Clear the session state input after displaying
    if st.session_state.user_input:
        st.session_state.user_input = ""

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")

    if ask_button and user_question.strip():
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_question
        })
        
        # Show thinking indicator
        with st.spinner("🤔 Thinking..."):
            try:
                # Initialize RAG system if needed
                if not rag_system.initialize():
                    st.error("❌ Failed to initialize RAG system")
                    return
                
                # Get response from RAG system
                response_data = rag_system.query(user_question, st.session_state.chat_history[:-1])
                
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

    # Footer
    st.markdown(f'''
    <div class="footer">
        <strong>GitPulseAI</strong> - GitLab Handbook Assistant<br>
        Powered by Google Gemini {'+ Supabase Vector Database' if USE_SUPABASE else '+ Local Embeddings'}<br>
        <em>Built for exploring GitLab's culture and practices</em>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 