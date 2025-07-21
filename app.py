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
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }
    
    /* Header styling */
    .app-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2d3748;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #718096;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        border-radius: 25px;
        font-size: 0.875rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(72, 187, 120, 0.3);
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Welcome message */
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .welcome-text {
        font-size: 1rem;
        opacity: 0.95;
        line-height: 1.6;
    }
    
    /* Chat messages */
    .message-container {
        display: flex;
        margin-bottom: 1.5rem;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .user-message-container {
        justify-content: flex-end;
        flex-direction: row-reverse;
    }
    
    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #4299e1, #3182ce);
        color: white;
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        position: relative;
        line-height: 1.5;
        font-size: 0.95rem;
        word-wrap: break-word;
    }
    
    .user-message {
        background: linear-gradient(135deg, #4299e1, #3182ce);
        color: white !important;
        border-bottom-right-radius: 6px;
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
    }
    
    .assistant-message {
        background: #f7fafc;
        color: #2d3748 !important;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 6px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Input area */
    .input-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTextInput input {
        border-radius: 25px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.75rem 1.25rem !important;
        font-size: 0.95rem !important;
        background: white !important;
        color: #2d3748 !important;
        transition: all 0.2s ease;
    }
    
    .stTextInput input:focus {
        border-color: #4299e1 !important;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.15) !important;
        outline: none !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #4299e1, #3182ce) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(66, 153, 225, 0.4) !important;
    }
    
         /* Sidebar styling */
     .css-1d391kg {
         background: rgba(255, 255, 255, 0.95);
         backdrop-filter: blur(10px);
         border-right: 1px solid rgba(255, 255, 255, 0.2);
     }
     
     .sidebar-content {
         padding: 1rem;
     }
     
     .example-questions-title {
         color: #2d3748;
         font-size: 1.1rem;
         font-weight: 700;
         margin-bottom: 1rem;
         padding-bottom: 0.5rem;
         border-bottom: 2px solid #e2e8f0;
     }
     
     /* Sidebar buttons */
     .css-1d391kg .stButton button {
         width: 100% !important;
         background: white !important;
         color: #4a5568 !important;
         border: 1px solid #e2e8f0 !important;
         border-radius: 12px !important;
         padding: 0.75rem 1rem !important;
         font-weight: 500 !important;
         font-size: 0.85rem !important;
         text-align: left !important;
         margin-bottom: 0.5rem !important;
         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
         transition: all 0.2s ease !important;
     }
     
     .css-1d391kg .stButton button:hover {
         background: #f7fafc !important;
         border-color: #cbd5e0 !important;
         transform: translateY(-1px) !important;
         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
     }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #a0aec0;
    }
    
    /* Loading spinner */
    .stSpinner {
        color: #4299e1 !important;
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
    status_text = "🌩️ Cloud Mode: Supabase + Google Gemini" if USE_SUPABASE else "💻 Local Mode: Hybrid embeddings + Google Gemini"
    st.markdown(f'''
    <div class="app-header">
        <div class="app-title">GitPulseAI</div>
        <div class="app-subtitle">GitLab Handbook Assistant</div>
        <div class="status-badge">
            {status_text}
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    if not rag_system:
        st.error("Failed to initialize the RAG system. Please check your configuration.")
        st.stop()

    # Sidebar with example questions
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<div class="example-questions-title">💡 Example Questions</div>', unsafe_allow_html=True)
        
        example_questions = [
            "What are GitLab's core values?",
            "How does GitLab handle remote work?",
            "What is GitLab's anti-harassment policy?",
            "Tell me about GitLab's diversity and inclusion initiatives",
            "How does GitLab approach collaboration?",
            "What are the company's results-focused principles?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"ex_{i}", help="Click to ask this question", use_container_width=True):
                st.session_state.selected_question = question
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize selected question
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Welcome message (only show when no chat history)
    if not st.session_state.chat_history:
        st.markdown('''
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to GitPulseAI!</div>
            <div class="welcome-text">
                I'm here to help you explore GitLab's handbook and policies.<br><br>
                <strong>Ask me about:</strong> Company values • Remote work • Policies • Culture • Best practices<br><br>
                <em>Powered by Google Gemini and Supabase</em>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Display chat history with new message structure
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'''
            <div class="message-container user-message-container">
                <div class="message-avatar user-avatar">👤</div>
                <div class="message-bubble user-message">
                    {message["content"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="message-container">
                <div class="message-avatar assistant-avatar">🤖</div>
                <div class="message-bubble assistant-message">
                    {message["content"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
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
            key="question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

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