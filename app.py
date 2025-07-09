"""
GitLab Handbook Assistant - Main Application

A Streamlit-based chatbot that helps users access GitLab's handbook and direction pages
through conversational AI using RAG (Retrieval-Augmented Generation) architecture.

Author: GitPulseAI Team
Version: 1.0.0
"""

import streamlit as st
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import custom modules (will be created in subsequent tasks)
# from src.rag_system import RAGSystem
# from src.data_processor import DataProcessor
# from src.config import Config

def configure_page():
    """Configure the Streamlit page with proper settings and theme."""
    st.set_page_config(
        page_title="GitLab Handbook Assistant",
        page_icon="🦊",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get help': 'https://github.com/YourUsername/GitPulseAI',
            'Report a bug': 'https://github.com/YourUsername/GitPulseAI/issues',
            'About': """
            # GitLab Handbook Assistant
            
            An AI-powered chatbot that helps you navigate GitLab's handbook and direction pages.
            Built with Streamlit, OpenAI, and RAG architecture.
            
            Version: 1.0.0
            """
        }
    )

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {
            "high_contrast": False,
            "show_sources": True,
            "show_follow_ups": True
        }

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        'OPENAI_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please create a .env file with the required variables. See .env.example for reference.")
        st.stop()

def render_header():
    """Render the application header with GitLab branding."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #FC6D26; margin-bottom: 0;">🦊 GitLab Handbook Assistant</h1>
        <p style="color: #666; font-size: 1.1rem; margin-top: 0;">
            Your AI-powered guide to GitLab's culture, processes, and policies
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with user preferences and information."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # User preferences
        st.session_state.user_preferences["high_contrast"] = st.checkbox(
            "High Contrast Mode",
            value=st.session_state.user_preferences["high_contrast"],
            help="Improve visibility for accessibility"
        )
        
        st.session_state.user_preferences["show_sources"] = st.checkbox(
            "Show Sources",
            value=st.session_state.user_preferences["show_sources"],
            help="Display source citations for responses"
        )
        
        st.session_state.user_preferences["show_follow_ups"] = st.checkbox(
            "Show Follow-up Questions",
            value=st.session_state.user_preferences["show_follow_ups"],
            help="Display suggested follow-up questions"
        )
        
        st.divider()
        
        # Information section
        st.header("ℹ️ Information")
        st.markdown("""
        **How to use:**
        - Type your question in the chat input
        - Press Enter or click Send
        - View sources by expanding the references
        - Use suggested follow-up questions for more info
        
        **Example questions:**
        - What are GitLab's core values?
        - How does GitLab handle remote work?
        - What is the GitLab review process?
        """)
        
        st.divider()
        
        # Statistics (placeholder)
        st.header("📊 Session Stats")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Conversation ID", st.session_state.conversation_id[-8:])

def validate_user_input(user_input: str) -> bool:
    """
    Validate and sanitize user input.
    
    Args:
        user_input (str): The user's input message
        
    Returns:
        bool: True if input is valid, False otherwise
    """
    # Check for empty input
    if not user_input.strip():
        st.warning("⚠️ Please enter a question or message.")
        return False
    
    # Check message length
    if len(user_input) > 1000:
        st.warning("⚠️ Message too long. Please keep it under 1000 characters.")
        return False
    
    # Basic content filtering
    prohibited_patterns = ["<script>", "javascript:", "eval(", "onload="]
    for pattern in prohibited_patterns:
        if pattern.lower() in user_input.lower():
            st.error("❌ Invalid input detected. Please rephrase your question.")
            return False
    
    return True

def process_user_message(user_message: str) -> Dict:
    """
    Process user message through the RAG system.
    
    Args:
        user_message (str): The user's message
        
    Returns:
        Dict: Response from the RAG system
    """
    # Placeholder implementation - will be replaced with actual RAG system
    logger.info(f"Processing message: {user_message[:50]}...")
    
    # Simulate processing time
    import time
    time.sleep(1)
    
    # Mock response for now
    return {
        "response": f"Thank you for your question about: '{user_message}'. This is a placeholder response. The actual RAG system will be implemented in the next phase.",
        "sources": [
            {
                "title": "GitLab Handbook - Culture",
                "url": "https://about.gitlab.com/handbook/company/culture/",
                "relevance_score": 0.85
            },
            {
                "title": "GitLab Values",
                "url": "https://about.gitlab.com/handbook/values/",
                "relevance_score": 0.78
            }
        ],
        "suggested_questions": [
            "What are GitLab's core values?",
            "How does GitLab approach remote work?",
            "What is GitLab's hiring process?"
        ]
    }

def render_chat_message(message: Dict, is_user: bool = False):
    """
    Render a chat message with proper formatting.
    
    Args:
        message (Dict): Message data
        is_user (bool): Whether this is a user message
    """
    role = "user" if is_user else "assistant"
    
    with st.chat_message(role):
        st.markdown(message["content"])
        
        # Show sources if available and enabled
        if not is_user and message.get("sources") and st.session_state.user_preferences["show_sources"]:
            with st.expander("📚 Sources & References"):
                for source in message["sources"]:
                    st.markdown(f"- [{source['title']}]({source['url']})")
                    if source.get("relevance_score"):
                        st.caption(f"Relevance: {source['relevance_score']:.2f}")
        
        # Show suggested follow-up questions
        if not is_user and message.get("suggested_questions") and st.session_state.user_preferences["show_follow_ups"]:
            st.markdown("**💭 You might also ask:**")
            cols = st.columns(min(len(message["suggested_questions"]), 3))
            for i, question in enumerate(message["suggested_questions"]):
                with cols[i % 3]:
                    if st.button(question, key=f"suggest_{len(st.session_state.messages)}_{i}"):
                        # Add the suggested question as a user message
                        st.session_state.messages.append({
                            "role": "user",
                            "content": question
                        })
                        # Process the suggested question
                        process_chat_input(question)
                        st.rerun()

def process_chat_input(user_input: str):
    """
    Process chat input and generate response.
    
    Args:
        user_input (str): User's input message
    """
    # Validate input
    if not validate_user_input(user_input):
        return
    
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Process through RAG system
    try:
        with st.spinner("🤔 Thinking..."):
            response = process_user_message(user_input)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["response"],
                "sources": response.get("sources", []),
                "suggested_questions": response.get("suggested_questions", [])
            })
            
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        st.error("❌ An error occurred while processing your message. Please try again.")

def render_welcome_message():
    """Render the welcome message for new users."""
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown("""
            👋 **Welcome to the GitLab Handbook Assistant!**
            
            I'm here to help you navigate GitLab's handbook and direction pages. I can assist you with:
            
            - 🏢 **Company culture and values**
            - 📋 **Policies and procedures**
            - 💼 **Career development guidance**
            - 🔧 **Technical processes and workflows**
            - 🌍 **Remote work practices**
            
            **How to get started:**
            1. Type your question in the chat input below
            2. Press Enter or click the send button
            3. Explore the sources and follow-up questions in my responses
            
            **Example questions to try:**
            - "What are GitLab's core values?"
            - "How does GitLab handle performance reviews?"
            - "What is GitLab's remote work policy?"
            
            What would you like to know about GitLab?
            """)

def render_chat_interface():
    """Render the main chat interface."""
    # Render welcome message
    render_welcome_message()
    
    # Render chat history
    for message in st.session_state.messages:
        render_chat_message(message, is_user=(message["role"] == "user"))
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about GitLab..."):
        process_chat_input(prompt)
        st.rerun()

def main():
    """Main application function."""
    # Configure page
    configure_page()
    
    # Validate environment
    validate_environment()
    
    # Initialize session state
    initialize_session_state()
    
    # Render UI components
    render_header()
    render_sidebar()
    
    # Add accessibility features
    st.markdown("""
    <style>
    /* Accessibility improvements */
    .stButton > button {
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* High contrast mode */
    .high-contrast {
        filter: contrast(150%) brightness(1.2);
    }
    
    /* Focus indicators */
    .stTextInput > div > div > input:focus {
        outline: 2px solid #FC6D26;
        outline-offset: 2px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Apply high contrast if enabled
    if st.session_state.user_preferences["high_contrast"]:
        st.markdown('<div class="high-contrast">', unsafe_allow_html=True)
    
    # Render main chat interface
    render_chat_interface()
    
    # Add keyboard shortcuts info
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; background: rgba(0,0,0,0.1); 
                padding: 5px 10px; border-radius: 5px; font-size: 12px;">
        💡 <strong>Tip:</strong> Use Tab to navigate, Enter to send messages
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 