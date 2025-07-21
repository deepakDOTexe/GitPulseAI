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
st.set_page_config(page_title="🔍💬 GitPulseAI Chatbot")

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
    st.title('🔍💬 GitPulseAI Chatbot')
    
    # Status
    status_text = "🌩️ Cloud Mode" if USE_SUPABASE else "💻 Local Mode"
    st.success(f'{status_text} - System Ready!', icon='✅')
    
    st.subheader('Example Questions')
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
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    # Clear chat history button
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you explore GitLab's handbook today?"}]
    st.button('Clear Chat History', on_click=clear_chat_history)

# 2. Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you explore GitLab's handbook today?"}]

# 2. Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 3. Create the LLM response generation function
def generate_gitpulse_response(prompt_input):
    """Generate response using GitPulseAI RAG system."""
    try:
        # Check if RAG system is available
        if not rag_system:
            return "❌ RAG system not available"
        
        # Initialize RAG system if needed
        if not rag_system.initialize():
            return "❌ Failed to initialize RAG system"
        
        # Get response from RAG system
        response_data = rag_system.query(prompt_input, st.session_state.messages[:-1])
        
        if response_data.get("status") == "success":
            return response_data.get("response", "I couldn't generate a response.")
        else:
            return f"❌ {response_data.get('message', 'Something went wrong.')}"
            
    except Exception as e:
        return f"❌ Sorry, I encountered an error: {str(e)}"

# 4. Accept prompt input
if prompt := st.chat_input("Ask about GitLab's handbook, policies, values, and culture..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# 5. Generate a new LLM response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_gitpulse_response(st.session_state.messages[-1]["content"])
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message) 