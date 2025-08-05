"""
GitLab Handbook Assistant - Streamlit App

This is a Streamlit-based chat interface for the GitLab Handbook Assistant.
It uses a hybrid RAG system with Hugging Face embeddings and Google Gemini LLM.
"""

import os
import streamlit as st
from datetime import datetime
import warnings
import logging
import sys
warnings.filterwarnings("ignore")

# Configure logging first so we can log any startup issues
log_level = "INFO"  # Default to INFO until we can read from secrets
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_file = "logs/app.log"  # Default log location

# Ensure log directory exists
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Setup initial logging
logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("GitPulseAI")
logger.info("Starting GitPulseAI application")

# Load directly from Streamlit secrets instead of .env
try:
    logger.info("Loading configuration from Streamlit secrets")
    # Copy secrets to environment variables for compatibility with existing code
    for key, value in st.secrets.items():
        if isinstance(value, dict):
            # Skip nested sections
            continue
        # Set as environment variable
        os.environ[key] = str(value)
        if key in ["GEMINI_API_KEY", "SUPABASE_KEY"]:
            # Don't log sensitive values
            logger.debug(f"Loaded secret: {key}=******")
        else:
            logger.debug(f"Loaded config: {key}={value}")
            
    # Update log level from secrets if available
    if "LOG_LEVEL" in st.secrets:
        log_level = st.secrets["LOG_LEVEL"].upper()
        logging.getLogger().setLevel(getattr(logging, log_level))
        logger.info(f"Updated log level to {log_level}")
        
    # Update log file location if specified in secrets
    if "LOG_FILE" in st.secrets:
        log_file = st.secrets["LOG_FILE"]
        logger.info(f"Log file location: {log_file}")
        
    logger.info(f"Successfully loaded {len(st.secrets)} configuration values")
except Exception as e:
    logger.exception("Error loading from Streamlit secrets")
    logger.warning("Continuing with default configuration")
    
    # Fallback to .env file if secrets fail
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded fallback configuration from .env file")
    except Exception as e:
        logger.exception("Error loading .env file")
        logger.warning("Using default configuration values")

# Add additional log file handler if log location changed
if "LOG_FILE" in st.secrets:
    try:
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logging.getLogger().removeHandler(handler)
                
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Updated log file handler to {log_file}")
    except Exception as e:
        logger.exception(f"Error updating log file handler to {log_file}")

# 1. App title
st.set_page_config(
    page_title="🔍💬 GitPulseAI Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detect deployment mode
try:
    # Get the raw value first for debugging
    raw_supabase_value = os.environ.get("USE_SUPABASE", "false")
    logger.info(f"Raw USE_SUPABASE value: '{raw_supabase_value}'")
    
    # Handle different formats of boolean values
    if raw_supabase_value.lower() in ["true", "yes", "1", "t", "y"]:
        USE_SUPABASE = True
    else:
        USE_SUPABASE = False
    
    # Log the decision
    deployment_mode = "Cloud" if USE_SUPABASE else "Local"
    logger.info(f"Deployment mode set to: {deployment_mode} mode")
    
    # Display the mode in the UI for verification
    st.sidebar.success(f"Debug: Mode = {deployment_mode}", icon="🔧")
except Exception as e:
    logger.exception("Error determining deployment mode")
    USE_SUPABASE = False
    logger.warning("Defaulting to Local mode due to error")

@st.cache_resource
def initialize_rag_system():
    """Initialize the appropriate RAG system based on environment."""
    try:
        logger.info("Initializing RAG system...")
        
        if USE_SUPABASE:
            # Use Supabase cloud RAG system
            logger.info("Using Supabase cloud RAG system")
            from src.supabase_rag_system import create_supabase_rag_system
            
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY") 
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            
            # Log configurations (masking sensitive values)
            logger.debug(f"SUPABASE_URL: {supabase_url[:10]}...{supabase_url[-5:] if supabase_url else None}")
            logger.debug(f"GEMINI_API_KEY: {'*' * 8}{gemini_api_key[-4:] if gemini_api_key else None}")
            logger.debug(f"SUPABASE_KEY: {'*' * 8}{supabase_key[-4:] if supabase_key else None}")
            
            if not supabase_url or not supabase_key or not gemini_api_key:
                error_msg = "❌ Missing required environment variables: "
                error_msg += "SUPABASE_URL, " if not supabase_url else ""
                error_msg += "SUPABASE_KEY, " if not supabase_key else ""
                error_msg += "GEMINI_API_KEY" if not gemini_api_key else ""
                logger.error(error_msg)
                st.error(error_msg)
                return None
            
            logger.info("Creating Supabase RAG system...")
            return create_supabase_rag_system(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                gemini_api_key=gemini_api_key
            )
        else:
            # Use local hybrid RAG system
            logger.info("Using local hybrid RAG system")
            data_file = os.getenv("SAMPLE_DATA_FILE")
            logger.info(f"Loading data from: {data_file}")
            
            from src.hybrid_rag_system import HybridRAGSystem
            return HybridRAGSystem()
            
    except Exception as e:
        error_msg = f"❌ Failed to initialize RAG system: {str(e)}"
        logger.exception(error_msg)
        st.error(error_msg)
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
try:
    logger.info("Initializing chat session state")
    if "messages" not in st.session_state.keys():
        logger.info("Creating initial message in session state")
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you explore GitLab's handbook today?"}]
    logger.info(f"Current session state has {len(st.session_state.messages)} messages")
except Exception as e:
    logger.exception("CRITICAL ERROR during session state initialization")
    st.error(f"Session initialization error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# Welcome banner (only show when no chat history except initial message)
if len(st.session_state.messages) == 1:
    st.info(
        "👋 **Welcome to GitPulseAI!** Ask me anything about GitLab's handbook, policies, values, and culture. "
        "Use the example questions in the sidebar to get started, or type your own question below.",
        icon="💬"
    )

# 2. Display chat messages
try:
    logger.info("Starting chat message display")
    
    # Verify messages exist and have proper format
    if not isinstance(st.session_state.messages, list):
        logger.error(f"Messages is not a list: {type(st.session_state.messages)}")
        st.error("Invalid message format detected. Resetting chat.")
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you explore GitLab's handbook today?"}]
    
    for i, message in enumerate(st.session_state.messages):
        logger.debug(f"Displaying message {i}: role={message.get('role', 'unknown')}, content_length={len(message.get('content', ''))}")
        
        # Verify message format
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            logger.warning(f"Skipping malformed message at index {i}: {message}")
            continue
            
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Add response feedback for assistant messages (except first welcome)
            if message["role"] == "assistant" and i > 0:
                try:
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
                except Exception as e:
                    logger.exception(f"Error displaying feedback buttons for message {i}")
except Exception as e:
    logger.exception("CRITICAL ERROR while displaying chat messages")
    st.error("Error displaying chat messages. Please try refreshing the page.")
    import traceback
    st.code(traceback.format_exc())

# 3. Create the LLM response generation function
def generate_gitpulse_response(prompt_input):
    """Generate response using GitPulseAI RAG system."""
    start_time = datetime.now()
    query_id = f"q-{start_time.strftime('%Y%m%d-%H%M%S')}"
    
    logger.info(f"[{query_id}] Processing query: {prompt_input[:50]}...")
    
    try:
        # Check if RAG system is available
        if not rag_system:
            logger.error(f"[{query_id}] RAG system not available")
            return {"response": "❌ RAG system not available", "sources": []}
        
        # Initialize RAG system if needed
        logger.debug(f"[{query_id}] Initializing RAG system")
        if not rag_system.initialize():
            logger.error(f"[{query_id}] Failed to initialize RAG system")
            return {"response": "❌ Failed to initialize RAG system", "sources": []}
        
        # Get response from RAG system
        logger.info(f"[{query_id}] Sending query to RAG system")
        conversation_context = st.session_state.messages[:-1]
        logger.debug(f"[{query_id}] Context size: {len(conversation_context)} messages")
        
        response_data = rag_system.query(prompt_input, conversation_context)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{query_id}] Response generated in {response_time:.2f}s")
        
        if response_data.get("status") == "success":
            source_count = len(response_data.get("sources", []))
            logger.info(f"[{query_id}] Success - {source_count} sources found")
            
            # Log source details at debug level
            if source_count > 0:
                for i, source in enumerate(response_data.get("sources", [])[:3]):
                    logger.debug(f"[{query_id}] Source {i+1}: {source.get('title', 'Unknown')} - Score: {source.get('similarity_score', 0):.3f}")
            
            return {
                "response": response_data.get("response", "I couldn't generate a response."),
                "sources": response_data.get("sources", [])
            }
        else:
            error_msg = response_data.get('message', 'Something went wrong.')
            logger.warning(f"[{query_id}] Response error: {error_msg}")
            return {
                "response": f"❌ {error_msg}",
                "sources": []
            }
            
    except Exception as e:
        logger.exception(f"[{query_id}] Exception while generating response: {str(e)}")
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
try:
    logger.info("Checking if new response is needed")
    
    # Safety check for empty messages array
    if not st.session_state.messages:
        logger.warning("Messages array is empty, resetting")
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you explore GitLab's handbook today?"}]
        st.rerun()
    
    # Check if last message is from user
    if st.session_state.messages[-1]["role"] != "assistant":
        user_message = st.session_state.messages[-1]["content"]
        logger.info(f"Generating response to: {user_message[:50]}...")
        
        with st.spinner("🤔 Thinking..."):
            try:
                response_data = generate_gitpulse_response(user_message)
                logger.info("Response generated successfully")
            except Exception as e:
                logger.exception("Error during response generation")
                response_data = {
                    "response": f"⚠️ I encountered an error while generating a response: {str(e)}",
                    "sources": []
                }
        
        # Store sources for sidebar display
        try:
            if response_data.get("sources"):
                logger.debug(f"Storing {len(response_data['sources'])} sources in session state")
                st.session_state.last_sources = response_data["sources"]
        except Exception as e:
            logger.exception("Error storing sources")
        
        # Add assistant response to session state
        try:
            message = {"role": "assistant", "content": response_data.get("response", "Error generating response")}
            logger.info(f"Adding assistant response ({len(message['content'])} chars)")
            st.session_state.messages.append(message)
            logger.info(f"Session now has {len(st.session_state.messages)} messages")
            st.rerun()
        except Exception as e:
            logger.exception("Error adding response to session state")
            st.error(f"Failed to process response: {str(e)}")
except Exception as e:
    logger.exception("CRITICAL ERROR in response generation flow")
    st.error("An unexpected error occurred. Please refresh the page and try again.")
    import traceback
    st.code(traceback.format_exc()) 