"""
Streamlit Cloud deployment for GitPulseAI

This is a minimal wrapper around app.py that properly reads from .streamlit/secrets.toml
"""

import os
import streamlit as st
import logging
import sys
from dotenv import load_dotenv
from datetime import datetime

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("GitPulseAI-Launcher")
logger.info(f"Starting GitPulseAI launcher at {datetime.now().isoformat()}")

# First try to load from .env file
logger.info("Loading environment variables from .env file")
load_dotenv()

# Then, if on Streamlit Cloud, load from st.secrets
# This will override any .env settings with the same name
if st.runtime.exists():
    logger.info("Running in Streamlit Cloud - loading secrets")
    secrets_count = 0
    
    for key, value in st.secrets.items():
        if isinstance(value, dict):
            # Skip nested sections (not environment variables)
            logger.debug(f"Skipping nested section: {key}")
            continue
            
        # Set the environment variable
        os.environ[key] = str(value)
        
        # Log the key names but not values for sensitive data
        if key in ["GEMINI_API_KEY", "SUPABASE_KEY"]:
            masked_value = f"{'*' * 8}{str(value)[-4:] if value else 'None'}"
            logger.debug(f"Loaded secret: {key}={masked_value}")
        else:
            # For non-sensitive data, we can log the actual value
            logger.debug(f"Loaded environment variable: {key}={value}")
            
        secrets_count += 1
        
    logger.info(f"🌩️ Loaded {secrets_count} environment variables from Streamlit secrets")
    
    # Log critical configuration for debugging
    logger.info(f"USE_SUPABASE value: {os.environ.get('USE_SUPABASE', 'Not set')}")
    logger.info(f"SUPABASE_URL value: {os.environ.get('SUPABASE_URL', 'Not set')[:10]}...")
    logger.info(f"DATA_DIR value: {os.environ.get('DATA_DIR', 'Not set')}")
    logger.info(f"LOG_LEVEL value: {os.environ.get('LOG_LEVEL', 'Not set')}")
else:
    logger.info("Not running in Streamlit Cloud - using local environment variables")
    
# Now run the main app
logger.info("Importing main app module...")

try:
    # Prevent traceback suppression
    import sys
    sys.tracebacklimit = 1000
    
    # Emergency debugging
    print("==== EMERGENCY DEBUG INFO ====")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Environment variables: USE_SUPABASE={os.environ.get('USE_SUPABASE')}")
    print("============================")
    
    # Import the main app with full error handling
    from app import *
    logger.info("GitPulseAI launcher complete")
except Exception as e:
    logger.exception("!!! CRITICAL STARTUP FAILURE !!!")
    print(f"FATAL ERROR: {str(e)}")
    import traceback
    traceback_text = traceback.format_exc()
    print(traceback_text)
    
    # Try to display error in Streamlit
    try:
        import streamlit as st
        st.error("Application failed to start")
        st.error(str(e))
        st.code(traceback_text)
        st.warning("Check the logs and console for detailed error information")
    except:
        print("Could not display error in Streamlit")