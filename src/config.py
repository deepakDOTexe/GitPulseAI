"""
Configuration module for GitLab Handbook Assistant

This module handles all configuration settings, environment variables,
and application constants.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for application settings."""
    
    # Application Information
    APP_NAME: str = os.getenv("APP_NAME", "GitPulseAI")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Google Gemini Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # LLM Configuration
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
    
    # Vector Search Configuration
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    
    # Data Configuration
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    SAMPLE_DATA_FILE: str = os.getenv("SAMPLE_DATA_FILE", "data/gitlab_complete_handbook.json")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Streamlit Configuration
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # No longer requiring OpenAI API key since we use Google Gemini
        required_fields = []
        
        # Check for Gemini API key if it's being used
        if not cls.GEMINI_API_KEY:
            print("⚠️  Google Gemini API key not configured")
            print("💡 Set GEMINI_API_KEY in your .env file for full functionality")
            print("🔗 Get your free key from: https://makersuite.google.com/app/apikey")
        
        missing_fields = []
        for field in required_fields:
            if not getattr(cls, field):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required configuration: {', '.join(missing_fields)}")
            return False
        
        return True

# Create a global config instance
config = Config()

# Validate configuration on import
if not config.validate():
    print("⚠️  Configuration validation failed. Please check your environment variables.")

# Export commonly used configurations
__all__ = [
    "Config",
    "config"
] 