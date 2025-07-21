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
    APP_NAME: str = os.getenv("APP_NAME", "GitLab Handbook Assistant")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # OpenAI Configuration (Legacy)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    
    # Google Gemini Configuration (Free Tier)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # RAG System Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    
    # Vector Search Configuration
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/chatbot.log")
    
    # Streamlit Configuration
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    # Data Configuration
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    SAMPLE_DATA_FILE: str = os.getenv("SAMPLE_DATA_FILE", "data/sample_gitlab_data.json")
    
    # UI Configuration
    GITLAB_ORANGE: str = "#FC6D26"
    GITLAB_PURPLE: str = "#6B46C1"
    
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
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """
        Get OpenAI configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: OpenAI configuration
        """
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE,
            "max_tokens": cls.OPENAI_MAX_TOKENS
        }
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """
        Get embedding configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Embedding configuration
        """
        return {
            "model": cls.EMBEDDING_MODEL,
            "dimension": cls.EMBEDDING_DIMENSION
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """
        Get search configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Search configuration
        """
        return {
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "max_results": cls.MAX_SEARCH_RESULTS
        }

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