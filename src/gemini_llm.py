"""
Google Gemini LLM Interface for GitLab Handbook Assistant

This module provides the LLM interface using Google's Gemini API with REST transport
for better reliability and performance.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st

# Import Google AI SDK with error handling
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

from src.config import config

logger = logging.getLogger(__name__)

class GeminiLLM:
    """
    Google Gemini LLM interface with enhanced error handling and monitoring.
    
    Features:
    - REST transport for better reliability
    - Retry logic for transient failures  
    - Rate limiting for free tier compliance
    - Comprehensive error handling
    - Performance monitoring
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini LLM.
        
        Args:
            api_key: Google API key. Uses config if None.
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name
        self.model = None
        
        # Configuration
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        
        # Enhanced rate limiting and monitoring
        self.last_request_time = 0
        self.min_request_interval = 4.1  # ~15 requests per minute
        self.request_count = 0
        self.error_count = 0
        self.total_tokens_used = 0
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # System prompt for GitLab context
        self.system_prompt = """You are a helpful assistant for GitLab's Handbook and Direction pages. 
You have access to GitLab's internal documentation about their company culture, values, processes, 
and policies. Your role is to:

1. Provide accurate, helpful information based on the GitLab documentation
2. Always cite your sources when providing information
3. Be conversational and friendly while maintaining professionalism
4. If you don't know something, admit it rather than guessing
5. Focus on GitLab-specific context and practices
6. Help users understand GitLab's "build in public" philosophy and culture

When responding:
- Use the provided context from GitLab documentation
- Include relevant source citations
- Be specific and actionable when possible
- Maintain GitLab's friendly, inclusive tone"""
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model with enhanced error handling."""
        if not GENAI_AVAILABLE:
            logger.error("google-generativeai not available. Install with: pip install google-generativeai")
            return
        
        if not self.api_key:
            logger.error("Google API key not provided")
            return
        
        try:
            # Configure the API with REST transport (more reliable than gRPC)
            if genai:
                genai.configure(api_key=self.api_key, transport='rest')
                
                # Create the model with safety settings
                generation_config = {
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": self.max_tokens,
                }
                
                safety_settings = [
                    {
                        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    },
                ]
                
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    system_instruction=self.system_prompt
                )
            
            logger.info(f"Initialized Gemini model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if the Gemini model is available."""
        return GENAI_AVAILABLE and self.model is not None and bool(self.api_key)
    
    @st.cache_data(ttl=300)  # Cache responses for 5 minutes
    def _cached_generate(_self, prompt: str, context_hash: int) -> str:
        """Cached generation to reduce API calls for similar queries."""
        return _self._generate_with_retry(prompt)
    
    def _generate_with_retry(self, prompt: str) -> str:
        """Generate response with retry logic and error handling."""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self._enforce_rate_limit()
                
                # Make the API call
                if self.model:
                    response = self.model.generate_content(prompt)
                else:
                    raise Exception("Model not initialized")
                
                # Update metrics
                self.request_count += 1
                if hasattr(response, 'usage_metadata'):
                    self.total_tokens_used += getattr(response.usage_metadata, 'total_token_count', 0)
                
                if response.text:
                    return response.text.strip()
                else:
                    logger.warning("Empty response from Gemini API")
                    return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                    
            except Exception as e:
                self.error_count += 1
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    return f"I'm experiencing technical difficulties. Please try again later. (Error: {str(e)[:100]})"
        
        return "I'm currently unavailable due to technical issues. Please try again later."
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting for free tier compliance."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and usage."""
        return {
            "model_name": self.model_name,
            "available": self.is_available(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "total_tokens_used": self.total_tokens_used,
            "error_rate": self.error_count / max(self.request_count, 1)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the LLM service."""
        info = self.get_model_info()
        
        # Determine health based on error rate
        error_rate = info["error_rate"]
        if error_rate == 0:
            status = "healthy"
        elif error_rate < 0.1:
            status = "warning"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "error_rate": f"{error_rate:.1%}",
            "requests": info["request_count"],
            "errors": info["error_count"],
            "tokens_used": info["total_tokens_used"]
        }

    def generate_response(self, 
                         query: str, 
                         context: str, 
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate response using Google Gemini (public interface).
        
        Args:
            query: User query
            context: Retrieved document context
            conversation_history: Previous conversation turns
            
        Returns:
            str: Generated response
        """
        if not self.is_available():
            return "I apologize, but the Gemini model is not available. Please check your configuration."
        
        try:
            # Build the prompt
            prompt = self._build_prompt(query, context, conversation_history or [])
            
            # Generate response with caching and retry logic
            context_hash = hash(f"{context[:500]}{str(conversation_history)}")
            response = self._cached_generate(prompt, context_hash)
            
            logger.debug(f"Generated response with Gemini ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _build_prompt(self, 
                      query: str, 
                      context: str, 
                      conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build the complete prompt for Gemini.
        
        Args:
            query: User query
            context: Retrieved document context
            conversation_history: Previous conversation turns
            
        Returns:
            str: Complete prompt
        """
        prompt_parts = []
        
        # Add conversation history if provided
        if conversation_history:
            prompt_parts.append("=== Conversation History ===")
            for turn in conversation_history[-3:]:  # Keep last 3 turns
                if isinstance(turn, dict):
                    if turn.get("user"):
                        prompt_parts.append(f"User: {turn['user']}")
                    elif turn.get("role") == "user":
                        prompt_parts.append(f"User: {turn.get('content', '')}")
                    elif turn.get("assistant"):
                        prompt_parts.append(f"Assistant: {turn['assistant']}")
                    elif turn.get("role") == "assistant":
                        prompt_parts.append(f"Assistant: {turn.get('content', '')}")
            prompt_parts.append("")
        
        # Add context and current query
        prompt_parts.extend([
            "=== GitLab Documentation Context ===",
            context,
            "",
            "=== Current Question ===",
            f"User: {query}",
            "",
            "Please provide a helpful response based on the GitLab documentation provided above. Include relevant source citations where appropriate."
        ])
        
        return "\n".join(prompt_parts)

def create_gemini_llm(api_key: Optional[str] = None, 
                      model_name: str = "gemini-1.5-flash") -> GeminiLLM:
    """
    Create and initialize a Gemini LLM instance.
    
    Args:
        api_key: Google API key
        model_name: Name of the Gemini model to use
        
    Returns:
        GeminiLLM: Initialized Gemini LLM
    """
    llm = GeminiLLM(api_key, model_name)
    
    if not llm.is_available():
        logger.error("Failed to initialize Gemini LLM")
    
    return llm

# Recommended Gemini models
GEMINI_MODELS = {
    "fast": "gemini-1.5-flash",      # Fast and efficient
    "quality": "gemini-1.5-pro",     # Higher quality
    "experimental": "gemini-exp-1114" # Latest experimental
}

def get_recommended_gemini_model(preference: str = "fast") -> str:
    """
    Get a recommended Gemini model based on user preference.
    
    Args:
        preference: "fast", "quality", or "experimental"
        
    Returns:
        str: Recommended model name
    """
    return GEMINI_MODELS.get(preference, GEMINI_MODELS["fast"])

# Export main classes and functions
__all__ = [
    'GeminiLLM',
    'create_gemini_llm',
    'get_recommended_gemini_model',
    'GEMINI_MODELS'
] 