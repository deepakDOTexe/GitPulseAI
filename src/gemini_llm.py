"""
Google Gemini LLM Interface for GitLab Handbook Assistant

This module provides a Google Gemini LLM interface with generous free tier.
15 requests per minute, 1 million tokens per day - much better than OpenAI!
"""

import os
import logging
from typing import Dict, List, Any, Optional
import time
from src.config import config

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("google-generativeai not installed. Run: pip install google-generativeai")
    genai = None

logger = logging.getLogger(__name__)

class GeminiLLM:
    """
    Google Gemini LLM interface with generous free tier.
    
    Free tier includes:
    - 15 requests per minute
    - 1 million tokens per day
    - Much more generous than OpenAI
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
        
        # Rate limiting for free tier
        self.last_request_time = 0
        self.min_request_interval = 4.1  # ~15 requests per minute
        
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
        """Initialize the Gemini model."""
        if genai is None:
            logger.error("google-generativeai not available. Install with: pip install google-generativeai")
            return
        
        if not self.api_key:
            logger.error("Google API key not provided")
            return
        
        try:
            # Configure the API with REST transport to avoid gRPC SSL issues
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
        """
        Check if Gemini is available and properly configured.
        
        Returns:
            bool: True if Gemini is available
        """
        return self.model is not None
    
    def _rate_limit(self):
        """Implement rate limiting for free tier."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate_response(self, 
                         query: str, 
                         context: str, 
                         conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate response using Google Gemini.
        
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
            # Rate limiting for free tier
            self._rate_limit()
            
            # Build the prompt
            prompt = self._build_prompt(query, context, conversation_history)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if response.text:
                generated_text = response.text.strip()
                logger.debug(f"Generated response with Gemini ({len(generated_text)} chars)")
                return generated_text
            else:
                logger.warning("Gemini returned empty response")
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _build_prompt(self, 
                      query: str, 
                      context: str, 
                      conversation_history: List[Dict[str, str]] = None) -> str:
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
                if turn.get("user"):
                    prompt_parts.append(f"User: {turn['user']}")
                if turn.get("assistant"):
                    prompt_parts.append(f"Assistant: {turn['assistant']}")
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
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Gemini models.
        
        Returns:
            List[str]: List of available model names
        """
        if genai is None:
            return []
        
        try:
            models = genai.list_models()
            return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict: Model information
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available": self.is_available(),
            "rate_limit": f"{60/self.min_request_interval:.1f} requests/minute",
            "free_tier": "1 million tokens/day"
        }

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