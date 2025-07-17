"""
Local LLM Interface for GitLab Handbook Assistant

This module provides a local LLM interface using Ollama.
No API keys required - completely free and runs offline.
"""

import logging
import json
from typing import Dict, List, Any, Optional
import requests
from src.config import config

logger = logging.getLogger(__name__)

class LocalLLM:
    """
    Local LLM interface using Ollama.
    
    Completely free and runs offline without any API dependencies.
    """
    
    def __init__(self, 
                 model_name: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434"):
        """
        Initialize the local LLM.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
        
        # Configuration
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        
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
        
        logger.info(f"Initialized LocalLLM with model: {self.model_name}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and the model is loaded.
        
        Returns:
            bool: True if Ollama is available and model is ready
        """
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if our model is available
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            # Check for exact match or partial match
            model_available = any(
                self.model_name in name or name.startswith(self.model_name.split(':')[0])
                for name in model_names
            )
            
            if not model_available:
                logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                return False
            
            logger.info(f"Ollama is available with model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    def pull_model(self) -> bool:
        """
        Pull the model if it's not already available.
        
        Returns:
            bool: True if model was pulled successfully
        """
        try:
            logger.info(f"Pulling model: {self.model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minute timeout for model pulling
            )
            
            if response.status_code == 200:
                logger.info(f"Model {self.model_name} pulled successfully")
                return True
            else:
                logger.error(f"Failed to pull model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def generate_response(self, 
                         query: str, 
                         context: str, 
                         conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate response using the local LLM.
        
        Args:
            query: User query
            context: Retrieved document context
            conversation_history: Previous conversation turns
            
        Returns:
            str: Generated response
        """
        try:
            # Build the prompt
            prompt = self._build_prompt(query, context, conversation_history)
            
            # Make request to Ollama
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "stop": ["Human:", "User:"]
                    },
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                logger.debug(f"Generated response with local LLM ({len(generated_text)} chars)")
                return generated_text
            else:
                logger.error(f"LLM request failed: {response.text}")
                return "I apologize, but I encountered an error while generating a response."
                
        except Exception as e:
            logger.error(f"Error generating response with local LLM: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _build_prompt(self, 
                      query: str, 
                      context: str, 
                      conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved document context
            conversation_history: Previous conversation turns
            
        Returns:
            str: Complete prompt
        """
        prompt_parts = [self.system_prompt]
        
        # Add conversation history if provided
        if conversation_history:
            prompt_parts.append("\n--- Conversation History ---")
            for turn in conversation_history[-3:]:  # Keep last 3 turns
                if turn.get("user"):
                    prompt_parts.append(f"Human: {turn['user']}")
                if turn.get("assistant"):
                    prompt_parts.append(f"Assistant: {turn['assistant']}")
        
        # Add context and current query
        prompt_parts.extend([
            "\n--- GitLab Documentation Context ---",
            context,
            "\n--- Current Question ---",
            f"Human: {query}",
            "\nAssistant: "
        ])
        
        return "\n".join(prompt_parts)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            else:
                logger.error(f"Failed to get models: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict: Model information
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get model info: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

def create_local_llm(model_name: str = "llama3.1:8b") -> LocalLLM:
    """
    Create and initialize a local LLM instance.
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        LocalLLM: Initialized local LLM
    """
    llm = LocalLLM(model_name)
    
    # Check if model is available
    if not llm.is_available():
        logger.info(f"Model {model_name} not available, attempting to pull...")
        if not llm.pull_model():
            logger.error(f"Failed to pull model {model_name}")
    
    return llm

# Common model recommendations
RECOMMENDED_MODELS = {
    "fast": "llama3.1:8b",      # Good balance of speed and quality
    "quality": "llama3.1:70b",  # Higher quality but slower
    "small": "llama3.2:3b",     # Fastest, good for basic tasks
    "code": "codellama:7b",     # Good for code-related queries
}

def get_recommended_model(preference: str = "fast") -> str:
    """
    Get a recommended model based on user preference.
    
    Args:
        preference: "fast", "quality", "small", or "code"
        
    Returns:
        str: Recommended model name
    """
    return RECOMMENDED_MODELS.get(preference, RECOMMENDED_MODELS["fast"])

# Export main classes and functions
__all__ = [
    'LocalLLM',
    'create_local_llm',
    'get_recommended_model',
    'RECOMMENDED_MODELS'
] 