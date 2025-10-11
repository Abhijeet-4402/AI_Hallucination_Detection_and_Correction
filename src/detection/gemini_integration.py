"""
Fallback Gemini integration that handles API issues gracefully, including rate limits.
"""

import os
import logging
import time
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiLLM:
    """Fallback Gemini LLM that handles API issues gracefully"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the Gemini LLM, finds a working model, and sets up a fallback.
        """
        logger.info("Initializing Gemini LLM...")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        self.api_working = False
        self._initialize_llm()

    def _initialize_llm(self):
        """
        Configures the API and dynamically finds a supported generative model.
        """
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found. Cannot initialize Gemini LLM.")
            return

        try:
            genai.configure(api_key=self.api_key)
            
            supported_model_name = None
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    supported_model_name = m.name
                    break
            
            if supported_model_name:
                logger.info(f"Found supported model: {supported_model_name}")
                self.model = genai.GenerativeModel(supported_model_name)
                self.api_working = True
                logger.info("Gemini LLM initialized successfully.")
            else:
                logger.warning("No model supporting 'generateContent' found for your API key.")
                self.use_fallback("No supported model found.")

        except Exception as e:
            logger.warning(f"Gemini API not available: {e}")
            self.use_fallback(str(e))

    def use_fallback(self, reason: str):
        """Switches to fallback mode and logs the reason."""
        self.api_working = False
        logger.info(f"Using fallback mode - will return placeholder responses. Reason: {reason}")

    def generate_answer(self, question: str, max_retries: int = 3) -> str:
        """
        Generates an answer using the Gemini model, with retries for rate limit errors.
        """
        if not self.api_working or not self.model:
            return "Error: Unable to access Gemini API. Please check configuration and API key."

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(question)
                return response.text.strip().replace('\n', ' ')
            except exceptions.ResourceExhausted as e:
                logger.warning(f"Rate limit exceeded (Attempt {attempt + 1}/{max_retries}). Waiting to retry...")
                last_exception = e
                # Use the suggested retry delay from the API error if available, otherwise use exponential backoff
                retry_delay = e.retry_delay if hasattr(e, 'retry_delay') and e.retry_delay else (2 ** attempt) * 5
                time.sleep(retry_delay + 1) # Add a small buffer to be safe
            except Exception as e:
                logger.error(f"An unexpected error occurred during Gemini content generation: {e}")
                return f"Error: Failed to generate answer from API. {e}"
        
        logger.error(f"Failed to get answer after {max_retries} retries due to persistent rate limiting.")
        return f"Error: Failed to generate answer from API after multiple retries. Last error: {last_exception}"

# Global Gemini instance
_gemini_llm = None

def get_gemini_llm() -> GeminiLLM:
    """Singleton pattern to get the global Gemini LLM instance."""
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = GeminiLLM()
    return _gemini_llm

def generate_answer(question: str) -> str:
    """Convenience function to generate an answer using the global instance."""
    llm = get_gemini_llm()
    return llm.generate_answer(question)
