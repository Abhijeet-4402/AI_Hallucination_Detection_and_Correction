"""
Gemini Pro Integration for AI Hallucination Detection System

This module handles the integration with Google's Gemini Pro API
using LangChain for answer generation.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLM:
    """Wrapper class for Gemini Pro integration using LangChain"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Initialize Gemini LLM
        
        Args:
            api_key: Gemini API key (if None, will use environment variable)
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.llm = None
        self.chain = None
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LangChain Gemini LLM and chain"""
        try:
            logger.info("Initializing Gemini Pro LLM...")
            
            # Initialize the ChatGoogleGenerativeAI model
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.7,
                max_output_tokens=1024
            )
            
            # Create a prompt template for answer generation
            prompt_template = PromptTemplate(
                input_variables=["question"],
                template="""You are a helpful AI assistant. Please provide a clear, accurate, and informative answer to the following question.

Question: {question}

Answer:"""
            )
            
            # Create the LLM chain
            self.chain = LLMChain(llm=self.llm, prompt=prompt_template)
            
            logger.info("Gemini Pro LLM initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini LLM: {e}")
            raise
    
    def generate_answer(self, question: str) -> str:
        """
        Generate an answer for the given question
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer string
        """
        try:
            logger.info(f"Generating answer for question: {question[:100]}...")
            
            # Generate answer using the chain
            response = self.chain.run(question=question)
            
            logger.info("Answer generated successfully!")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def generate_answer_with_context(self, question: str, context: List[str]) -> str:
        """
        Generate an answer with additional context
        
        Args:
            question: The question to answer
            context: List of context documents
            
        Returns:
            Generated answer string
        """
        try:
            logger.info(f"Generating answer with context for question: {question[:100]}...")
            
            # Create context string
            context_str = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(context)])
            
            # Create a more detailed prompt with context
            prompt_template = PromptTemplate(
                input_variables=["question", "context"],
                template="""You are a helpful AI assistant. Please provide a clear, accurate, and informative answer to the following question, using the provided context when relevant.

Context:
{context}

Question: {question}

Answer:"""
            )
            
            # Create temporary chain with context
            temp_chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = temp_chain.run(question=question, context=context_str)
            
            logger.info("Answer with context generated successfully!")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer with context: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test the connection to Gemini API
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing Gemini API connection...")
            
            test_response = self.generate_answer("What is 2+2?")
            
            if test_response and len(test_response) > 0:
                logger.info("Gemini API connection test successful!")
                return True
            else:
                logger.error("Gemini API connection test failed - no response received")
                return False
                
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False

# Global Gemini instance
_gemini_llm = None

def get_gemini_llm() -> GeminiLLM:
    """Get or create the global Gemini LLM instance"""
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = GeminiLLM()
    return _gemini_llm

def generate_answer(question: str) -> str:
    """
    Convenience function to generate an answer
    
    Args:
        question: The question to answer
        
    Returns:
        Generated answer string
    """
    llm = get_gemini_llm()
    return llm.generate_answer(question)
