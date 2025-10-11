"""
Wikipedia API Integration for Evidence Retrieval

This module handles the integration with Wikipedia API to fetch
real-time evidence documents based on user questions.
"""

import wikipedia
import re
import logging
import warnings
from bs4 import GuessedAtParserWarning
from typing import List, Optional

# Suppress the specific parser warning from the wikipedia library
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaRetriever:
    """Handles Wikipedia API integration for evidence retrieval"""
    
    def __init__(self, max_chars: int = 2000, max_results: int = 3):
        self.max_chars = max_chars
        self.max_results = max_results

    def retrieve_evidence_documents(self, question: str) -> List[str]:
        """
        Retrieve evidence documents from Wikipedia for a given question.
        This version is more robust to search failures and disambiguation.
        """
        logger.info(f"Starting evidence retrieval for question: {question}")
        
        documents = []
        seen_titles = set()

        # First, try a direct page lookup on the most likely subject of the question
        # This is often the most reliable method for "What is X" questions.
        try:
            # A simple regex to find capitalized words, which are often subjects.
            subjects = re.findall(r'\b[A-Z][a-z]*\b', question)
            main_subject = subjects[-1] if subjects else question # Default to full question
            
            logger.info(f"Attempting direct page lookup for subject: '{main_subject}'")
            page = wikipedia.page(main_subject, auto_suggest=True, redirect=True)
            
            if page.title not in seen_titles:
                content = page.content[:self.max_chars]
                documents.append(content)
                seen_titles.add(page.title)
                logger.info(f"Added evidence document from direct lookup: {page.title}")

        except Exception as e:
            logger.warning(f"Direct page lookup failed: {e}. Falling back to search.")

        # Fallback to searching if direct lookup fails or we need more documents
        if len(documents) < self.max_results:
            try:
                search_results = wikipedia.search(question, results=self.max_results)
                for title in search_results:
                    if len(documents) >= self.max_results:
                        break
                    if title not in seen_titles:
                        try:
                            page = wikipedia.page(title, auto_suggest=False, redirect=True)
                            content = page.content[:self.max_chars]
                            documents.append(content)
                            seen_titles.add(page.title)
                            logger.info(f"Added evidence document from search result: {page.title}")
                        except Exception as page_e:
                            logger.warning(f"Could not retrieve page '{title}': {page_e}")
            except Exception as search_e:
                logger.error(f"An unexpected error occurred during Wikipedia search: {search_e}")

        logger.info(f"Retrieved {len(documents)} unique evidence documents from Wikipedia.")
        return documents

    def _extract_keywords(self, question: str) -> List[str]:
        """
        Extract keywords from a question for better search results
        
        Args:
            question: User's question
            
        Returns:
            List of extracted keywords
        """
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 
            'during', 'before', 'after', 'above', 'below', 'between', 'among', 'how', 
            'when', 'where', 'why', 'who', 'which', 'that', 'this', 'these', 'those',
            'our', 'my', 'your', 'their', 'its', 'some', 'any', 'all', 'every', 'each'
        }
        
        # Clean and split the question
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Prioritize important terms (nouns, proper nouns, etc.)
        important_keywords = []
        other_keywords = []
        
        for keyword in keywords:
            # Check if it's a proper noun (capitalized in original question)
            original_words = re.findall(r'\b\w+\b', question)
            if keyword.title() in original_words or keyword.upper() in original_words:
                important_keywords.append(keyword)
            else:
                other_keywords.append(keyword)
        
        # Combine important keywords first, then others
        all_keywords = important_keywords + other_keywords
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in all_keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:5]  # Limit to top 5 keywords
    
    def _generate_search_queries(self, question: str, keywords: List[str]) -> List[str]:
        """
        Generate a list of search queries from a question and keywords.
        
        Args:
            question: User's question
            keywords: List of extracted keywords
            
        Returns:
            List of search queries
        """
        search_queries = []
        
        # Add the full question first
        search_queries.append(question)
        
        # Add specific combinations for better targeting
        if len(keywords) >= 2:
            # Combine top keywords for more specific searches
            search_queries.append(f"{keywords[0]} {keywords[1]}")
            if len(keywords) >= 3:
                search_queries.append(f"{keywords[0]} {keywords[1]} {keywords[2]}")
        
        # Add individual important keywords
        search_queries.extend(keywords[:3])  # Only top 3 individual keywords
        
        # Remove duplicates while preserving order
        seen_queries = set()
        unique_queries = []
        for query in search_queries:
            if query.lower() not in seen_queries:
                seen_queries.add(query.lower())
                unique_queries.append(query)
        
        return unique_queries

    def get_page_summary(self, page_title: str) -> Optional[str]:
        """
        Get a summary of a Wikipedia page
        
        Args:
            page_title: Title of the Wikipedia page
            
        Returns:
            Page summary or None if error
        """
        try:
            summary = wikipedia.summary(page_title, sentences=3)
            return summary
        except Exception as e:
            logger.error(f"Error getting summary for {page_title}: {e}")
            return None
