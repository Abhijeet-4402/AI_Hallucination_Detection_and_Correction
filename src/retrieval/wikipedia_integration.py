"""
Wikipedia API Integration for Evidence Retrieval

This module handles the integration with Wikipedia API to fetch
real-time evidence documents based on user questions.
"""

import wikipedia
import logging
from typing import List, Dict, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaRetriever:
    """Handles Wikipedia API integration for evidence retrieval"""
    
    def __init__(self, language: str = "en", max_results: int = 5):
        """
        Initialize Wikipedia retriever
        
        Args:
            language: Wikipedia language code (default: "en")
            max_results: Maximum number of results to return
        """
        self.language = language
        self.max_results = max_results
        wikipedia.set_lang(language)
        
        # Set rate limiting to avoid API issues
        wikipedia.set_rate_limiting(True)
        
    def extract_keywords(self, question: str) -> List[str]:
        """
        Extract keywords from the question for better search results
        
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
            'when', 'where', 'why', 'who', 'which', 'that', 'this', 'these', 'those'
        }
        
        # Clean and split the question
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:5]  # Limit to top 5 keywords
    
    def search_pages(self, query: str) -> List[str]:
        """
        Search for Wikipedia pages related to the query
        
        Args:
            query: Search query
            
        Returns:
            List of page titles
        """
        try:
            logger.info(f"Searching Wikipedia for: {query}")
            search_results = wikipedia.search(query, results=self.max_results)
            logger.info(f"Found {len(search_results)} search results")
            return search_results
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            return []
    
    def get_page_content(self, page_title: str) -> Optional[str]:
        """
        Get the content of a Wikipedia page
        
        Args:
            page_title: Title of the Wikipedia page
            
        Returns:
            Page content or None if error
        """
        try:
            logger.info(f"Fetching content for page: {page_title}")
            page = wikipedia.page(page_title)
            
            # Clean the content - remove references and extra whitespace
            content = page.content
            
            # Remove reference markers like [1], [2], etc.
            content = re.sub(r'\[\d+\]', '', content)
            
            # Remove extra whitespace and newlines
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Limit content length to avoid overwhelming the system
            if len(content) > 2000:
                content = content[:2000] + "..."
            
            logger.info(f"Retrieved {len(content)} characters of content")
            return content
            
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Disambiguation error for {page_title}: {e}")
            # Try to get the first option
            try:
                page = wikipedia.page(e.options[0])
                content = page.content
                content = re.sub(r'\[\d+\]', '', content)
                content = re.sub(r'\s+', ' ', content).strip()
                if len(content) > 2000:
                    content = content[:2000] + "..."
                return content
            except Exception as e2:
                logger.error(f"Error with disambiguation option: {e2}")
                return None
                
        except wikipedia.exceptions.PageError:
            logger.warning(f"Page not found: {page_title}")
            return None
        except Exception as e:
            logger.error(f"Error fetching page content: {e}")
            return None
    
    def retrieve_evidence_documents(self, question: str) -> List[str]:
        """
        Retrieve evidence documents from Wikipedia based on the question
        
        Args:
            question: User's question
            
        Returns:
            List of evidence documents (page contents)
        """
        logger.info(f"Starting evidence retrieval for question: {question}")
        
        # Extract keywords from the question
        keywords = self.extract_keywords(question)
        logger.info(f"Extracted keywords: {keywords}")
        
        evidence_docs = []
        
        # Search using the full question first
        search_queries = [question] + keywords
        
        for query in search_queries:
            if len(evidence_docs) >= self.max_results:
                break
                
            # Search for pages
            page_titles = self.search_pages(query)
            
            # Get content for each page
            for page_title in page_titles:
                if len(evidence_docs) >= self.max_results:
                    break
                    
                content = self.get_page_content(page_title)
                if content and content not in evidence_docs:  # Avoid duplicates
                    evidence_docs.append(content)
                    logger.info(f"Added evidence document from: {page_title}")
        
        logger.info(f"Retrieved {len(evidence_docs)} evidence documents")
        return evidence_docs
    
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
