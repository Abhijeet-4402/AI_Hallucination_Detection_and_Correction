<<<<<<< HEAD
"""
Vector Database Implementation using ChromaDB

This module handles the storage and retrieval of text embeddings
using ChromaDB for efficient semantic search.
"""

import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
import uuid
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Handles vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = "evidence_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector database
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Evidence documents for hallucination detection"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add documents to the vector database
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []
        
        # Generate unique IDs for documents
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{"source": "wikipedia", "index": i} for i in range(len(documents))]
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector database")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=doc_ids
            )
            logger.info("Documents added successfully")
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            logger.info(f"Searching for similar documents to: {query[:100]}...")
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'document': doc,
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'id': results['ids'][0][i] if results['ids'] else None
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document with metadata or None if not found
        """
        try:
            results = self.collection.get(ids=[doc_id])
            if results['documents'] and results['documents'][0]:
                return {
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {},
                    'id': doc_id
                }
            return None
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Evidence documents for hallucination detection"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def update_document(self, doc_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a document in the collection
        
        Args:
            doc_id: Document ID to update
            document: New document content
            metadata: Optional new metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {
                'ids': [doc_id],
                'documents': [document]
            }
            
            if metadata:
                update_data['metadatas'] = [metadata]
            
            self.collection.update(**update_data)
            logger.info(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
=======
"""
Vector Database Implementation using ChromaDB

This module handles the storage and retrieval of text embeddings
using ChromaDB for efficient semantic search.
"""

import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
import uuid
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Handles vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = "evidence_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector database
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Evidence documents for hallucination detection"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add documents to the vector database
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []
        
        # Generate unique IDs for documents
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{"source": "wikipedia", "index": i} for i in range(len(documents))]
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector database")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=doc_ids
            )
            logger.info("Documents added successfully")
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            logger.info(f"Searching for similar documents to: {query[:100]}...")
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'document': doc,
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'id': results['ids'][0][i] if results['ids'] else None
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document with metadata or None if not found
        """
        try:
            results = self.collection.get(ids=[doc_id])
            if results['documents'] and results['documents'][0]:
                return {
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {},
                    'id': doc_id
                }
            return None
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Evidence documents for hallucination detection"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def update_document(self, doc_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a document in the collection
        
        Args:
            doc_id: Document ID to update
            document: New document content
            metadata: Optional new metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {
                'ids': [doc_id],
                'documents': [document]
            }
            
            if metadata:
                update_data['metadatas'] = [metadata]
            
            self.collection.update(**update_data)
            logger.info(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
>>>>>>> fb3451155c14f135a09046a366815aba1850f393
