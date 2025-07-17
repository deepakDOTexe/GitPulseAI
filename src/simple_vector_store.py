"""
Simple Vector Store for GitLab Handbook Assistant

This module provides a simple vector store using basic word embeddings that work
completely offline without requiring model downloads.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import config

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """
    Simple vector store using TF-IDF and basic word embeddings.
    
    Completely offline and doesn't require any model downloads.
    """
    
    def __init__(self):
        """Initialize the simple vector store."""
        self.documents: List[Dict[str, Any]] = []
        self.doc_contents: List[str] = []
        self.index_to_doc_id: Dict[int, str] = {}
        self.doc_id_to_index: Dict[str, int] = {}
        
        # TF-IDF vectorizer for basic text similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        
        # Configuration
        self.similarity_threshold = max(0.1, config.SIMILARITY_THRESHOLD * 0.3)  # More lenient for TF-IDF
        self.max_results = config.MAX_SEARCH_RESULTS
        self.embedding_dimension = 1000  # TF-IDF feature dimension
        
        logger.info("Initialized SimpleVectorStore (offline TF-IDF based)")
    
    def is_available(self) -> bool:
        """
        Check if the vector store is available.
        
        Returns:
            bool: Always True since this is a simple offline implementation
        """
        return True
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching.
        
        Args:
            text: Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]', ' ', text)
        
        # Strip and return
        return text.strip()
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """
        Add a document to the simple vector store.
        
        Args:
            document: Document with 'id', 'content', and metadata
            
        Returns:
            bool: True if document added successfully
        """
        try:
            doc_id = document.get('id')
            content = document.get('content', '')
            
            if not doc_id:
                logger.error("Document missing required 'id' field")
                return False
            
            if not content.strip():
                logger.warning(f"Document {doc_id} has empty content, skipping")
                return False
            
            # Check if document already exists
            if doc_id in self.doc_id_to_index:
                logger.warning(f"Document {doc_id} already exists, skipping")
                return False
            
            # Preprocess content
            processed_content = self._preprocess_text(content)
            
            # Add to storage
            index = len(self.documents)
            self.documents.append(document)
            self.doc_contents.append(processed_content)
            self.index_to_doc_id[index] = doc_id
            self.doc_id_to_index[doc_id] = index
            
            logger.debug(f"Added document {doc_id} to simple vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document.get('id', 'unknown')}: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add multiple documents to the simple vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            int: Number of documents successfully added
        """
        successful_adds = 0
        
        for i, doc in enumerate(documents):
            if self.add_document(doc):
                successful_adds += 1
            
            # Log progress for large batches
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(documents)} documents")
        
        # Build TF-IDF matrix after adding all documents
        if successful_adds > 0:
            self._build_tfidf_matrix()
        
        logger.info(f"Successfully added {successful_adds}/{len(documents)} documents to simple vector store")
        return successful_adds
    
    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix from document contents."""
        if not self.doc_contents:
            logger.warning("No documents to build TF-IDF matrix")
            return
        
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_contents)
            logger.info(f"Built TF-IDF matrix with shape: {self.tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"Error building TF-IDF matrix: {e}")
            self.tfidf_matrix = None
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using TF-IDF similarity.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Similar documents with similarity scores
        """
        max_results = max_results or self.max_results
        
        if not self.documents:
            logger.warning("No documents in simple vector store")
            return []
        
        try:
            # Preprocess query
            processed_query = self._preprocess_text(query)
            
            # Use TF-IDF similarity if available
            if self.tfidf_matrix is not None:
                return self._tfidf_search(processed_query, max_results)
            else:
                # Fallback to keyword search
                return self._keyword_search(processed_query, max_results)
                
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return self._keyword_search(query, max_results)
    
    def _tfidf_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Perform TF-IDF based search.
        
        Args:
            query: Preprocessed query
            max_results: Maximum number of results
            
        Returns:
            List[Dict]: Search results
        """
        try:
            # Transform query using the fitted vectorizer
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Create results with similarity scores
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= self.similarity_threshold:
                    document = self.documents[i].copy()
                    document['similarity_score'] = float(similarity)
                    results.append(document)
            
            # Sort by similarity score (descending) and limit results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            results = results[:max_results]
            
            logger.debug(f"TF-IDF search found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return self._keyword_search(query, max_results)
    
    def _keyword_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search as fallback.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List[Dict]: Search results
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for i, document in enumerate(self.documents):
            content_lower = document.get('content', '').lower()
            title_lower = document.get('title', '').lower()
            keywords_lower = [kw.lower() for kw in document.get('keywords', [])]
            
            # Calculate relevance score
            score = 0
            
            # Exact phrase match (highest weight)
            if query_lower in content_lower:
                score += 10
            
            # Word overlap in content
            content_words = set(content_lower.split())
            word_overlap = len(query_words.intersection(content_words))
            score += word_overlap * 2
            
            # Title relevance
            if any(word in title_lower for word in query_words):
                score += 5
            
            # Keyword matches
            keyword_matches = sum(1 for word in query_words if word in keywords_lower)
            score += keyword_matches * 3
            
            if score > 0:
                result = document.copy()
                result['similarity_score'] = min(score / 20.0, 1.0)  # Normalize to 0-1 range
                results.append(result)
        
        # Sort by relevance score and return top results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        results = results[:max_results]
        
        logger.debug(f"Keyword search found {len(results)} relevant documents")
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Optional[Dict]: Document if found, None otherwise
        """
        index = self.doc_id_to_index.get(doc_id)
        if index is not None:
            return self.documents[index]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the simple vector store.
        
        Returns:
            Dict: Statistics about stored documents
        """
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dimension,
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "embedding_model": "TF-IDF (offline)",
            "model_loaded": True,
            "tfidf_features": self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0
        }
    
    def clear(self):
        """Clear all documents from the simple vector store."""
        self.documents = []
        self.doc_contents = []
        self.index_to_doc_id = {}
        self.doc_id_to_index = {}
        self.tfidf_matrix = None
        logger.info("Simple vector store cleared")

def create_simple_vector_store_from_documents(documents: List[Dict[str, Any]]) -> SimpleVectorStore:
    """
    Convenience function to create and populate a simple vector store.
    
    Args:
        documents: List of documents to add to the store
        
    Returns:
        SimpleVectorStore: Populated simple vector store
    """
    store = SimpleVectorStore()
    store.add_documents(documents)
    return store

# Export main classes and functions
__all__ = [
    'SimpleVectorStore',
    'create_simple_vector_store_from_documents'
] 