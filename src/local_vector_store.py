"""
Local Vector Store for GitLab Handbook Assistant

This module provides a local vector store using sentence-transformers for embeddings.
No API keys required - completely free and runs offline.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import ssl
import urllib.request

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed. Run: pip install sentence-transformers")
    SentenceTransformer = None

from sklearn.metrics.pairwise import cosine_similarity
from src.config import config

logger = logging.getLogger(__name__)

class LocalVectorStore:
    """
    Local vector store using sentence-transformers for embeddings.
    
    Completely free and runs offline without any API dependencies.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local vector store.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.embedding_model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Storage for documents and embeddings
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.index_to_doc_id: Dict[int, str] = {}
        self.doc_id_to_index: Dict[str, int] = {}
        
        # Configuration
        self.similarity_threshold = config.SIMILARITY_THRESHOLD
        self.max_results = config.MAX_SEARCH_RESULTS
        
        # Initialize the embedding model
        self._initialize_model()
        
    def _fix_ssl_context(self):
        """Fix SSL context for macOS and other systems with certificate issues."""
        try:
            # Create unverified SSL context for downloading models
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Install the context for HTTPS requests
            urllib.request.install_opener(
                urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=ssl_context)
                )
            )
            
            logger.info("SSL context configured for model downloading")
            return True
            
        except Exception as e:
            logger.warning(f"Could not configure SSL context: {e}")
            return False
        
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        if SentenceTransformer is None:
            logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            return
            
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            
            # First try to load normally
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info(f"Model loaded successfully: {self.model_name}")
                
            except Exception as ssl_error:
                if "SSL" in str(ssl_error) or "certificate" in str(ssl_error).lower():
                    logger.warning(f"SSL error encountered: {ssl_error}")
                    logger.info("Attempting to fix SSL configuration...")
                    
                    # Try to fix SSL and retry
                    if self._fix_ssl_context():
                        logger.info("Retrying model download with fixed SSL...")
                        self.embedding_model = SentenceTransformer(self.model_name)
                        logger.info(f"Model loaded successfully after SSL fix: {self.model_name}")
                    else:
                        raise ssl_error
                else:
                    raise ssl_error
            
            # Get actual embedding dimension
            if self.embedding_model:
                sample_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
                self.embedding_dimension = sample_embedding.shape[1]
                logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.error("This could be due to:")
            logger.error("1. Network connectivity issues")
            logger.error("2. SSL certificate problems (common on macOS)")
            logger.error("3. Firewall/proxy settings")
            logger.error("4. Missing dependencies")
            logger.error("")
            logger.error("Troubleshooting steps:")
            logger.error("1. Check your internet connection")
            logger.error("2. Try running: pip install --upgrade certifi")
            logger.error("3. On macOS, try: /Applications/Python\\ 3.x/Install\\ Certificates.command")
            logger.error("4. Or set environment variable: CURL_CA_BUNDLE=''")
            
            self.embedding_model = None
    
    def is_available(self) -> bool:
        """
        Check if the embedding model is available.
        
        Returns:
            bool: True if model is loaded and ready to use
        """
        return self.embedding_model is not None
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using sentence-transformers.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not initialized. Check logs for initialization errors.")
        
        try:
            # Clean text for embedding
            text = text.strip()
            if not text:
                raise ValueError("Empty text cannot be embedded")
            
            # Generate embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            
            # Convert to list for JSON serialization
            embedding_list = embedding.tolist()
            
            logger.debug(f"Generated local embedding with {len(embedding_list)} dimensions")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error generating local embedding: {e}")
            raise
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """
        Add a document to the local vector store.
        
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
            
            # Check if embedding model is available
            if not self.is_available():
                logger.error(f"Embedding model not available. Cannot add document {doc_id}.")
                return False
            
            # Generate embedding
            embedding = self._get_embedding(content)
            
            # Add to storage
            index = len(self.documents)
            self.documents.append(document)
            self.embeddings.append(embedding)
            self.index_to_doc_id[index] = doc_id
            self.doc_id_to_index[doc_id] = index
            
            logger.debug(f"Added document {doc_id} with local embedding")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document.get('id', 'unknown')}: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add multiple documents to the local vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            int: Number of documents successfully added
        """
        if not self.is_available():
            logger.error("Embedding model not available. Cannot add documents.")
            logger.error("Please check the model initialization logs above for troubleshooting steps.")
            return 0
        
        successful_adds = 0
        
        for i, doc in enumerate(documents):
            if self.add_document(doc):
                successful_adds += 1
            
            # Log progress for large batches
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(documents)} documents")
        
        logger.info(f"Successfully added {successful_adds}/{len(documents)} documents to local vector store")
        return successful_adds
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using local embeddings.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Similar documents with similarity scores
        """
        max_results = max_results or self.max_results
        
        if not self.documents:
            logger.warning("No documents in local vector store")
            return []
        
        if self.embedding_model is None:
            logger.warning("Embedding model not available, falling back to keyword search")
            return self._keyword_search(query, max_results)
        
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            query_vector = np.array(query_embedding).reshape(1, -1)
            
            # Calculate similarities with all documents
            document_vectors = np.array(self.embeddings)
            similarities = cosine_similarity(query_vector, document_vectors)[0]
            
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
            
            logger.debug(f"Local vector search found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during local vector search: {e}")
            return self._keyword_search(query, max_results)
    
    def _keyword_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search as fallback."""
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
                result['similarity_score'] = score / 20.0  # Normalize to 0-1 range
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
        Get statistics about the local vector store.
        
        Returns:
            Dict: Statistics about stored documents and embeddings
        """
        stats = {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dimension,
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "embedding_model": self.model_name,
            "model_loaded": self.embedding_model is not None
        }
        
        if self.embeddings and len(self.embeddings) > 1:
            # Calculate some basic statistics
            document_vectors = np.array(self.embeddings)
            similarities_matrix = cosine_similarity(document_vectors)
            # Get upper triangle (excluding diagonal)
            indices = np.triu_indices(len(self.embeddings), k=1)
            all_similarities = similarities_matrix[indices]
            
            stats.update({
                "average_document_similarity": float(np.mean(all_similarities)),
                "min_document_similarity": float(np.min(all_similarities)),
                "max_document_similarity": float(np.max(all_similarities))
            })
        
        return stats
    
    def clear(self):
        """Clear all documents and embeddings from the local vector store."""
        self.documents = []
        self.embeddings = []
        self.index_to_doc_id = {}
        self.doc_id_to_index = {}
        logger.info("Local vector store cleared")

def create_local_vector_store_from_documents(documents: List[Dict[str, Any]]) -> LocalVectorStore:
    """
    Convenience function to create and populate a local vector store.
    
    Args:
        documents: List of documents to add to the store
        
    Returns:
        LocalVectorStore: Populated local vector store
    """
    store = LocalVectorStore()
    store.add_documents(documents)
    return store

# Export main classes and functions
__all__ = [
    'LocalVectorStore',
    'create_local_vector_store_from_documents'
] 