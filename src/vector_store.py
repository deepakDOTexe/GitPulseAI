"""
Simple Vector Store for GitLab Handbook Assistant

This module provides a basic in-memory vector store for embeddings and similarity search.
It uses OpenAI embeddings and cosine similarity for document retrieval.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from src.config import config

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """
    Simple in-memory vector store for document embeddings.
    
    Uses OpenAI embeddings and cosine similarity for retrieval.
    Provides basic functionality for adding documents and searching.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            api_key: OpenAI API key. Uses config if None.
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        
        # Storage for documents and embeddings
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.index_to_doc_id: Dict[int, str] = {}
        self.doc_id_to_index: Dict[str, int] = {}
        
        # Configuration
        self.embedding_model = config.EMBEDDING_MODEL
        self.similarity_threshold = config.SIMILARITY_THRESHOLD
        self.max_results = config.MAX_SEARCH_RESULTS
        
        logger.info(f"Initialized SimpleVectorStore with model {self.embedding_model}")
    
    def _get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Generate embedding for text using OpenAI API with retry logic.
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            Exception: If embedding generation fails after all retries
        """
        # Clean text for embedding
        text = text.strip()
        if not text:
            raise ValueError("Empty text cannot be embedded")
        
        # Truncate text if too long (OpenAI has token limits)
        max_chars = 8000  # Conservative limit for embedding models
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.warning(f"Text truncated to {max_chars} characters for embedding")
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                    encoding_format="float"
                )
                
                embedding = response.data[0].embedding
                logger.debug(f"Generated embedding with {len(embedding)} dimensions (attempt {attempt + 1})")
                
                return embedding
                
            except Exception as e:
                last_error = e
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    import time
                    wait_time = (2 ** attempt) * 1  # 1s, 2s, 4s
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} embedding attempts failed")
        
        # If we get here, all retries failed
        raise Exception(f"Failed to generate embedding after {max_retries} attempts: {last_error}")
    
    def add_document(self, document: Dict[str, Any], skip_embedding: bool = False) -> bool:
        """
        Add a document to the vector store.
        
        Args:
            document: Document with 'id', 'content', and metadata
            skip_embedding: If True, store document without embedding (fallback mode)
            
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
            
            # Generate embedding (or use placeholder in fallback mode)
            if skip_embedding:
                # Use a zero vector as placeholder for fallback mode
                embedding = [0.0] * 1536  # Standard embedding dimension
                logger.debug(f"Added document {doc_id} without embedding (fallback mode)")
            else:
                embedding = self._get_embedding(content)
                logger.debug(f"Added document {doc_id} with embedding")
            
            # Add to storage
            index = len(self.documents)
            self.documents.append(document)
            self.embeddings.append(embedding)
            self.index_to_doc_id[index] = doc_id
            self.doc_id_to_index[doc_id] = index
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document.get('id', 'unknown')}: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]], fallback_mode: bool = False) -> int:
        """
        Add multiple documents to the vector store.
        
        Args:
            documents: List of documents to add
            fallback_mode: If True, try without embeddings if normal mode fails
            
        Returns:
            int: Number of documents successfully added
        """
        successful_adds = 0
        failed_docs = []
        
        # First try with embeddings
        for i, doc in enumerate(documents):
            if self.add_document(doc, skip_embedding=False):
                successful_adds += 1
            else:
                failed_docs.append(doc)
            
            # Log progress for large batches
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(documents)} documents")
        
        # If fallback mode is enabled and we have failures, try without embeddings
        if fallback_mode and failed_docs:
            logger.warning(f"Trying to add {len(failed_docs)} failed documents in fallback mode (no embeddings)")
            
            for doc in failed_docs:
                if self.add_document(doc, skip_embedding=True):
                    successful_adds += 1
        
        logger.info(f"Successfully added {successful_adds}/{len(documents)} documents to vector store")
        return successful_adds
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity or keyword fallback.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Similar documents with similarity scores
        """
        max_results = max_results or self.max_results
        
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        # Check if we have any real embeddings (not just zero vectors)
        has_real_embeddings = any(
            sum(embedding) != 0 for embedding in self.embeddings
        )
        
        if not has_real_embeddings:
            logger.info("No embeddings available, using keyword-based search")
            return self._keyword_search(query, max_results)
        
        try:
            # Try vector search first
            return self._vector_search(query, max_results)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}, falling back to keyword search")
            return self._keyword_search(query, max_results)
    
    def _vector_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform vector-based similarity search."""
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
        
        logger.debug(f"Vector search found {len(results)} similar documents")
        return results
    
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
        Get statistics about the vector store.
        
        Returns:
            Dict: Statistics about stored documents and embeddings
        """
        if not self.embeddings:
            return {
                "total_documents": 0,
                "embedding_dimension": 0,
                "average_similarity_threshold": self.similarity_threshold
            }
        
        embedding_dimension = len(self.embeddings[0]) if self.embeddings else 0
        
        # Calculate some basic statistics
        all_similarities = []
        if len(self.embeddings) > 1:
            document_vectors = np.array(self.embeddings)
            similarities_matrix = cosine_similarity(document_vectors)
            # Get upper triangle (excluding diagonal)
            indices = np.triu_indices(len(self.embeddings), k=1)
            all_similarities = similarities_matrix[indices].tolist()
        
        stats = {
            "total_documents": len(self.documents),
            "embedding_dimension": embedding_dimension,
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "embedding_model": self.embedding_model
        }
        
        if all_similarities:
            stats.update({
                "average_document_similarity": np.mean(all_similarities),
                "min_document_similarity": np.min(all_similarities),
                "max_document_similarity": np.max(all_similarities)
            })
        
        return stats
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save the vector store to a file (excluding embeddings for size).
        
        Args:
            filepath: Path to save the vector store
            
        Returns:
            bool: True if saved successfully
        """
        try:
            data = {
                "documents": self.documents,
                "config": {
                    "embedding_model": self.embedding_model,
                    "similarity_threshold": self.similarity_threshold,
                    "max_results": self.max_results
                },
                "stats": self.get_stats()
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Vector store saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load vector store from a file (will need to regenerate embeddings).
        
        Args:
            filepath: Path to load the vector store from
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = data.get('documents', [])
            config_data = data.get('config', {})
            
            # Update configuration
            self.embedding_model = config_data.get('embedding_model', self.embedding_model)
            self.similarity_threshold = config_data.get('similarity_threshold', self.similarity_threshold)
            self.max_results = config_data.get('max_results', self.max_results)
            
            # Clear existing data
            self.documents = []
            self.embeddings = []
            self.index_to_doc_id = {}
            self.doc_id_to_index = {}
            
            # Re-add documents (this will regenerate embeddings)
            logger.info("Regenerating embeddings for loaded documents...")
            successful_adds = self.add_documents(documents)
            
            logger.info(f"Vector store loaded from {filepath}, {successful_adds} documents processed")
            return successful_adds > 0
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def clear(self):
        """Clear all documents and embeddings from the vector store."""
        self.documents = []
        self.embeddings = []
        self.index_to_doc_id = {}
        self.doc_id_to_index = {}
        logger.info("Vector store cleared")

def create_vector_store_from_documents(documents: List[Dict[str, Any]]) -> SimpleVectorStore:
    """
    Convenience function to create and populate a vector store.
    
    Args:
        documents: List of documents to add to the store
        
    Returns:
        SimpleVectorStore: Populated vector store
    """
    store = SimpleVectorStore()
    store.add_documents(documents)
    return store

# Export main classes and functions
__all__ = [
    'SimpleVectorStore',
    'create_vector_store_from_documents'
] 