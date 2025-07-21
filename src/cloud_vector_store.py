"""
Cloud-Optimized Vector Store for Streamlit Community Cloud

This module provides a vector store optimized for Streamlit Cloud deployment:
- Loads precomputed embeddings (no model downloads)
- Uses Streamlit caching for fast startup
- Minimal memory footprint
- Fast similarity search
"""

import streamlit as st
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import warnings

logger = logging.getLogger(__name__)

class CloudVectorStore:
    """
    Cloud-optimized vector store for Streamlit deployment.
    
    Features:
    - Loads precomputed embeddings (no model downloads)
    - Uses Streamlit caching
    - Fast similarity search
    - Fallback to TF-IDF when needed
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.3,
                 max_results: int = 5):
        """
        Initialize the cloud vector store.
        
        Args:
            similarity_threshold: Minimum similarity score for results
            max_results: Maximum number of results to return
        """
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # These will be loaded from cached data
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dimension: int = 0
        self.embedding_model: str = "unknown"
        
        logger.info("Initialized CloudVectorStore (precomputed embeddings)")
    
    @st.cache_data
    def _load_precomputed_data(_self, data_file: str) -> Dict[str, Any]:
        """
        Load precomputed embeddings data with Streamlit caching.
        
        This method is cached by Streamlit for fast reloads.
        """
        try:
            logger.info(f"Loading precomputed embeddings from: {data_file}")
            
            if not Path(data_file).exists():
                raise FileNotFoundError(f"Precomputed data file not found: {data_file}")
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if 'documents' not in data or 'embeddings' not in data:
                raise ValueError("Invalid precomputed data format - missing documents or embeddings")
            
            documents = data['documents']
            embeddings = data['embeddings']
            metadata = data.get('metadata', {})
            
            if len(documents) != len(embeddings):
                raise ValueError(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")
            
            logger.info(f"Loaded precomputed data:")
            logger.info(f"  - Documents: {len(documents)}")
            logger.info(f"  - Embeddings: {len(embeddings)}")
            logger.info(f"  - Model: {metadata.get('embedding_model', 'unknown')}")
            logger.info(f"  - Dimensions: {metadata.get('embedding_dimension', 0)}")
            
            return {
                'documents': documents,
                'embeddings': np.array(embeddings, dtype=np.float32),  # Use float32 to save memory
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading precomputed data: {e}")
            raise
    
    def load_data(self, data_file: str) -> bool:
        """
        Load precomputed data into the vector store.
        
        Args:
            data_file: Path to precomputed embeddings JSON file
            
        Returns:
            bool: True if data loaded successfully
        """
        try:
            # Use cached loading function
            cached_data = self._load_precomputed_data(data_file)
            
            # Set instance variables
            self.documents = cached_data['documents']
            self.embeddings = cached_data['embeddings']
            
            metadata = cached_data['metadata']
            self.embedding_dimension = metadata.get('embedding_dimension', 0)
            self.embedding_model = metadata.get('embedding_model', 'unknown')
            
            logger.info(f"Cloud vector store initialized with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using precomputed embeddings.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Similar documents with similarity scores
        """
        max_results = max_results or self.max_results
        
        if not self.documents or self.embeddings is None:
            logger.warning("No data loaded in cloud vector store")
            return []
        
        try:
            # For precomputed embeddings, we need to generate query embedding
            # Since we don't have the model in cloud deployment, we'll use a simpler approach
            return self._keyword_search(query, max_results)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self._keyword_search(query, max_results)
    
    def search_with_precomputed_query(self, query_embedding: List[float], max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search using a precomputed query embedding.
        
        This method is useful when you have the query embedding already computed.
        
        Args:
            query_embedding: Precomputed embedding for the query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Similar documents with similarity scores
        """
        max_results = max_results or self.max_results
        
        if not self.documents or self.embeddings is None:
            logger.warning("No data loaded in cloud vector store")
            return []
        
        try:
            # Convert query embedding to numpy array
            query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Calculate similarities with warning suppression
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                similarities = cosine_similarity(query_vector, self.embeddings)[0]
            
            # Handle numerical issues
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create results with similarity scores
            results = []
            for i, similarity in enumerate(similarities):
                if np.isfinite(similarity) and similarity >= self.similarity_threshold:
                    document = self.documents[i].copy()
                    document['similarity_score'] = float(similarity)
                    results.append(document)
            
            # Sort by similarity score (descending) and limit results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            results = results[:max_results]
            
            logger.debug(f"Vector search found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _keyword_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based search when vector search is not available.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List[Dict]: Documents matching keywords
        """
        if not self.documents:
            return []
        
        query_lower = query.lower()
        query_words = query_lower.split()
        
        results = []
        
        for doc in self.documents:
            content = doc.get('content', '').lower()
            title = doc.get('title', '').lower()
            keywords = [kw.lower() for kw in doc.get('keywords', [])]
            
            # Calculate simple keyword match score
            score = 0
            
            # Title matches get higher score
            for word in query_words:
                if word in title:
                    score += 2
                if word in content:
                    score += 1
                if word in keywords:
                    score += 1.5
            
            # Normalize score by query length
            if score > 0:
                normalized_score = score / len(query_words)
                if normalized_score >= 0.5:  # Minimum threshold
                    doc_result = doc.copy()
                    doc_result['similarity_score'] = min(normalized_score / 3.0, 1.0)  # Cap at 1.0
                    results.append(doc_result)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        results = results[:max_results]
        
        logger.debug(f"Keyword search found {len(results)} matching documents")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.documents:
            return {
                'total_documents': 0,
                'embedding_model': 'none',
                'embedding_dimension': 0,
                'status': 'not_loaded'
            }
        
        total_chars = sum(len(doc.get('content', '')) for doc in self.documents)
        
        return {
            'total_documents': len(self.documents),
            'embedding_model': self.embedding_model,
            'embedding_dimension': self.embedding_dimension,
            'total_characters': total_chars,
            'avg_characters_per_doc': total_chars // len(self.documents) if self.documents else 0,
            'has_embeddings': self.embeddings is not None,
            'status': 'loaded'
        }
    
    def is_available(self) -> bool:
        """Check if the vector store is ready for use."""
        return len(self.documents) > 0


# Utility functions for Streamlit integration

@st.cache_resource
def create_cloud_vector_store(similarity_threshold: float = 0.3, max_results: int = 5) -> CloudVectorStore:
    """
    Create a cached CloudVectorStore instance.
    
    This function is cached by Streamlit's resource cache.
    """
    return CloudVectorStore(similarity_threshold, max_results)

@st.cache_data
def load_cloud_vector_data(_store: CloudVectorStore, data_file: str) -> bool:
    """
    Load data into the cloud vector store with caching.
    
    Args:
        _store: CloudVectorStore instance (not cached)
        data_file: Path to precomputed embeddings file
        
    Returns:
        bool: True if loaded successfully
    """
    return _store.load_data(data_file)


# Factory function for easy integration
def create_cloud_rag_system(data_file: str, 
                          similarity_threshold: float = 0.3,
                          max_results: int = 5) -> Optional[CloudVectorStore]:
    """
    Create a complete cloud RAG system with caching.
    
    Args:
        data_file: Path to precomputed embeddings file
        similarity_threshold: Minimum similarity for results
        max_results: Maximum results to return
        
    Returns:
        CloudVectorStore: Ready-to-use vector store or None if failed
    """
    try:
        # Create cached vector store
        store = create_cloud_vector_store(similarity_threshold, max_results)
        
        # Load data with caching
        if load_cloud_vector_data(store, data_file):
            return store
        else:
            st.error(f"Failed to load precomputed data from {data_file}")
            return None
            
    except Exception as e:
        st.error(f"Error creating cloud RAG system: {e}")
        logger.exception("Detailed error creating cloud RAG system")
        return None 