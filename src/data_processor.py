"""
Data Processor for GitLab Handbook Assistant

This module handles loading, processing, and preparing GitLab documentation
data for the RAG system. It includes text chunking, cleaning, and metadata extraction.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import config

logger = logging.getLogger(__name__)

class GitLabDataProcessor:
    """
    Processes GitLab documentation data for RAG system consumption.
    
    Handles loading from JSON files, text cleaning, chunking, and metadata extraction.
    """
    
    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize the data processor.
        
        Args:
            data_file: Path to the GitLab data file. Uses config default if None.
        """
        self.data_file = data_file or config.SAMPLE_DATA_FILE
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.documents = []
        self.processed_chunks = []
        
    def load_data(self) -> bool:
        """
        Load GitLab documentation data from JSON file.
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            data_path = Path(self.data_file)
            if not data_path.exists():
                logger.error(f"Data file not found: {self.data_file}")
                return False
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'documents' not in data:
                logger.error("Invalid data format: 'documents' key not found")
                return False
            
            self.documents = data['documents']
            logger.info(f"Loaded {len(self.documents)} documents from {self.data_file}")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in data file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]', ' ', text)
        
        # Remove multiple periods/dots
        text = re.sub(r'\.{2,}', '.', text)
        
        # Normalize quotes
        text = re.sub(r'[\u201c\u201d\u201e]', '"', text)  # Smart double quotes
        text = re.sub(r'[\u2018\u2019\u201a]', "'", text)  # Smart single quotes
        
        # Strip and ensure single spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_keywords(self, text: str, existing_keywords: List[str]) -> List[str]:
        """
        Extract additional keywords from text content.
        
        Args:
            text: Text to extract keywords from
            existing_keywords: Pre-existing keywords
            
        Returns:
            List[str]: Combined and deduplicated keywords
        """
        # Simple keyword extraction - find important words
        text_lower = text.lower()
        
        # Common GitLab-related terms to look for
        important_terms = [
            'gitlab', 'remote', 'work', 'team', 'process', 'review', 'merge',
            'culture', 'values', 'policy', 'hiring', 'employee', 'management',
            'development', 'engineering', 'security', 'product', 'customer'
        ]
        
        found_keywords = []
        for term in important_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        # Combine with existing keywords and remove duplicates
        all_keywords = list(set(existing_keywords + found_keywords))
        return all_keywords
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into smaller chunks for better retrieval.
        
        Args:
            document: Document to chunk
            
        Returns:
            List[Dict]: List of document chunks with metadata
        """
        content = document.get('content', '')
        if not content:
            logger.warning(f"Empty content for document: {document.get('id', 'unknown')}")
            return []
        
        # Clean the content
        cleaned_content = self.clean_text(content)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(cleaned_content)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
                
            chunk_obj = {
                'id': f"{document['id']}_chunk_{i}",
                'content': chunk,
                'source_document_id': document['id'],
                'title': document.get('title', ''),
                'url': document.get('url', ''),
                'section': document.get('section', ''),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'keywords': self.extract_keywords(chunk, document.get('keywords', [])),
                'metadata': {
                    'source_title': document.get('title', ''),
                    'source_url': document.get('url', ''),
                    'source_section': document.get('section', ''),
                    'chunk_length': len(chunk),
                    'original_doc_id': document['id']
                }
            }
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def process_documents(self) -> List[Dict[str, Any]]:
        """
        Process all loaded documents into chunks ready for embedding.
        
        Returns:
            List[Dict]: Processed document chunks
        """
        if not self.documents:
            logger.error("No documents loaded. Call load_data() first.")
            return []
        
        all_chunks = []
        
        for doc in self.documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
                logger.debug(f"Processed document '{doc.get('id')}' into {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                continue
        
        self.processed_chunks = all_chunks
        logger.info(f"Processed {len(self.documents)} documents into {len(all_chunks)} chunks")
        
        return all_chunks
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processed documents.
        
        Returns:
            Dict: Statistics about documents and chunks
        """
        if not self.processed_chunks:
            return {"error": "No processed chunks available"}
        
        total_chunks = len(self.processed_chunks)
        avg_chunk_length = sum(len(chunk['content']) for chunk in self.processed_chunks) / total_chunks
        
        sections = set(chunk['section'] for chunk in self.processed_chunks)
        unique_docs = set(chunk['source_document_id'] for chunk in self.processed_chunks)
        
        all_keywords = []
        for chunk in self.processed_chunks:
            all_keywords.extend(chunk.get('keywords', []))
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_documents": len(unique_docs),
            "total_chunks": total_chunks,
            "average_chunk_length": round(avg_chunk_length, 1),
            "sections": list(sections),
            "top_keywords": top_keywords,
            "data_file": self.data_file
        }
    
    def search_chunks_by_keywords(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search through processed chunks.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: Matching chunks with relevance scores
        """
        if not self.processed_chunks:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for chunk in self.processed_chunks:
            content_lower = chunk['content'].lower()
            chunk_words = set(content_lower.split())
            keywords_lower = [kw.lower() for kw in chunk.get('keywords', [])]
            
            # Calculate relevance score
            score = 0
            
            # Exact phrase match (highest weight)
            if query_lower in content_lower:
                score += 10
            
            # Word overlap in content
            word_overlap = len(query_words.intersection(chunk_words))
            score += word_overlap * 2
            
            # Keyword matches
            keyword_matches = sum(1 for word in query_words if word in keywords_lower)
            score += keyword_matches * 3
            
            # Title/section relevance
            title_lower = chunk.get('title', '').lower()
            section_lower = chunk.get('section', '').lower()
            
            if any(word in title_lower for word in query_words):
                score += 5
            if any(word in section_lower for word in query_words):
                score += 3
            
            if score > 0:
                result = chunk.copy()
                result['relevance_score'] = score
                results.append(result)
        
        # Sort by relevance score and return top results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]

def load_and_process_gitlab_data() -> GitLabDataProcessor:
    """
    Convenience function to load and process GitLab data.
    
    Returns:
        GitLabDataProcessor: Initialized and processed data processor
    """
    processor = GitLabDataProcessor()
    
    if not processor.load_data():
        logger.error("Failed to load GitLab data")
        return processor
    
    processor.process_documents()
    
    # Log statistics
    stats = processor.get_document_stats()
    logger.info(f"Data processing complete: {stats}")
    
    return processor

# Export main classes and functions
__all__ = [
    'GitLabDataProcessor',
    'load_and_process_gitlab_data'
] 