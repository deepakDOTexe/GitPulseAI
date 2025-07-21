#!/usr/bin/env python3
"""
Simple GitLab Two-Page Extractor

Extracts content from just 2 specific GitLab pages:
1. https://handbook.gitlab.com/
2. https://about.gitlab.com/direction/

Usage:
    python scripts/extract_two_pages.py
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitLabTwoPageExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; GitPulseAI/1.0; Educational)'
        })
        
        # The 2 specific pages to extract
        self.target_pages = [
            {
                "url": "https://handbook.gitlab.com/",
                "id": "gitlab-handbook-main",
                "title": "The GitLab Handbook",
                "section": "Company Handbook"
            },
            {
                "url": "https://about.gitlab.com/direction/",
                "id": "gitlab-direction",
                "title": "GitLab Direction",
                "section": "Product Strategy"
            }
        ]
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,!?:;()\'\"$%/]+', '', text)
        
        # Remove navigation and footer text patterns
        text = re.sub(r'© \d{4} GitLab.*?$', '', text)
        text = re.sub(r'Edit this page.*?$', '', text)
        text = re.sub(r'View source.*?$', '', text)
        
        return text
    
    def extract_keywords(self, title: str, content: str, url: str) -> List[str]:
        """Extract relevant keywords from content."""
        keywords = set()
        
        # Extract from title
        title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        keywords.update(title_words)
        
        # GitLab-specific important terms
        gitlab_terms = [
            'gitlab', 'handbook', 'direction', 'strategy', 'devops', 'devsecops',
            'values', 'culture', 'remote', 'engineering', 'security', 'product',
            'development', 'ci/cd', 'continuous', 'integration', 'deployment',
            'collaboration', 'transparency', 'efficiency', 'iteration', 'results',
            'diversity', 'inclusion', 'open', 'source', 'community', 'platform',
            'workflow', 'pipeline', 'merge', 'review', 'planning', 'monitoring'
        ]
        
        content_lower = content.lower()
        for term in gitlab_terms:
            if term in content_lower:
                keywords.add(term)
        
        # Extract capitalized words (likely important terms)
        important_words = re.findall(r'\b[A-Z][a-z]{3,}\b', content)
        keywords.update([word.lower() for word in important_words[:20]])
        
        return sorted(list(keywords))[:20]
    
    def extract_page_content(self, page_info: Dict) -> Optional[Dict]:
        """Extract content from a single page."""
        url = page_info["url"]
        
        try:
            logger.info(f"Extracting content from: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['nav', 'footer', 'header', 'aside', 'script', 'style']):
                element.decompose()
            
            # Remove elements with specific classes/ids that are navigation
            for element in soup.find_all(class_=['navigation', 'nav', 'sidebar', 'footer', 'header']):
                element.decompose()
                
            for element in soup.find_all(id=['navigation', 'nav', 'sidebar', 'footer', 'header']):
                element.decompose()
            
            # Extract main content
            content_selectors = [
                'main',
                '.content',
                'article',
                '.handbook-content',
                '.markdown-body',
                'body'
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            if not content:
                # Fallback to body
                content = soup.find('body')
            
            if not content:
                logger.error(f"No content found for {url}")
                return None
            
            # Get text content
            content_text = content.get_text()
            content_text = self.clean_text(content_text)
            
            if len(content_text) < 200:
                logger.warning(f"Very little content extracted from {url}")
            
            # Extract keywords
            keywords = self.extract_keywords(page_info["title"], content_text, url)
            
            document = {
                "id": page_info["id"],
                "title": page_info["title"],
                "url": url,
                "section": page_info["section"],
                "content": content_text,
                "keywords": keywords,
                "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "content_length": len(content_text)
            }
            
            logger.info(f"Successfully extracted {len(content_text)} characters from: {page_info['title']}")
            return document
            
        except requests.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
        
        return None
    
    def extract_all_pages(self) -> Dict:
        """Extract content from both target pages."""
        logger.info("Starting GitLab two-page extraction...")
        
        documents = []
        
        for page_info in self.target_pages:
            document = self.extract_page_content(page_info)
            if document:
                documents.append(document)
            
            # Be respectful with delays
            time.sleep(2)
        
        logger.info(f"Extraction complete. Collected {len(documents)} documents")
        
        return {
            "documents": documents,
            "metadata": {
                "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(documents),
                "source": "GitLab Handbook + Direction Pages",
                "extractor_version": "1.0",
                "pages_extracted": [doc["url"] for doc in documents]
            }
        }


def main():
    """Main extraction function."""
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize extractor
    extractor = GitLabTwoPageExtractor()
    
    # Extract content from both pages
    data = extractor.extract_all_pages()
    
    if not data["documents"]:
        logger.error("No documents were successfully extracted!")
        return
    
    # Save to JSON file
    output_file = output_dir / "gitlab_two_pages.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data['documents'])} documents to {output_file}")
    
    # Print summary
    print(f"\n🎉 GitLab Two-Page Extraction Complete!")
    print(f"📄 Pages processed: {len(data['documents'])}")
    print(f"💾 Saved to: {output_file}")
    
    # Show content summary
    for doc in data["documents"]:
        print(f"\n📋 {doc['title']}:")
        print(f"   URL: {doc['url']}")
        print(f"   Content: {doc['content_length']:,} characters")
        print(f"   Keywords: {len(doc['keywords'])} terms")
    
    print(f"\n🔧 To use this data:")
    print(f"1. Update your .env file:")
    print(f"   SAMPLE_DATA_FILE=data/gitlab_two_pages.json")
    print(f"2. Restart your GitPulseAI app")
    print(f"   streamlit run app.py")


if __name__ == "__main__":
    main() 