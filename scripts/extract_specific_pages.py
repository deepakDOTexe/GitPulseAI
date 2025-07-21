#!/usr/bin/env python3
"""
GitLab Specific Pages Extractor

Extracts content from specific GitLab handbook policy pages.
Includes anti-harassment policy and other key policies.

Usage:
    python scripts/extract_specific_pages.py
"""

import requests
from bs4 import BeautifulSoup, Tag
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitLabSpecificPagesExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; GitPulseAI/1.0; Educational)'
        })
        
        # Specific policy pages to extract
        self.target_pages = [
            {
                "url": "https://handbook.gitlab.com/handbook/people-group/anti-harassment/",
                "id": "anti-harassment-policy",
                "title": "Anti-Harassment Policy",
                "section": "People Group Policies"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/values/",
                "id": "gitlab-values",
                "title": "GitLab Values",
                "section": "Company Values"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/people-group/inclusion/",
                "id": "diversity-inclusion",
                "title": "Diversity, Inclusion & Belonging",
                "section": "People Group Policies"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/communication/",
                "id": "communication-guidelines",
                "title": "Communication Guidelines",
                "section": "Company Processes"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/engineering/",
                "id": "engineering-practices",
                "title": "Engineering Practices",
                "section": "Engineering"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/security/",
                "id": "security-policies",
                "title": "Security Policies",
                "section": "Security"
            }
        ]
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,!?:;()\'\"$%/\n]+', '', text)
        
        # Remove navigation and footer text patterns
        text = re.sub(r'© \d{4} GitLab.*?$', '', text)
        text = re.sub(r'Edit this page.*?$', '', text)
        text = re.sub(r'View source.*?$', '', text)
        text = re.sub(r'Talk to an Expert.*?$', '', text)
        text = re.sub(r'Get free trial.*?$', '', text)
        
        return text
    
    def extract_keywords(self, title: str, content: str, url: str) -> List[str]:
        """Extract relevant keywords from content."""
        keywords = set()
        
        # Extract from title
        title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        keywords.update(title_words)
        
        # GitLab-specific important terms
        gitlab_terms = [
            'gitlab', 'handbook', 'policy', 'harassment', 'anti-harassment',
            'values', 'culture', 'remote', 'engineering', 'security', 'inclusion',
            'diversity', 'belonging', 'communication', 'collaboration', 'transparency', 
            'efficiency', 'iteration', 'results', 'development', 'team', 'employee',
            'workplace', 'conduct', 'behavior', 'respect', 'inclusive', 'discrimination',
            'retaliation', 'complaint', 'investigation', 'reporting', 'violation'
        ]
        
        content_lower = content.lower()
        for term in gitlab_terms:
            if term in content_lower:
                keywords.add(term)
        
        # Extract important capitalized words (likely key terms)
        important_words = re.findall(r'\b[A-Z][a-z]{3,}\b', content)
        keywords.update([word.lower() for word in important_words[:15]])
        
        return sorted(list(keywords))[:25]
    
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
            
            # Remove elements with navigation/menu classes
            for element in soup.find_all(class_=['navigation', 'nav', 'sidebar', 'footer', 'header', 'menu']):
                element.decompose()
                
            for element in soup.find_all(id=['navigation', 'nav', 'sidebar', 'footer', 'header', 'menu']):
                element.decompose()
            
            # Extract main content - try multiple selectors
            content_selectors = [
                'main',
                '.content',
                'article',
                '.handbook-content',
                '.markdown-body',
                '.md-content',
                '.post-content',
                '.page-content'
            ]
            
            content = None
            for selector in content_selectors:
                candidate = soup.select_one(selector)
                if candidate and hasattr(candidate, 'get_text') and len(candidate.get_text().strip()) > 500:
                    content = candidate
                    break
            
            if not content:
                # Fallback to body but remove obvious navigation
                content = soup.find('body')
                if content and isinstance(content, Tag):
                    # Remove common navigation elements
                    for nav_elem in content.find_all(['nav', 'header', 'footer']):
                        nav_elem.decompose()
            
            if not content or not isinstance(content, Tag):
                logger.error(f"No valid content found for {url}")
                return None
            
            # Get text content
            content_text = content.get_text()
            content_text = self.clean_text(content_text)
            
            # Validate content quality
            if len(content_text) < 300:
                logger.warning(f"Very little content extracted from {url}: {len(content_text)} chars")
                return None
            
            # Check if this looks like actual content vs navigation
            if self._is_navigation_content(content_text):
                logger.warning(f"Content appears to be navigation/menu for {url}")
                return None
            
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
    
    def _is_navigation_content(self, text: str) -> bool:
        """Check if content is primarily navigation/menu items."""
        # Look for patterns that suggest navigation content
        nav_indicators = [
            'About GitLab Values Mission Vision',
            'Company Handbook People Group',
            'Engineering Customer Support Department',
            'Marketing Sales Finance Product'
        ]
        
        for indicator in nav_indicators:
            if indicator in text:
                return True
        
        # Check ratio of short lines (common in navigation)
        lines = text.split('\n')
        short_lines = [line for line in lines if len(line.strip()) < 50 and len(line.strip()) > 5]
        
        if len(lines) > 0 and len(short_lines) / len(lines) > 0.7:
            return True
            
        return False
    
    def extract_all_pages(self) -> Dict:
        """Extract content from all target pages."""
        logger.info("Starting GitLab specific pages extraction...")
        
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
                "source": "GitLab Handbook Specific Pages",
                "extractor_version": "1.0",
                "pages_extracted": [doc["url"] for doc in documents],
                "note": "Includes anti-harassment policy and key GitLab policies"
            }
        }


def main():
    """Main extraction function."""
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize extractor
    extractor = GitLabSpecificPagesExtractor()
    
    # Extract content from specific pages
    data = extractor.extract_all_pages()
    
    if not data["documents"]:
        logger.error("No documents were successfully extracted!")
        return
    
    # Save to JSON file
    output_file = output_dir / "gitlab_specific_policies.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data['documents'])} documents to {output_file}")
    
    # Print summary
    print(f"\n🎉 GitLab Specific Pages Extraction Complete!")
    print(f"📄 Pages processed: {len(data['documents'])}")
    print(f"💾 Saved to: {output_file}")
    
    # Show content summary
    for doc in data["documents"]:
        print(f"\n📋 {doc['title']}:")
        print(f"   URL: {doc['url']}")
        print(f"   Content: {doc['content_length']:,} characters")
        print(f"   Keywords: {len(doc['keywords'])} terms")
        if 'anti-harassment' in doc['id'] or 'harassment' in doc['title'].lower():
            print(f"   ✅ Contains anti-harassment policy content")
    
    print(f"\n🔧 To use this data:")
    print(f"1. Update your .env file:")
    print(f"   SAMPLE_DATA_FILE=data/gitlab_specific_policies.json")
    print(f"2. Restart your GitPulseAI app")
    print(f"   streamlit run app.py")


if __name__ == "__main__":
    main() 