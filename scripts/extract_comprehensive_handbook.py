#!/usr/bin/env python3
"""
Comprehensive GitLab Handbook Extractor

Extracts content from all major GitLab handbook sections to create
a comprehensive knowledge base for the chatbot.

Usage:
    python scripts/extract_comprehensive_handbook.py
"""

import requests
from bs4 import BeautifulSoup, Tag
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
import re
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitLabComprehensiveExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; GitPulseAI/1.0; Educational)'
        })
        
        # Comprehensive list of GitLab handbook sections to extract
        self.target_pages = [
            # Company & Culture
            {
                "url": "https://handbook.gitlab.com/handbook/values/",
                "id": "gitlab-values",
                "title": "GitLab Values",
                "section": "Company Culture"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/company/culture/",
                "id": "company-culture",
                "title": "Company Culture",
                "section": "Company Culture"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/communication/",
                "id": "communication-guidelines",
                "title": "Communication Guidelines", 
                "section": "Company Processes"
            },
            
            # People Group Policies
            {
                "url": "https://handbook.gitlab.com/handbook/people-group/anti-harassment/",
                "id": "anti-harassment-policy",
                "title": "Anti-Harassment Policy",
                "section": "People Group Policies"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/hiring/",
                "id": "hiring-process",
                "title": "Hiring Process",
                "section": "People Group Policies"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/people-group/inclusion/",
                "id": "diversity-inclusion",
                "title": "Diversity, Inclusion & Belonging",
                "section": "People Group Policies"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/people-group/learning-and-development/",
                "id": "learning-development",
                "title": "Learning & Development",
                "section": "People Group Policies"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/people-group/general-onboarding/",
                "id": "onboarding",
                "title": "Onboarding Process",
                "section": "People Group Policies"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/people-group/offboarding/",
                "id": "offboarding",
                "title": "Offboarding Process",
                "section": "People Group Policies"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/total-rewards/",
                "id": "total-rewards",
                "title": "Total Rewards & Benefits",
                "section": "People Group Policies"
            },
            
            # Engineering & Development
            {
                "url": "https://handbook.gitlab.com/handbook/engineering/",
                "id": "engineering-practices",
                "title": "Engineering Practices",
                "section": "Engineering"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/engineering/development/",
                "id": "development-department",
                "title": "Development Department",
                "section": "Engineering"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/engineering/infrastructure/",
                "id": "infrastructure",
                "title": "Infrastructure",
                "section": "Engineering"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/engineering/developer-experience/",
                "id": "developer-experience",
                "title": "Developer Experience",
                "section": "Engineering"
            },
            
            # Security
            {
                "url": "https://handbook.gitlab.com/handbook/security/",
                "id": "security-policies",
                "title": "Security Policies",
                "section": "Security"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/security/product-security/",
                "id": "product-security",
                "title": "Product Security",
                "section": "Security"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/security/security-operations/",
                "id": "security-operations",
                "title": "Security Operations",
                "section": "Security"
            },
            
            # Product & Marketing
            {
                "url": "https://handbook.gitlab.com/handbook/product/",
                "id": "product-management",
                "title": "Product Management",
                "section": "Product"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/marketing/",
                "id": "marketing",
                "title": "Marketing",
                "section": "Marketing"
            },
            
            # Sales & Customer Success
            {
                "url": "https://handbook.gitlab.com/handbook/sales/",
                "id": "sales-process",
                "title": "Sales Process",
                "section": "Sales"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/customer-success/",
                "id": "customer-success",
                "title": "Customer Success",
                "section": "Sales"
            },
            
            # Finance & Legal
            {
                "url": "https://handbook.gitlab.com/handbook/finance/",
                "id": "finance",
                "title": "Finance",
                "section": "Finance"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/legal/",
                "id": "legal-affairs",
                "title": "Legal and Corporate Affairs",
                "section": "Legal"
            },
            
            # Leadership & Management
            {
                "url": "https://handbook.gitlab.com/handbook/leadership/",
                "id": "leadership",
                "title": "Leadership",
                "section": "Leadership"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/ceo/",
                "id": "ceo-readme",
                "title": "CEO Readme",
                "section": "Leadership"
            },
            
            # Remote Work & TeamOps
            {
                "url": "https://handbook.gitlab.com/handbook/company/culture/all-remote/",
                "id": "all-remote",
                "title": "All-Remote Work",
                "section": "Remote Work"
            },
            {
                "url": "https://handbook.gitlab.com/handbook/teamops/",
                "id": "teamops",
                "title": "TeamOps",
                "section": "Remote Work"
            }
        ]
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation and newlines
        text = re.sub(r'[^\w\s\-.,!?:;()\'\"$%/\n]+', '', text)
        
        # Remove navigation and footer text patterns
        text = re.sub(r'© \d{4} GitLab.*?$', '', text)
        text = re.sub(r'Edit this page.*?$', '', text)
        text = re.sub(r'View source.*?$', '', text)
        text = re.sub(r'Talk to an Expert.*?$', '', text)
        text = re.sub(r'Get free trial.*?$', '', text)
        text = re.sub(r'Last modified.*?View page source.*?$', '', text)
        
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
            'workplace', 'conduct', 'behavior', 'respect', 'inclusive', 'hiring',
            'onboarding', 'offboarding', 'leadership', 'management', 'product',
            'marketing', 'sales', 'customer', 'success', 'finance', 'legal',
            'all-remote', 'teamops', 'devops', 'devsecops', 'infrastructure',
            'performance', 'review', 'career', 'growth', 'training', 'learning'
        ]
        
        content_lower = content.lower()
        for term in gitlab_terms:
            if term in content_lower:
                keywords.add(term)
        
        # Extract important capitalized words (likely key terms)
        important_words = re.findall(r'\b[A-Z][a-z]{3,}\b', content)
        keywords.update([word.lower() for word in important_words[:15]])
        
        return sorted(list(keywords))[:30]
    
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
            for element in soup.find_all(class_=['navigation', 'nav', 'sidebar', 'footer', 'header', 'menu', 'breadcrumb']):
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
                '.page-content',
                '[role="main"]'
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
            if len(content_text) < 500:
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
        if len(lines) > 10:
            short_lines = [line for line in lines if len(line.strip()) < 50 and len(line.strip()) > 5]
            if len(short_lines) / len(lines) > 0.6:
                return True
            
        return False
    
    def extract_all_pages(self, max_pages: Optional[int] = None) -> Dict:
        """Extract content from all target pages."""
        logger.info("Starting comprehensive GitLab handbook extraction...")
        
        documents = []
        pages_to_process = self.target_pages
        
        if max_pages:
            pages_to_process = self.target_pages[:max_pages]
            logger.info(f"Limiting extraction to {max_pages} pages")
        
        for i, page_info in enumerate(pages_to_process, 1):
            logger.info(f"Progress: {i}/{len(pages_to_process)}")
            
            document = self.extract_page_content(page_info)
            if document:
                documents.append(document)
            
            # Be respectful with delays - don't overwhelm the server
            time.sleep(2)
        
        logger.info(f"Extraction complete. Collected {len(documents)} documents")
        
        return {
            "documents": documents,
            "metadata": {
                "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(documents),
                "source": "GitLab Handbook Comprehensive",
                "extractor_version": "1.0",
                "pages_extracted": [doc["url"] for doc in documents],
                "sections_covered": list(set([doc["section"] for doc in documents])),
                "note": "Comprehensive GitLab handbook content covering all major sections"
            }
        }


def main():
    """Main extraction function."""
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize extractor
    extractor = GitLabComprehensiveExtractor()
    
    # Ask user for extraction scope
    print("🎯 GitLab Handbook Comprehensive Extraction")
    print(f"📄 Total pages available: {len(extractor.target_pages)}")
    print()
    print("Choose extraction scope:")
    print("1. Extract ALL pages (~30-40 pages, takes 5-10 minutes)")
    print("2. Extract first 15 pages (good balance, takes ~3 minutes)")  
    print("3. Extract first 10 pages (quick test, takes ~2 minutes)")
    print()
    
    choice = input("Enter choice (1-3, default=2): ").strip()
    
    if choice == "1":
        max_pages = None
        print("🚀 Extracting ALL handbook pages...")
    elif choice == "3":
        max_pages = 10
        print("🚀 Extracting first 10 pages...")
    else:
        max_pages = 15
        print("🚀 Extracting first 15 pages...")
    
    # Extract content
    data = extractor.extract_all_pages(max_pages=max_pages)
    
    if not data["documents"]:
        logger.error("No documents were successfully extracted!")
        return
    
    # Save to JSON file
    output_file = output_dir / "gitlab_comprehensive_handbook.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data['documents'])} documents to {output_file}")
    
    # Print summary
    print(f"\n🎉 GitLab Comprehensive Extraction Complete!")
    print(f"📄 Pages extracted: {len(data['documents'])}")
    print(f"💾 Saved to: {output_file}")
    print(f"📂 Sections covered: {', '.join(data['metadata']['sections_covered'])}")
    
    # Show content summary by section
    sections = {}
    total_chars = 0
    for doc in data["documents"]:
        section = doc["section"]
        if section not in sections:
            sections[section] = []
        sections[section].append(doc)
        total_chars += doc["content_length"]
    
    print(f"\n📊 Content Summary:")
    print(f"📝 Total content: {total_chars:,} characters")
    print(f"📋 By section:")
    for section, docs in sections.items():
        char_count = sum(doc["content_length"] for doc in docs)
        print(f"   {section}: {len(docs)} pages ({char_count:,} chars)")
    
    print(f"\n🔧 To use this comprehensive data:")
    print(f"1. Update your .env file:")
    print(f"   SAMPLE_DATA_FILE=data/gitlab_comprehensive_handbook.json")
    print(f"2. Restart your GitPulseAI app")
    print(f"   streamlit run app.py")
    print(f"\n✨ Your chatbot will now have comprehensive GitLab knowledge!")


if __name__ == "__main__":
    main() 