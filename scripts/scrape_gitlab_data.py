#!/usr/bin/env python3
"""
GitLab Handbook Data Scraper

This script scrapes GitLab's public handbook pages and converts them 
to the JSON format expected by GitPulseAI.

Usage:
    python scripts/scrape_gitlab_data.py
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Optional, Dict, List
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitLabHandbookScraper:
    def __init__(self):
        self.base_url = "https://about.gitlab.com"
        self.handbook_url = "https://about.gitlab.com/handbook/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; GitPulseAI/1.0; Educational)'
        })
        self.documents = []
        
        # Key handbook sections to scrape
        self.target_sections = [
            "/handbook/values/",
            "/handbook/company/culture/",
            "/handbook/engineering/",
            "/handbook/security/",
            "/handbook/people-group/",
            "/handbook/hiring/",
            "/handbook/communication/",
            "/handbook/leadership/",
            "/handbook/product/",
            "/handbook/marketing/",
            "/handbook/sales/",
            "/handbook/finance/",
            "/handbook/legal/",
            "/handbook/support/",
            "/handbook/customer-success/",
        ]
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,!?:;()\'\"]+', '', text)
        
        return text
    
    def extract_keywords(self, title: str, content: str, url: str) -> List[str]:
        """Extract relevant keywords from title, content, and URL."""
        keywords = set()
        
        # Extract from title
        title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        keywords.update(title_words)
        
        # Extract from URL path
        url_parts = urlparse(url).path.split('/')
        url_words = [part.replace('-', ' ').replace('_', ' ') for part in url_parts if part]
        for part in url_words:
            keywords.update(re.findall(r'\b[a-zA-Z]{3,}\b', part.lower()))
        
        # Extract important terms from content
        content_lower = content.lower()
        important_terms = [
            'gitlab', 'remote', 'values', 'culture', 'engineering', 'security',
            'diversity', 'inclusion', 'collaboration', 'transparency', 'efficiency',
            'iteration', 'results', 'development', 'process', 'workflow', 'policy',
            'guideline', 'practice', 'standard', 'procedure', 'framework'
        ]
        
        for term in important_terms:
            if term in content_lower:
                keywords.add(term)
        
        # Limit to most relevant keywords
        return sorted(list(keywords))[:15]
    
    def scrape_page(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """Scrape a single GitLab handbook page."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Scraping: {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = soup.find('h1')
                title_text = title.get_text().strip() if title else "Unknown"
                
                # Extract main content
                content_selectors = [
                    'main',
                    '.handbook-content', 
                    '.content',
                    'article',
                    '.post-content'
                ]
                
                content = None
                for selector in content_selectors:
                    content = soup.select_one(selector)
                    if content:
                        break
                
                if not content:
                    # Fallback: get all paragraphs
                    paragraphs = soup.find_all('p')
                    content_text = ' '.join([p.get_text() for p in paragraphs])
                else:
                    # Remove navigation, footer, and sidebar elements
                    for element in content.find_all(['nav', 'footer', 'aside', '.sidebar']):
                        element.decompose()
                    content_text = content.get_text()
                
                # Clean content
                content_text = self.clean_text(content_text)
                
                if len(content_text) < 100:  # Skip pages with minimal content
                    logger.warning(f"Skipping {url} - insufficient content")
                    return None
                
                # Extract section from URL
                url_path = urlparse(url).path
                section_parts = url_path.split('/')[2:]  # Skip '', 'handbook'
                section = ' > '.join(part.replace('-', ' ').title() for part in section_parts if part)
                
                # Generate unique ID
                doc_id = url_path.replace('/', '-').strip('-')
                
                # Extract keywords
                keywords = self.extract_keywords(title_text, content_text, url)
                
                document = {
                    "id": doc_id,
                    "title": title_text,
                    "url": url,
                    "section": section or "General",
                    "content": content_text,
                    "keywords": keywords,
                    "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                logger.info(f"Successfully scraped: {title_text}")
                return document
                
            except requests.RequestException as e:
                logger.error(f"Request error for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                break
        
        return None
    
    def discover_handbook_pages(self, base_section: str, max_depth: int = 2) -> List[str]:
        """Discover handbook pages by following links."""
        discovered_urls = set()
        to_visit = [(base_section, 0)]
        visited = set()
        
        while to_visit:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
                
            visited.add(url)
            full_url = urljoin(self.base_url, url)
            
            try:
                response = self.session.get(full_url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find handbook links
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(full_url, href)
                    parsed = urlparse(absolute_url)
                    
                    # Only include handbook pages from GitLab domain
                    if (parsed.netloc == 'about.gitlab.com' and 
                        parsed.path.startswith('/handbook/') and
                        not parsed.path.endswith('.pdf') and
                        '#' not in parsed.path):
                        
                        discovered_urls.add(absolute_url)
                        
                        # Add to queue for further exploration
                        if depth < max_depth:
                            to_visit.append((parsed.path, depth + 1))
                
                time.sleep(1)  # Be respectful with rate limiting
                
            except Exception as e:
                logger.error(f"Error discovering pages from {url}: {e}")
                continue
        
        return list(discovered_urls)
    
    def scrape_handbook(self, max_pages: int = 50) -> Dict:
        """Scrape GitLab handbook pages."""
        logger.info("Starting GitLab handbook scraping...")
        
        all_urls = set()
        
        # Discover pages from target sections
        for section in self.target_sections:
            logger.info(f"Discovering pages in section: {section}")
            discovered = self.discover_handbook_pages(section, max_depth=1)
            all_urls.update(discovered)
            
            if len(all_urls) >= max_pages:
                break
        
        # Limit to max_pages
        urls_to_scrape = list(all_urls)[:max_pages]
        
        logger.info(f"Found {len(urls_to_scrape)} pages to scrape")
        
        # Scrape each page
        for i, url in enumerate(urls_to_scrape, 1):
            logger.info(f"Progress: {i}/{len(urls_to_scrape)}")
            
            document = self.scrape_page(url)
            if document:
                self.documents.append(document)
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"Scraping complete. Collected {len(self.documents)} documents")
        
        return {
            "documents": self.documents,
            "metadata": {
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(self.documents),
                "source": "GitLab Handbook",
                "scraper_version": "1.0"
            }
        }


def main():
    """Main scraping function."""
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize scraper
    scraper = GitLabHandbookScraper()
    
    # Scrape handbook (adjust max_pages as needed)
    data = scraper.scrape_handbook(max_pages=30)
    
    # Save to JSON file
    output_file = output_dir / "real_gitlab_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data['documents'])} documents to {output_file}")
    logger.info("Scraping complete!")
    
    # Print summary
    print(f"\n🎉 GitLab Handbook Scraping Complete!")
    print(f"📄 Documents collected: {len(data['documents'])}")
    print(f"💾 Saved to: {output_file}")
    print(f"\n📋 To use this data:")
    print(f"1. Update your .env file:")
    print(f"   SAMPLE_DATA_FILE=data/real_gitlab_data.json")
    print(f"2. Restart your GitPulseAI app")


if __name__ == "__main__":
    main() 