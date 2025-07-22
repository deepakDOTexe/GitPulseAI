#!/usr/bin/env python3
"""
Comprehensive GitLab Data Scraper

This script follows EVERY possible hyperlink from the main GitLab pages:
- https://handbook.gitlab.com/
- https://about.gitlab.com/direction/

It recursively crawls all internal GitLab links to ensure complete data coverage,
solving issues like missing diversity initiatives and other content gaps.

Usage:
    python scripts/comprehensive_gitlab_scraper.py
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Optional, Dict, List, Set
import re
from collections import deque
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveGitLabScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; GitPulseAI/1.0; Educational)'
        })
        
        # Starting URLs - the two main pages
        self.starting_urls = [
            "https://handbook.gitlab.com/",
            "https://about.gitlab.com/direction/"
        ]
        
        # Domain filtering - only scrape GitLab domains
        self.allowed_domains = {
            'handbook.gitlab.com',
            'about.gitlab.com'
        }
        
        # URL filtering patterns to skip
        self.skip_patterns = [
            r'\.pdf$',
            r'\.png$', r'\.jpg$', r'\.jpeg$', r'\.gif$', r'\.svg$',
            r'\.zip$', r'\.tar\.gz$',
            r'/edit$',
            r'/raw$',
            r'/blame$',
            r'/commits',
            r'/issues',
            r'/merge_requests',
            r'#',  # Skip anchor links
            r'\?',  # Skip query parameters for now
            r'mailto:',
            r'tel:',
            r'javascript:',
        ]
        
        self.visited_urls = set()
        self.failed_urls = set()
        self.documents = []
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled."""
        parsed = urlparse(url)
        
        # Must be from allowed domains
        if parsed.netloc not in self.allowed_domains:
            return False
            
        # Skip URLs matching skip patterns
        for pattern in self.skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
                
        return True
    
    def extract_links_from_page(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extract all valid internal links from a page."""
        links = []
        
        for link_tag in soup.find_all('a', href=True):
            href = link_tag.get('href')
            if not href:
                continue
                
            # Convert relative URLs to absolute
            absolute_url = urljoin(url, href)
            
            # Clean the URL (remove fragments, normalize)
            parsed = urlparse(absolute_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if clean_url.endswith('/'):
                clean_url = clean_url.rstrip('/')
                
            if self.is_valid_url(clean_url) and clean_url not in self.visited_urls:
                links.append(clean_url)
                
        return list(set(links))  # Remove duplicates
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove navigation and footer patterns
        text = re.sub(r'(Skip to content|Table of contents|Edit this page|Last modified|View page source)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'©\s*\d{4}\s*GitLab.*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def extract_keywords(self, title: str, content: str, url: str) -> List[str]:
        """Extract relevant keywords from title, content, and URL."""
        keywords = set()
        
        # Extract from title
        title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        keywords.update(title_words[:5])  # Top 5 from title
        
        # Extract from URL path
        url_parts = urlparse(url).path.split('/')
        url_words = [part.replace('-', ' ').replace('_', ' ') for part in url_parts if part]
        for part in url_words:
            keywords.update(re.findall(r'\b[a-zA-Z]{3,}\b', part.lower())[:2])
        
        # Important GitLab terms
        important_terms = [
            'gitlab', 'remote', 'values', 'culture', 'engineering', 'security',
            'diversity', 'inclusion', 'collaboration', 'transparency', 'efficiency',
            'iteration', 'results', 'development', 'process', 'workflow', 'policy',
            'guideline', 'practice', 'standard', 'procedure', 'framework', 'hiring',
            'onboarding', 'leadership', 'communication', 'marketing', 'sales',
            'customer', 'product', 'legal', 'finance', 'anti-harassment',
            'belonging', 'ally', 'tmrg', 'initiatives', 'team', 'member'
        ]
        
        content_lower = content.lower()
        for term in important_terms:
            if term in content_lower:
                keywords.add(term)
        
        # Limit to most relevant keywords
        return sorted(list(keywords))[:20]
    
    def scrape_page(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """Scrape a single page and extract content."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Scraping: {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title_tag = soup.find('h1')
                if not title_tag:
                    title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else "Unknown"
                
                # Remove navigation, sidebar, footer elements
                for element in soup.find_all(['nav', 'footer', 'aside', '.sidebar', '.toc', '.breadcrumb']):
                    element.decompose()
                
                # Extract main content
                content_selectors = [
                    'main',
                    '.content',
                    'article',
                    '.handbook-content',
                    '.post-content',
                    '.markdown-body',
                    '.content-wrapper'
                ]
                
                content = None
                for selector in content_selectors:
                    content = soup.select_one(selector)
                    if content:
                        break
                
                if not content:
                    # Fallback: try body or get all paragraphs
                    content = soup.find('body')
                    if not content:
                        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                        content_text = '\n'.join([p.get_text() for p in paragraphs])
                    else:
                        content_text = content.get_text()
                else:
                    content_text = content.get_text()
                
                # Clean content
                content_text = self.clean_text(content_text)
                
                # Skip pages with minimal content
                if len(content_text) < 200:
                    logger.warning(f"Skipping {url} - insufficient content ({len(content_text)} chars)")
                    return None
                
                # Extract section from URL
                url_path = urlparse(url).path
                if 'handbook.gitlab.com' in url:
                    section_parts = url_path.split('/')[2:]  # Skip '', 'handbook'
                    section = ' > '.join(part.replace('-', ' ').title() for part in section_parts if part)
                    if not section:
                        section = "Handbook Root"
                else:
                    section_parts = url_path.split('/')[1:]  # Skip ''
                    section = ' > '.join(part.replace('-', ' ').title() for part in section_parts if part)
                    if not section:
                        section = "GitLab Direction"
                
                # Generate unique ID
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                doc_id = f"gitlab-{section.lower().replace(' ', '-').replace('>', '-')}-{url_hash}"
                
                # Extract keywords
                keywords = self.extract_keywords(title, content_text, url)
                
                document = {
                    "id": doc_id,
                    "title": title,
                    "url": url,
                    "section": section,
                    "content": content_text,
                    "keywords": keywords,
                    "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "content_length": len(content_text)
                }
                
                logger.info(f"✅ Successfully scraped: {title} ({len(content_text):,} chars)")
                return document
                
            except requests.RequestException as e:
                logger.error(f"Request error for {url}: {e}")
                if attempt == max_retries - 1:
                    self.failed_urls.add(url)
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                if attempt == max_retries - 1:
                    self.failed_urls.add(url)
            
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
        
        return None
    
    def comprehensive_crawl(self, max_pages: int = 500) -> Dict:
        """Comprehensively crawl all GitLab pages starting from the main URLs."""
        logger.info("🚀 Starting comprehensive GitLab crawl...")
        logger.info(f"📄 Starting URLs: {self.starting_urls}")
        logger.info(f"🎯 Max pages limit: {max_pages}")
        
        # Use a queue for breadth-first crawling
        url_queue = deque(self.starting_urls)
        
        while url_queue and len(self.visited_urls) < max_pages:
            current_url = url_queue.popleft()
            
            if current_url in self.visited_urls or current_url in self.failed_urls:
                continue
            
            self.visited_urls.add(current_url)
            
            logger.info(f"📊 Progress: {len(self.visited_urls)}/{max_pages} pages | Queue: {len(url_queue)} | Found: {len(self.documents)} docs")
            
            # Scrape current page
            document = self.scrape_page(current_url)
            if document:
                self.documents.append(document)
                
                # Get the page content again for link extraction
                try:
                    response = self.session.get(current_url, timeout=30)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract new links to add to queue
                    new_links = self.extract_links_from_page(current_url, soup)
                    for link in new_links:
                        if link not in self.visited_urls and link not in [item for item in url_queue]:
                            url_queue.append(link)
                    
                    logger.info(f"🔗 Found {len(new_links)} new links from {current_url}")
                    
                except Exception as e:
                    logger.error(f"Error extracting links from {current_url}: {e}")
            
            # Rate limiting - be respectful
            time.sleep(1)
        
        logger.info(f"🎉 Crawling complete!")
        logger.info(f"📄 Total pages visited: {len(self.visited_urls)}")
        logger.info(f"📚 Documents extracted: {len(self.documents)}")
        logger.info(f"❌ Failed URLs: {len(self.failed_urls)}")
        
        return {
            "documents": self.documents,
            "metadata": {
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(self.documents),
                "total_urls_visited": len(self.visited_urls),
                "failed_urls": len(self.failed_urls),
                "starting_urls": self.starting_urls,
                "source": "Comprehensive GitLab Crawl",
                "scraper_version": "2.0",
                "domains_crawled": list(self.allowed_domains)
            }
        }


def main():
    """Main scraping function."""
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    print("🎯 Comprehensive GitLab Data Scraper")
    print("📖 This will crawl EVERY possible link from:")
    print("   • https://handbook.gitlab.com/")
    print("   • https://about.gitlab.com/direction/")
    print()
    
    # Ask user for crawl scope
    print("Choose crawl scope:")
    print("1. Comprehensive crawl (500+ pages, 20-30 minutes) - RECOMMENDED")
    print("2. Large crawl (200 pages, ~10 minutes)")
    print("3. Medium crawl (100 pages, ~5 minutes)")
    print("4. Test crawl (50 pages, ~3 minutes)")
    print()
    
    choice = input("Enter choice (1-4, default=1): ").strip()
    
    if choice == "2":
        max_pages = 200
        print("🚀 Starting large crawl (200 pages)...")
    elif choice == "3":
        max_pages = 100
        print("🚀 Starting medium crawl (100 pages)...")
    elif choice == "4":
        max_pages = 50
        print("🚀 Starting test crawl (50 pages)...")
    else:
        max_pages = 500
        print("🚀 Starting comprehensive crawl (500+ pages)...")
    
    # Initialize scraper
    scraper = ComprehensiveGitLabScraper()
    
    # Start crawling
    start_time = time.time()
    data = scraper.comprehensive_crawl(max_pages=max_pages)
    end_time = time.time()
    
    if not data["documents"]:
        logger.error("❌ No documents were successfully extracted!")
        return
    
    # Save to JSON file
    output_file = output_dir / "gitlab_comprehensive_crawl.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Calculate statistics
    total_chars = sum(doc['content_length'] for doc in data["documents"])
    avg_chars = total_chars / len(data["documents"]) if data["documents"] else 0
    
    # Print summary
    print(f"\n🎉 Comprehensive GitLab Crawl Complete!")
    print(f"⏰ Total time: {(end_time - start_time) / 60:.1f} minutes")
    print(f"📄 Documents extracted: {len(data['documents'])}")
    print(f"📚 Total content: {total_chars:,} characters")
    print(f"📊 Average per document: {avg_chars:,.0f} characters")
    print(f"🌐 URLs visited: {data['metadata']['total_urls_visited']}")
    print(f"❌ Failed URLs: {data['metadata']['failed_urls']}")
    print(f"💾 Saved to: {output_file}")
    
    # Show content breakdown by section
    sections = {}
    for doc in data["documents"]:
        section = doc["section"].split(" > ")[0]  # First part of section
        sections[section] = sections.get(section, 0) + 1
    
    print(f"\n📋 Content breakdown by section:")
    for section, count in sorted(sections.items()):
        print(f"   • {section}: {count} pages")
    
    print(f"\n🔧 To use this comprehensive data:")
    print(f"1. Update your .env file:")
    print(f"   SAMPLE_DATA_FILE=data/gitlab_comprehensive_crawl.json")
    print(f"2. Restart your GitPulseAI app")
    print(f"3. Ask about diversity initiatives - they should now be covered!")


if __name__ == "__main__":
    main() 