#!/usr/bin/env python3
"""
Comprehensive GitLab Handbook Data Extraction Script

This script extracts content from ALL major sections of the GitLab handbook
to provide comprehensive coverage for the GitPulseAI chatbot.

Based on the full menu structure from https://handbook.gitlab.com/
"""

import requests
from bs4 import BeautifulSoup, Tag
import json
import time
import re
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveHandbookExtractor:
    def __init__(self):
        self.base_url = "https://handbook.gitlab.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.extracted_data = []
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common navigation elements
        text = re.sub(r'(Skip to content|Table of contents|Edit this page)', '', text, flags=re.IGNORECASE)
        
        # Remove excessive repeated characters
        text = re.sub(r'(\n\s*){3,}', '\n\n', text)
        
        return text
    
    def extract_page_content(self, url: str, title: str, category: str) -> Dict[str, Any] | None:
        """Extract content from a single handbook page."""
        try:
            logger.info(f"Extracting: {title} ({url})")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove navigation, footer, and other non-content elements
            for element in soup.find_all(['nav', 'footer', 'header', '.navbar', '.sidebar', '.breadcrumb']):
                element.decompose()
            
            # Find main content area
            content_selectors = [
                'main',
                '.content',
                '#content',
                '.main-content',
                'article',
                '.markdown-body',
                '.handbook-content'
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            # Fallback to body if no main content found
            if not content:
                content = soup.find('body')
            
            if not content:
                logger.warning(f"No content found for {url}")
                return None
            
            # Extract text content
            text_content = content.get_text(separator='\n', strip=True)
            text_content = self.clean_text(text_content)
            
            # Skip if content is too short (likely navigation or error page)
            if len(text_content) < 200:
                logger.warning(f"Content too short for {url}: {len(text_content)} chars")
                return None
            
            # Extract headings for better structure
            headings = []
            if isinstance(content, Tag):
                headings = [h.get_text(strip=True) for h in content.find_all(['h1', 'h2', 'h3'])]
            
            return {
                "document_id": f"gitlab-handbook-{hash(url) % 1000000}",
                "url": url,
                "title": title,
                "category": category,
                "content": text_content,
                "headings": headings,
                "word_count": len(text_content.split()),
                "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error extracting {url}: {e}")
            return None
    
    def get_handbook_sections(self) -> List[Dict[str, str]]:
        """
        Define all handbook sections based on the complete menu structure.
        Returns list of {url, title, category} dictionaries.
        """
        sections = [
            # Company Section
            {"url": "https://handbook.gitlab.com/handbook/company/", "title": "About GitLab", "category": "Company"},
            {"url": "https://handbook.gitlab.com/handbook/values/", "title": "GitLab Values", "category": "Company"},
            {"url": "https://handbook.gitlab.com/handbook/mission/", "title": "Mission", "category": "Company"},
            {"url": "https://handbook.gitlab.com/handbook/vision/", "title": "Vision", "category": "Company"},
            {"url": "https://handbook.gitlab.com/handbook/communication/", "title": "Communication", "category": "Company"},
            {"url": "https://handbook.gitlab.com/handbook/company/culture/", "title": "Culture", "category": "Company"},
            {"url": "https://handbook.gitlab.com/handbook/teamops/", "title": "TeamOps", "category": "Company"},
            {"url": "https://handbook.gitlab.com/handbook/ceo/", "title": "CEO Readme", "category": "Company"},
            {"url": "https://handbook.gitlab.com/handbook/e-group-weekly/", "title": "E-Group Weekly", "category": "Company"},
            {"url": "https://handbook.gitlab.com/handbook/environmental-social-governance/", "title": "Environmental, Social, and Governance", "category": "Company"},
            
            # Handbook Meta
            {"url": "https://handbook.gitlab.com/handbook/", "title": "About the Handbook", "category": "Handbook"},
            {"url": "https://handbook.gitlab.com/handbook/handbook-usage/", "title": "Handbook Usage", "category": "Handbook"},
            {"url": "https://handbook.gitlab.com/handbook/style-guide/", "title": "Handbook Style Guide", "category": "Handbook"},
            {"url": "https://handbook.gitlab.com/handbook/editing/", "title": "Editing the Handbook", "category": "Handbook"},
            
            # People Group
            {"url": "https://handbook.gitlab.com/handbook/people-group/anti-harassment/", "title": "Anti-Harassment Policy", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/people-group/global-volunteer-month/", "title": "Global Volunteer Month", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/hiring/", "title": "Hiring", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/people-group/inclusion-and-diversity/", "title": "Inclusion & Diversity", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/people-group/labor-and-employment-notices/", "title": "Labor and Employment Notices", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/leadership/", "title": "Leadership", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/people-group/learning-and-development/", "title": "Learning & Development", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/people-group/general-onboarding/", "title": "Onboarding", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/people-group/offboarding/", "title": "Offboarding", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/spending-company-money/", "title": "Spending Company Money", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/people-group/talent-assessment/", "title": "Talent Assessment", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/people-group/team-member-relations/", "title": "Team Member Relations Philosophy", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/total-rewards/", "title": "Total Rewards", "category": "People Group"},
            {"url": "https://handbook.gitlab.com/handbook/tools-and-tips/", "title": "Tools and Tips", "category": "People Group"},
            
            # Engineering
            {"url": "https://handbook.gitlab.com/handbook/engineering/", "title": "Engineering Overview", "category": "Engineering"},
            {"url": "https://handbook.gitlab.com/handbook/support/", "title": "Customer Support Department", "category": "Engineering"},
            {"url": "https://handbook.gitlab.com/handbook/engineering/development/", "title": "Development Department", "category": "Engineering"},
            {"url": "https://handbook.gitlab.com/handbook/engineering/infrastructure/", "title": "Infrastructure Department", "category": "Engineering"},
            {"url": "https://handbook.gitlab.com/handbook/engineering/developer-experience/", "title": "Developer Experience", "category": "Engineering"},
            {"url": "https://handbook.gitlab.com/handbook/engineering/engineering-productivity/", "title": "Engineering Productivity", "category": "Engineering"},
            {"url": "https://handbook.gitlab.com/handbook/engineering/security/", "title": "Security Practices", "category": "Engineering"},
            {"url": "https://handbook.gitlab.com/handbook/engineering/open-source/", "title": "Open Source", "category": "Engineering"},
            
            # Security
            {"url": "https://handbook.gitlab.com/handbook/security/", "title": "Security Overview", "category": "Security"},
            {"url": "https://handbook.gitlab.com/handbook/security/security-standards/", "title": "Security Standards", "category": "Security"},
            {"url": "https://handbook.gitlab.com/handbook/security/product-security/", "title": "Product Security", "category": "Security"},
            {"url": "https://handbook.gitlab.com/handbook/security/security-operations/", "title": "Security Operations", "category": "Security"},
            {"url": "https://handbook.gitlab.com/handbook/security/threat-management/", "title": "Threat Management", "category": "Security"},
            {"url": "https://handbook.gitlab.com/handbook/security/security-assurance/", "title": "Security Assurance", "category": "Security"},
            
            # Marketing
            {"url": "https://handbook.gitlab.com/handbook/marketing/", "title": "Marketing Overview", "category": "Marketing"},
            {"url": "https://handbook.gitlab.com/handbook/marketing/team-member-social-media-policy/", "title": "Team Member Social Media Policy", "category": "Marketing"},
            {"url": "https://handbook.gitlab.com/handbook/marketing/blog/", "title": "Blog", "category": "Marketing"},
            {"url": "https://handbook.gitlab.com/handbook/marketing/brand-and-product-marketing/", "title": "Brand and Product Marketing", "category": "Marketing"},
            {"url": "https://handbook.gitlab.com/handbook/marketing/marketing-operations/", "title": "Marketing Operations and Analytics", "category": "Marketing"},
            {"url": "https://handbook.gitlab.com/handbook/marketing/developer-relations/", "title": "Developer Relations", "category": "Marketing"},
            {"url": "https://handbook.gitlab.com/handbook/marketing/corporate-communications/", "title": "Corporate Communications", "category": "Marketing"},
            
            # Sales
            {"url": "https://handbook.gitlab.com/handbook/sales/", "title": "Sales Overview", "category": "Sales"},
            {"url": "https://handbook.gitlab.com/handbook/alliances/", "title": "Alliances", "category": "Sales"},
            {"url": "https://handbook.gitlab.com/handbook/sales/commercial/", "title": "Commercial", "category": "Sales"},
            {"url": "https://handbook.gitlab.com/handbook/customer-success/", "title": "Customer Success", "category": "Sales"},
            {"url": "https://handbook.gitlab.com/handbook/customer-success/csm/", "title": "Customer Success Management", "category": "Sales"},
            {"url": "https://handbook.gitlab.com/handbook/reseller-channels/", "title": "Reseller Channels", "category": "Sales"},
            {"url": "https://handbook.gitlab.com/handbook/sales/field-operations/", "title": "Field Operations", "category": "Sales"},
            {"url": "https://handbook.gitlab.com/handbook/solutions-architecture/", "title": "Solutions Architecture", "category": "Sales"},
            
            # Finance
            {"url": "https://handbook.gitlab.com/handbook/finance/", "title": "Finance Overview", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/finance/accounts-payable/", "title": "Accounts Payable", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/finance/accounts-receivable/", "title": "Accounts Receivable", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/business-technology/", "title": "Business Technology", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/finance/expenses/", "title": "Expenses", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/finance/financial-planning-and-analysis/", "title": "Financial Planning & Analysis", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/finance/payroll/", "title": "Payroll", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/finance/procurement/", "title": "Procurement", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/tax/", "title": "Tax", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/board-meetings/", "title": "Board meetings", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/internal-audit/", "title": "Internal Audit", "category": "Finance"},
            {"url": "https://handbook.gitlab.com/handbook/stock-options/", "title": "Equity Compensation", "category": "Finance"},
            
            # Product
            {"url": "https://handbook.gitlab.com/handbook/product/", "title": "Product Overview", "category": "Product"},
            {"url": "https://handbook.gitlab.com/releases/", "title": "Release posts", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/product/being-a-product-manager/", "title": "Being a Product Manager at GitLab", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/product/product-principles/", "title": "Product Principles", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/product/product-processes/", "title": "Product Processes", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/product/categories/", "title": "Product sections, stages, groups, and categories", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/product/product-development-flow/", "title": "Product Development Flow", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/engineering/workflow/", "title": "Product Development Timeline", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/product/product-analysis/", "title": "Data for Product Managers", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/company/pricing/", "title": "Product Pricing Model", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/acquisitions/", "title": "Corporate Development / Acquisitions", "category": "Product"},
            {"url": "https://handbook.gitlab.com/handbook/product/ux/", "title": "UX Department", "category": "Product"},
            
            # Legal and Corporate Affairs
            {"url": "https://handbook.gitlab.com/handbook/legal/", "title": "Legal Overview", "category": "Legal"},
            {"url": "https://handbook.gitlab.com/handbook/legal/commercial/", "title": "Commercial", "category": "Legal"},
            {"url": "https://handbook.gitlab.com/handbook/legal/corporate/", "title": "Corporate", "category": "Legal"},
            {"url": "https://handbook.gitlab.com/handbook/legal/employment-law/", "title": "Employment", "category": "Legal"},
            {"url": "https://handbook.gitlab.com/handbook/legal/ESG/", "title": "Environment, Social, and Governance (ESG)", "category": "Legal"},
            {"url": "https://handbook.gitlab.com/handbook/legal/privacy/", "title": "Privacy", "category": "Legal"},
            {"url": "https://handbook.gitlab.com/handbook/legal/risk-management-and-dispute-resolution/", "title": "Risk Management and Dispute Resolution", "category": "Legal"},
            {"url": "https://handbook.gitlab.com/handbook/legal/trade-compliance/", "title": "Trade Compliance", "category": "Legal"},
            
            # Direction (Strategy)
            {"url": "https://about.gitlab.com/direction/", "title": "GitLab Direction", "category": "Direction"},
        ]
        
        return sections
    
    def extract_all_sections(self):
        """Extract content from all handbook sections."""
        sections = self.get_handbook_sections()
        total_sections = len(sections)
        
        logger.info(f"Starting extraction of {total_sections} handbook sections...")
        
        for i, section in enumerate(sections, 1):
            logger.info(f"Progress: {i}/{total_sections} - {section['title']}")
            
            content = self.extract_page_content(
                section['url'], 
                section['title'], 
                section['category']
            )
            
            if content:
                self.extracted_data.append(content)
                logger.info(f"✅ Extracted: {content['title']} ({content['word_count']} words)")
            else:
                logger.warning(f"❌ Failed to extract: {section['title']}")
            
            # Rate limiting - be respectful to GitLab's servers
            time.sleep(1)
        
        logger.info(f"Extraction complete! Total documents: {len(self.extracted_data)}")
    
    def save_data(self, filename: str = "data/gitlab_complete_handbook.json"):
        """Save extracted data to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.extracted_data, f, indent=2, ensure_ascii=False)
            
            # Calculate statistics
            total_words = sum(doc.get('word_count', 0) for doc in self.extracted_data)
            categories = {}
            for doc in self.extracted_data:
                cat = doc.get('category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            logger.info(f"✅ Data saved to {filename}")
            logger.info(f"📊 Statistics:")
            logger.info(f"  - Total documents: {len(self.extracted_data)}")
            logger.info(f"  - Total words: {total_words:,}")
            logger.info(f"  - Categories: {dict(categories)}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")

def main():
    """Main extraction function."""
    extractor = ComprehensiveHandbookExtractor()
    
    print("🔍 GitPulseAI - Complete GitLab Handbook Extraction")
    print("=" * 60)
    print("This will extract content from ALL major GitLab handbook sections")
    print("to provide comprehensive coverage for the chatbot.")
    print()
    
    confirm = input("Proceed with extraction? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Extraction cancelled.")
        return
    
    try:
        extractor.extract_all_sections()
        extractor.save_data()
        
        print("\n🎉 Complete handbook extraction finished successfully!")
        print("📁 Data saved to: data/gitlab_complete_handbook.json")
        print("\nNext steps:")
        print("1. Review the extracted data")
        print("2. Update your RAG system to use this comprehensive dataset")
        print("3. Test the chatbot with queries about various GitLab handbook sections")
        
    except KeyboardInterrupt:
        print("\n⚠️  Extraction interrupted by user")
        if extractor.extracted_data:
            print(f"💾 Saving partial data ({len(extractor.extracted_data)} documents)...")
            extractor.save_data("data/gitlab_partial_handbook.json")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    main() 