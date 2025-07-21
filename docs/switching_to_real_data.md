# Switching from Sample to Real GitLab Data

This guide explains how to switch your GitPulseAI from sample data to real GitLab handbook content.

## 🎯 Overview

Currently, GitPulseAI uses a small sample of GitLab handbook content (~12 documents). To make it truly comprehensive, you can switch to real GitLab data with hundreds of handbook pages.

## 📋 Required JSON Format

Your data file must follow this structure:

```json
{
    "documents": [
        {
            "id": "unique-document-id",
            "title": "Page Title", 
            "url": "https://about.gitlab.com/handbook/...",
            "section": "Category/Section",
            "content": "Full text content of the page",
            "keywords": ["keyword1", "keyword2", "..."]
        }
    ],
    "metadata": {
        "scraped_at": "2024-01-20 15:30:00",
        "total_documents": 150,
        "source": "GitLab Handbook"
    }
}
```

## 🚀 Option 1: Quick Configuration Switch

### Step 1: Update Environment Configuration

Edit your `.env` file:

```bash
# Change this line:
SAMPLE_DATA_FILE=data/sample_gitlab_data.json

# To this (when you have real data):
SAMPLE_DATA_FILE=data/real_gitlab_data.json
```

### Step 2: Restart Your App

```bash
streamlit run app.py
```

## 🤖 Option 2: Automated Scraping (Recommended)

### Step 1: Install Additional Dependencies

```bash
pip install beautifulsoup4 lxml
```

Or update your requirements:

```bash
pip install -r requirements.txt
```

### Step 2: Run the GitLab Scraper

```bash
# Scrape 30 pages (good starting point)
python scripts/scrape_gitlab_data.py

# For more comprehensive data, edit the script to increase max_pages
```

### Step 3: Update Configuration

The scraper automatically creates `data/real_gitlab_data.json`. Update your `.env`:

```bash
SAMPLE_DATA_FILE=data/real_gitlab_data.json
```

### Step 4: Restart and Test

```bash
streamlit run app.py
```

## 🛠️ Option 3: GitLab API (Advanced)

### Prerequisites

- GitLab API access token
- Understanding of GitLab's API structure

### Implementation

Create `scripts/gitlab_api_scraper.py`:

```python
import requests
import json
from pathlib import Path

class GitLabAPIClient:
    def __init__(self, access_token=None):
        self.base_url = "https://gitlab.com/api/v4"
        self.headers = {}
        if access_token:
            self.headers["Authorization"] = f"Bearer {access_token}"
    
    def get_handbook_project(self):
        # GitLab's handbook is in a public repository
        # You can adapt this to access specific projects
        pass
    
    def fetch_wiki_pages(self, project_id):
        # Fetch wiki pages from a GitLab project
        pass
```

## 🌐 Option 4: Manual Data Collection

### Step 1: Create Data Template

```json
{
    "documents": [],
    "metadata": {
        "created_at": "2024-01-20",
        "source": "Manual Collection",
        "total_documents": 0
    }
}
```

### Step 2: Add Documents Manually

For each GitLab handbook page you want to include:

1. **Visit the page** (e.g., https://about.gitlab.com/handbook/values/)
2. **Copy the content** 
3. **Add to your JSON** following the format above
4. **Generate keywords** based on the content

### Step 3: Validate Your Data

```python
# Quick validation script
import json

def validate_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    required_keys = ["id", "title", "url", "section", "content", "keywords"]
    
    for doc in data["documents"]:
        for key in required_keys:
            if key not in doc:
                print(f"Missing key '{key}' in document {doc.get('id', 'unknown')}")
    
    print(f"Validation complete. {len(data['documents'])} documents checked.")

# Run validation
validate_data("data/your_data.json")
```

## 📊 Scaling Considerations

### Small Scale (30-50 documents)
- **Use**: Automated scraper with default settings
- **Benefits**: Quick setup, covers major topics
- **Limitations**: May miss specialized content

### Medium Scale (100-200 documents)
- **Use**: Scraper with increased `max_pages`
- **Benefits**: Comprehensive coverage
- **Considerations**: Longer initial setup time

### Large Scale (500+ documents)
- **Use**: GitLab API + custom processing
- **Benefits**: Complete handbook coverage
- **Requirements**: More complex setup, API limits

## ⚡ Performance Impact

### Embedding Generation Time
- **Sample data (12 docs)**: ~30 seconds
- **Real data (50 docs)**: ~2-3 minutes  
- **Real data (200 docs)**: ~8-10 minutes

### Search Performance
- **Sample data**: Instant
- **Real data (50 docs)**: <1 second
- **Real data (200+ docs)**: 1-2 seconds

### Memory Usage
- **Sample data**: ~50MB
- **Real data (50 docs)**: ~200MB
- **Real data (200+ docs)**: ~500MB+

## 🔧 Troubleshooting

### Common Issues

**1. "No documents found"**
```bash
# Check file path
ls -la data/real_gitlab_data.json

# Verify JSON format
python -m json.tool data/real_gitlab_data.json
```

**2. "Permission denied" during scraping**
```bash
# Add delays between requests
# Check GitLab's robots.txt
# Use respectful scraping practices
```

**3. "Out of memory" with large datasets**
```bash
# Reduce chunk_size in text_splitter
# Process in batches
# Use more efficient embedding models
```

### Data Quality Tips

**1. Content Filtering**
- Remove navigation elements
- Filter out non-content sections  
- Clean up formatting artifacts

**2. Keyword Optimization**
- Use GitLab-specific terminology
- Include section-based keywords
- Add synonyms and variations

**3. Section Organization**
- Use consistent section naming
- Group related content
- Maintain hierarchy structure

## 🎯 Recommended Approach

### For Getting Started
1. **Use the automated scraper** with 30-50 pages
2. **Test thoroughly** with your common questions
3. **Gradually expand** based on usage patterns

### For Production
1. **Start with comprehensive scraping** (100+ pages)
2. **Monitor search quality** and user feedback
3. **Implement regular updates** to keep data fresh
4. **Consider GitLab API integration** for automatic updates

## 📈 Success Metrics

After switching to real data, you should see:

- ✅ **Higher answer accuracy** for GitLab-specific questions
- ✅ **More relevant source citations** 
- ✅ **Better coverage** of technical topics
- ✅ **Improved user satisfaction** with responses

## 🔄 Keeping Data Updated

### Manual Updates
- Re-run scraper monthly
- Monitor GitLab changelog for new content
- Update high-priority sections more frequently

### Automated Updates (Advanced)
- Set up scheduled scraping
- Use GitLab webhooks for content changes
- Implement differential updates for efficiency

## 🎉 Next Steps

1. **Choose your approach** based on your needs
2. **Run the migration** following the steps above
3. **Test thoroughly** with real questions
4. **Monitor performance** and adjust as needed
5. **Consider user feedback** for content priorities

Your GitPulseAI will become significantly more powerful with real GitLab data! 🚀 