# GitLab Handbook Assistant - Setup Guide

## Overview

This GitLab Handbook Assistant uses a **hybrid RAG architecture** combining:
- **Hugging Face sentence-transformers** for embeddings (free, local)
- **TF-IDF fallback** for when Hugging Face models can't be downloaded
- **Google Gemini** for LLM generation (generous free tier)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your FREE Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

**Free Tier Benefits:**
- 15 requests per minute
- 1 million tokens per day
- Much more generous than OpenAI's paid-only model

### 3. Configure Environment

```bash
# Copy the environment template
cp .envexample .env

# Edit .env file and add your API key
# GEMINI_API_KEY=your_api_key_here
```

### 4. Test the System

```bash
# Test the core functionality
python quick_test.py

# Test the full system (requires API key)
python test_hybrid_system.py
```

### 5. Run the Application

```bash
streamlit run app.py
```

## 🔧 Architecture Details

### Embedding Strategy (Free & Local)

The system uses a **two-tier approach**:

1. **Primary: Hugging Face Embeddings**
   - Model: `all-MiniLM-L6-v2` (384 dimensions)
   - Completely free and runs locally
   - High-quality semantic embeddings

2. **Fallback: TF-IDF Embeddings**
   - Uses scikit-learn's TF-IDF vectorizer
   - Works completely offline
   - Activated when Hugging Face models fail to download

### LLM Strategy (Generous Free Tier)

- **Google Gemini 1.5 Flash**
- 15 requests per minute
- 1 million tokens per day
- Built-in rate limiting
- Automatic error handling

## 🛠️ Troubleshooting

### SSL Certificate Issues (Common on macOS)

If you see SSL certificate errors during model download:

```bash
# Option 1: Upgrade certificates
pip install --upgrade certifi

# Option 2: Run the macOS certificate installer
/Applications/Python\ 3.x/Install\ Certificates.command

# Option 3: Use environment variables
export CURL_CA_BUNDLE=''
python test_hybrid_system.py
```

### System Will Automatically Fallback

If Hugging Face models fail to download, the system will:
1. Show SSL error messages with troubleshooting steps
2. Automatically switch to TF-IDF fallback mode
3. Continue working with offline embeddings

### No API Key Required for Testing

You can test the core functionality without any API keys:

```bash
python quick_test.py
```

## 📊 System Capabilities

### Document Processing
- Loads 12 sample GitLab documents
- Chunks text for optimal retrieval
- Extracts keywords and metadata

### Search Methods
- **Semantic Search**: Using embeddings for meaning-based matching
- **TF-IDF Search**: Statistical text similarity
- **Keyword Search**: Fallback pattern matching

### Response Generation
- Context-aware responses using retrieved documents
- Source citations and relevance scores
- Conversation history support

## 🎯 Usage Examples

### Basic Queries
```
"What are GitLab's core values?"
"How does GitLab handle remote work?"
"What is GitLab's engineering process?"
"How does GitLab approach diversity and inclusion?"
```

### Advanced Features
- **Source Citations**: See which documents were used
- **Relevance Scores**: Understand result quality
- **Conversation History**: Multi-turn conversations
- **Health Monitoring**: System status and statistics

## 📁 File Structure

```
GitPulseAI/
├── src/
│   ├── hybrid_rag_system.py     # Main RAG system
│   ├── simple_vector_store.py   # TF-IDF fallback
│   ├── local_vector_store.py    # Hugging Face embeddings
│   ├── gemini_llm.py           # Google Gemini interface
│   ├── data_processor.py       # Document processing
│   └── config.py               # Configuration
├── data/
│   └── sample_gitlab_data.json # Sample documents
├── app.py                      # Streamlit interface
├── test_hybrid_system.py       # Full system test
├── quick_test.py              # Core functionality test
└── requirements.txt           # Dependencies
```

## 🔒 Cost & Privacy

### Completely Free Core
- Document processing: 100% free
- TF-IDF embeddings: 100% free
- Hugging Face embeddings: 100% free

### Generous Free Tier
- Google Gemini: 1M tokens/day free
- No credit card required
- Much more generous than OpenAI

### Privacy-Friendly
- Embeddings computed locally
- Only queries sent to Gemini
- No personal data stored

## 🚨 Important Notes

### SSL Certificate Issues
- Common on macOS systems
- System automatically falls back to TF-IDF
- Multiple troubleshooting options provided

### Rate Limiting
- Built-in rate limiting for Gemini API
- Automatic retry with exponential backoff
- Clear error messages and guidance

### System Health
- Real-time health monitoring
- Comprehensive system statistics
- Fallback mode indicators

## 🆘 Support

If you encounter issues:

1. **Check the logs** in `logs/app.log`
2. **Run the test scripts** to isolate the problem
3. **Check system health** in the Streamlit sidebar
4. **Review the troubleshooting section** above

The system is designed to be **robust and fault-tolerant**, with multiple fallback mechanisms to ensure it works even when external services are unavailable.

## 🎉 Success Indicators

You'll know the system is working when:

✅ **Dependencies installed** without errors  
✅ **API key configured** (for full functionality)  
✅ **Test scripts pass** (quick_test.py for core, test_hybrid_system.py for full)  
✅ **Streamlit app starts** without errors  
✅ **Queries return relevant results** with source citations  

The system is designed to **gracefully degrade** and provide useful functionality even when some components are unavailable. 