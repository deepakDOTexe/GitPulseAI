# Deploying GitPulseAI to Streamlit Community Cloud

This guide explains how to deploy your GitPulseAI to **Streamlit Community Cloud** with optimized architecture for fast startup and reliable performance.

## 🎯 Overview

**Streamlit Community Cloud** has specific constraints that require architectural optimization:
- **⏰ Cold start timeouts**: Apps must start in <60 seconds
- **💾 Ephemeral file system**: No persistent storage between sessions
- **🧠 Memory limits**: ~1GB RAM limit
- **🔄 Frequent restarts**: Apps sleep and restart regularly

## 🔧 Architectural Changes for Cloud Deployment

### **Before (Local Development)**
```
Local App Startup:
1. Download sentence-transformers model (100s MB) ⏰ ~60-120 seconds
2. Load GitLab data from JSON ⏰ ~10 seconds  
3. Generate embeddings for all documents ⏰ ~60-180 seconds
Total: ~2-5 minutes startup time ❌
```

### **After (Cloud Optimized)**
```
Cloud App Startup:
1. Load precomputed embeddings from JSON ⏰ ~5-10 seconds
2. Initialize cached components ⏰ ~2-5 seconds
Total: ~10-15 seconds startup time ✅
```

## 📋 Step-by-Step Deployment Process

### **Step 1: Generate Precomputed Embeddings**

**On your local machine** (do this before deploying):

```bash
# Install sentence-transformers (only needed locally)
pip install sentence-transformers

# Generate precomputed embeddings
python scripts/precompute_embeddings.py
```

This creates `data/precomputed_gitlab_comprehensive_handbook.json` with:
- All document embeddings precomputed
- No model downloads required
- Fast JSON loading

### **Step 2: Update App for Cloud Mode**

Update your `app.py` to detect cloud deployment:

```python
import os
import streamlit as st

# Detect deployment environment
IS_CLOUD_DEPLOYMENT = os.getenv("STREAMLIT_SHARING_MODE") == "true" or "streamlit.app" in str(st._get_option("server.baseUrlPath", ""))

if IS_CLOUD_DEPLOYMENT:
    # Use cloud-optimized RAG system
    from src.cloud_hybrid_rag import create_cached_cloud_rag_system
    
    @st.cache_resource
    def initialize_rag_system():
        return create_cached_cloud_rag_system(
            precomputed_data_file="data/precomputed_gitlab_comprehensive_handbook.json"
        )
else:
    # Use regular RAG system for local development
    from src.hybrid_rag_system import HybridRAGSystem
    
    @st.cache_resource  
    def initialize_rag_system():
        return HybridRAGSystem()
```

### **Step 3: Create Cloud Configuration Files**

**`.streamlit/config.toml`** (Streamlit Cloud configuration):
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FC6D26"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

**`requirements.txt`** (Use cloud-optimized version):
```bash
# Copy cloud requirements
cp requirements-cloud.txt requirements.txt
```

### **Step 4: Set Environment Variables**

In your **Streamlit Cloud app settings**, add:

```bash
GEMINI_API_KEY=your_actual_gemini_api_key_here
SAMPLE_DATA_FILE=data/precomputed_gitlab_comprehensive_handbook.json
SIMILARITY_THRESHOLD=0.3
MAX_SEARCH_RESULTS=5
LOG_LEVEL=INFO
```

### **Step 5: Repository Structure for Deployment**

Ensure your repository has this structure:

```
GitPulseAI/
├── app.py                                    # Main Streamlit app
├── requirements.txt                          # Cloud dependencies (lightweight)
├── .streamlit/
│   └── config.toml                          # Streamlit Cloud config
├── src/
│   ├── cloud_hybrid_rag.py                 # Cloud-optimized RAG
│   ├── cloud_vector_store.py               # Precomputed embeddings
│   ├── gemini_llm.py                       # Gemini LLM
│   └── config.py                           # Configuration
├── data/
│   └── precomputed_gitlab_comprehensive_handbook.json  # IMPORTANT: Include this!
└── docs/
    └── streamlit_cloud_deployment.md       # This guide
```

**⚠️ Critical**: Include `data/precomputed_gitlab_comprehensive_handbook.json` in your repository!

### **Step 6: Deploy to Streamlit Cloud**

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add cloud deployment optimization"
   git push origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Connect your GitHub repository**

4. **Configure deployment**:
   - Repository: `your-username/GitPulseAI`
   - Branch: `main`
   - Main file path: `app.py`

5. **Add secrets** (Environment Variables):
   - `GEMINI_API_KEY`: Your Google Gemini API key

6. **Deploy!** 🚀

## 📊 Performance Comparison

| Metric | Local Development | Cloud Optimized |
|--------|-------------------|-----------------|
| **Startup Time** | 2-5 minutes | 10-15 seconds |
| **Memory Usage** | 500MB-1GB | 200-400MB |
| **Dependencies** | 15+ packages | 8 packages |
| **Model Downloads** | 100s of MB | None |
| **Cold Start Success** | ❌ Timeout | ✅ Fast |

## 🔍 Monitoring & Debugging

### **Check App Health**

Add health check to your app:

```python
# In your Streamlit app
if st.button("🏥 System Health Check"):
    if st.session_state.get('rag_system'):
        health = st.session_state.rag_system.health_check()
        if health['status'] == 'healthy':
            st.success("✅ System is healthy")
        else:
            st.error("❌ System has issues")
            for check in health['checks']:
                if check['status'] == 'failed':
                    st.error(f"❌ {check['name']}: {check['message']}")
```

### **View Logs**

In Streamlit Cloud, check the app logs for:
- Initialization time
- Memory usage warnings
- API call errors
- Cache hit/miss ratios

### **Common Issues & Solutions**

**🚨 App Timeout on Startup**
```
Solution: Ensure precomputed embeddings are included in repository
Check: data/precomputed_gitlab_comprehensive_handbook.json exists
```

**🚨 Memory Exceeded**
```
Solution: Use requirements-cloud.txt (fewer dependencies)
Optimize: Use float32 instead of float64 for embeddings
```

**🚨 Gemini API Errors**
```
Solution: Check GEMINI_API_KEY is set correctly in Streamlit secrets
Verify: API key has sufficient quota
```

**🚨 Missing Dependencies**
```
Solution: Use requirements-cloud.txt
Remove: sentence-transformers, beautifulsoup4, lxml from cloud deployment
```

## ⚡ Optimization Tips

### **1. Minimize Dependencies**
```bash
# Use requirements-cloud.txt instead of requirements.txt
# Remove development-only packages:
# - sentence-transformers (only for precomputation)
# - beautifulsoup4 (only for scraping)
# - jupyter (only for development)
```

### **2. Optimize Data Loading**
```python
# Use Streamlit caching extensively
@st.cache_data
def load_precomputed_data(file_path):
    # This loads only once and caches the result
    return json.load(open(file_path))

@st.cache_resource
def initialize_system():
    # This creates the system only once
    return CloudHybridRAGSystem()
```

### **3. Memory Optimization**
```python
# Use float32 instead of float64 for embeddings
embeddings = np.array(embeddings, dtype=np.float32)

# Limit context window size
max_context_chars = 4000  # Instead of 8000+

# Clear unused variables
del large_temporary_variables
```

### **4. Error Handling**
```python
try:
    # Initialize system
    system = initialize_system()
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.info("This might be due to missing precomputed data or API issues.")
    st.stop()
```

## 🔄 Updating Deployed App

### **Adding New Content**
1. **Update local data** (run scraper locally)
2. **Regenerate embeddings**: `python scripts/precompute_embeddings.py`
3. **Commit and push**: New embeddings file will be deployed
4. **App restarts automatically** with new content

### **Configuration Changes**
- Update environment variables in Streamlit Cloud settings
- Changes take effect on next app restart

## 🎯 Production Checklist

- [ ] ✅ Generated precomputed embeddings locally
- [ ] ✅ Included `data/precomputed_*.json` in repository  
- [ ] ✅ Using `requirements-cloud.txt` for dependencies
- [ ] ✅ Added `GEMINI_API_KEY` to Streamlit secrets
- [ ] ✅ Created `.streamlit/config.toml` configuration
- [ ] ✅ App starts in <60 seconds
- [ ] ✅ Memory usage <500MB
- [ ] ✅ Health check passes
- [ ] ✅ Sample queries work correctly
- [ ] ✅ Error handling covers API failures
- [ ] ✅ Logs show successful initialization

## 🚀 Launch!

Your GitPulseAI is now optimized for **Streamlit Community Cloud**:

- ⚡ **Fast startup**: 10-15 seconds vs 2-5 minutes
- 🧠 **Memory efficient**: <400MB vs >1GB  
- 📱 **Reliable**: Handles cold starts and restarts
- 🌐 **Accessible**: Public URL for sharing
- 🔒 **Secure**: API keys managed by Streamlit secrets

**Your comprehensive GitLab Handbook AI Assistant is now live and ready to help users explore GitLab's culture, processes, and policies!** 🎉 