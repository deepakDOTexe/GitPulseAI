# GitPulseAI - GitLab Handbook Assistant 🔍💬

A modern Streamlit-based AI chatbot that helps users navigate GitLab's handbook and policies through conversational AI using Google Gemini and RAG (Retrieval-Augmented Generation) architecture.

## 🚀 Features

- **Conversational AI**: Natural language interaction with GitLab documentation
- **Dual Deployment Modes**: Local development or cloud deployment with Supabase
- **Google Gemini Integration**: Powered by Google's Gemini LLM and embeddings
- **Clean Native UI**: Built with Streamlit's native chat components
- **Source Citations**: Responses include references to original GitLab pages
- **Hybrid Search**: Vector similarity with keyword fallback
- **Modern Architecture**: OpenAI-free, cost-effective solution

## 🛠️ Quick Setup

### **Prerequisites**
- Python 3.8+
- Google Gemini API key (free)
- Supabase account (for cloud deployment)

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/GitPulseAI.git
   cd GitPulseAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .envexample .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## ⚙️ Configuration

### **Local Development (.env)**
```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (for local mode)
SAMPLE_DATA_FILE=data/gitlab_comprehensive_handbook.json
USE_SUPABASE=false
```

### **Cloud Deployment (.env)**
```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here
USE_SUPABASE=true
```

## 🌩️ Cloud Deployment (Supabase)

For production deployment with Supabase vector database:

1. **Create Supabase Project**
   - Go to [supabase.com](https://supabase.com) and create a new project

2. **Set up Database**
   ```sql
   -- Run in Supabase SQL Editor
   -- See sql/supabase_setup.sql for complete setup
   ```

3. **Migrate Data**
   ```bash
   python scripts/migrate_to_supabase.py
   ```

4. **Deploy to Streamlit Cloud**
   - Connect your GitHub repository
   - Set environment variables
   - Deploy!

## 📁 Project Structure

```
GitPulseAI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies
├── .envexample                     # Environment template
├── src/                           # Core modules
│   ├── supabase_rag_system.py    # Cloud RAG system
│   ├── hybrid_rag_system.py      # Local RAG system
│   ├── gemini_llm.py             # Google Gemini integration
│   └── ...
├── data/                          # GitLab handbook data
├── scripts/                       # Utility scripts
├── docs/                         # Documentation
└── sql/                          # Database setup
```

## 🎯 Usage

1. **Start the application**: `streamlit run app.py`
2. **Choose example questions** from the sidebar or type your own
3. **Ask about GitLab**: Values, policies, remote work, culture, etc.
4. **Get sourced responses** with references to original documentation

## 🔧 Development

### **Adding New Data**
```bash
# Update GitLab handbook data
python scripts/extract_comprehensive_handbook.py

# For cloud deployment, migrate to Supabase
python scripts/migrate_to_supabase.py
```

### **Local Development**
- Uses Hugging Face embeddings (offline)
- TF-IDF fallback for reliability
- Google Gemini for response generation

### **Cloud Deployment**
- Supabase PostgreSQL + pgvector
- Google Gemini embeddings and LLM
- Optimized for Streamlit Cloud

## 📊 Architecture

- **Local Mode**: HuggingFace embeddings + TF-IDF fallback + Gemini LLM
- **Cloud Mode**: Supabase vector DB + Gemini embeddings + Gemini LLM
- **Frontend**: Streamlit with native chat components
- **Data**: GitLab handbook pages in structured JSON format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GitLab for their commitment to transparency and open handbook
- Google for Gemini API and generous free tier
- Supabase for excellent vector database capabilities
- Streamlit for the amazing web app framework

## 📞 Support

If you encounter any issues:

1. Check the existing [Issues](https://github.com/YourUsername/GitPulseAI/issues)
2. Create a new issue with detailed information
3. Include your deployment mode (local/cloud) and error messages

---

**Built with ❤️ using Google Gemini, Supabase, and Streamlit** 