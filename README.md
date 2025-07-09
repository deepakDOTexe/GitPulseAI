# GitPulseAI - GitLab Handbook Assistant

A Streamlit-based AI chatbot that helps users navigate GitLab's handbook and direction pages through conversational AI using RAG (Retrieval-Augmented Generation) architecture.

## 🚀 Features

- **Conversational AI**: Natural language interaction with GitLab documentation
- **Source Citations**: Responses include references to original GitLab pages
- **Follow-up Questions**: Intelligent suggestions for deeper exploration
- **Accessibility**: Built with accessibility features and keyboard navigation
- **Error Handling**: Robust error handling for smooth user experience
- **Clean UI**: Modern, intuitive interface with GitLab branding

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/GitPulseAI.git
   cd GitPulseAI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create a .env file in the project root
   touch .env
   
   # Add your OpenAI API key
   echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
   ```

## ⚙️ Configuration

The application uses environment variables for configuration. Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
APP_NAME=GitLab Handbook Assistant
APP_VERSION=1.0.0
DEBUG=False

# OpenAI Configuration
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=1000

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# Vector Search Configuration
SIMILARITY_THRESHOLD=0.7
MAX_SEARCH_RESULTS=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/chatbot.log
```

## 🏃‍♂️ Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the interface**
   - Open your browser and go to `http://localhost:8501`
   - Start chatting with the GitLab Handbook Assistant

3. **Example questions to try**
   - "What are GitLab's core values?"
   - "How does GitLab handle remote work?"
   - "What is the GitLab review process?"
   - "How do I contribute to GitLab?"

## 🏗️ Project Structure

```
GitPulseAI/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── PROJECT_WRITEUP.md    # Project overview
├── TECHNICAL_APPROACH.md # Technical documentation
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── rag_system.py     # RAG system implementation (coming soon)
│   └── data_processor.py # Data processing utilities (coming soon)
├── data/                 # Sample data files (coming soon)
├── logs/                 # Application logs
└── tests/                # Test files (coming soon)
```

## 🔧 Development Status

This is a 1-week MVP implementation. Current status:

- ✅ **Day 1-2**: Project setup and basic structure
- ⏳ **Day 3-4**: Core RAG system implementation
- ⏳ **Day 5-6**: Streamlit interface development
- ⏳ **Day 7**: Testing and polish

## 🎯 Features Implemented

### Current (Day 1-2)
- [x] Project structure and dependencies
- [x] Basic Streamlit interface with placeholder responses
- [x] Error handling and input validation
- [x] Accessibility features (high contrast, keyboard navigation)
- [x] GitLab branding and UI design
- [x] Session state management
- [x] Configuration management

### Coming Soon (Day 3-7)
- [ ] RAG system with OpenAI embeddings
- [ ] Sample GitLab documentation data
- [ ] Vector similarity search
- [ ] Real AI responses with source citations
- [ ] Comprehensive testing
- [ ] Performance optimization

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
- OpenAI for providing powerful AI capabilities
- Streamlit for the excellent web app framework
- The open-source community for inspiration and tools

## 📞 Support

If you encounter any issues or have questions:

1. Check the existing [Issues](https://github.com/YourUsername/GitPulseAI/issues)
2. Create a new issue with detailed information
3. Include logs from `logs/chatbot.log` if applicable

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your `.env` file contains a valid OpenAI API key
   - Check that the key has sufficient credits

2. **Dependencies Not Found**
   - Ensure you're in the correct virtual environment
   - Run `pip install -r requirements.txt` again

3. **Port Already in Use**
   - Use a different port: `streamlit run app.py --server.port 8502`

4. **Logs Not Creating**
   - Ensure the `logs/` directory exists
   - Check file permissions

### Performance Tips

- Use environment variables to configure OpenAI model (GPT-3.5-turbo is faster and cheaper than GPT-4)
- Adjust `OPENAI_MAX_TOKENS` based on your needs
- Monitor API usage in your OpenAI dashboard

---

**Built with ❤️ by the GitPulseAI Team** 