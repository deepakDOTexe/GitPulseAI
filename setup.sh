#!/bin/bash

# GitPulseAI - Setup Script
# This script helps you quickly set up the GitLab Handbook Assistant

echo "🦊 GitPulseAI - GitLab Handbook Assistant Setup"
echo "================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Application Configuration
APP_NAME=GitLab Handbook Assistant
APP_VERSION=1.0.0
DEBUG=False

# OpenAI Model Configuration
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
EOF
    echo "✅ .env file created"
    echo "⚠️  Please edit .env file and add your OpenAI API key"
else
    echo "⚠️  .env file already exists"
fi

# Create logs directory
mkdir -p logs

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Run: source venv/bin/activate"
echo "3. Run: streamlit run app.py"
echo "4. Open http://localhost:8501 in your browser"
echo ""
echo "For help, see README.md" 