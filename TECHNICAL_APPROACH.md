# GitPulseAI - Technical Approach & Architecture

## Project Architecture Overview

**System Type**: RAG (Retrieval-Augmented Generation) Chatbot  
**Primary Goal**: Interactive access to GitLab's Handbook and Direction pages  
**Architecture Pattern**: Microservices with API-first design

## Core Technical Requirements

### 1. Data Collection & Processing Pipeline

**Data Sources**:
- GitLab Handbook (https://handbook.gitlab.com/)
- GitLab Direction pages (https://about.gitlab.com/direction/)
- Public documentation repositories

**Implementation Components**:
- **Web Scraping**: Beautiful Soup + Selenium for dynamic content
- **Data Cleaning**: Regular expressions, HTML parsing, text normalization
- **Chunking Strategy**: 
  - Semantic chunking (512-1024 tokens per chunk)
  - Overlap of 50-100 tokens between chunks
  - Preserve document structure and context
- **Metadata Extraction**: 
  - Page titles, section headers
  - Source URLs, last modified dates
  - Document hierarchy and relationships

**Technical Stack**:
```python
# Core libraries
requests, beautifulsoup4, selenium
pandas, numpy
langchain-text-splitters
```

### 2. Vector Database & Embeddings

**What Are Embedding Models**:
Embedding models convert text into numerical vectors (arrays of numbers) that capture semantic meaning. They enable semantic search by placing similar concepts close together in high-dimensional vector space.

**Why We Need Embeddings**:
- **Semantic Search**: Find documents by meaning, not just exact keywords
- **Context Understanding**: Distinguish between different meanings of same words
- **Cross-language Support**: Handle multilingual content effectively
- **Robust Matching**: Work with typos, abbreviations, different phrasings

**Embedding Model Options**:
- **OpenAI text-embedding-ada-002**: High quality, 1536 dimensions, $0.0001/1K tokens
- **Sentence-BERT (all-MiniLM-L6-v2)**: Open source, 384 dimensions, fast inference
- **Cohere Embed**: Commercial alternative with multilingual support
- **E5 Models**: State-of-the-art open source options

**Vector Database Options**:
- **Pinecone**: Managed, scalable, good for production
- **Chroma**: Open source, lightweight, good for development
- **Weaviate**: Feature-rich, supports hybrid search
- **FAISS**: Facebook's library, good for local deployment

**Implementation Details**:
- **Similarity Search**: Cosine similarity for finding relevant documents
- **Batch Processing**: Process multiple texts efficiently
- **Index Refresh**: Strategy for updated GitLab content
- **Metadata Filtering**: Filter results by source, date, section
- **Hybrid Search**: Combine semantic + keyword search for better accuracy

**Example Implementation**:
```python
# OpenAI embeddings
from openai import OpenAI
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Similarity search
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Process GitLab content
def process_gitlab_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk['content'])
        chunk['embedding'] = embedding
        embeddings.append(chunk)
    return embeddings
```

**Benefits for GitLab Chatbot**:
- **Better Question Understanding**: "How do I get promoted?" → Finds performance reviews
- **Cross-Reference Discovery**: Hiring questions surface onboarding, culture content
- **Robust Search**: Works with typos, abbreviations, different phrasings

### 3. LLM Integration

**Model Options**:

**Commercial APIs**:
- **OpenAI GPT-4**: Best quality, higher cost
- **OpenAI GPT-3.5-turbo**: Good balance of quality/cost
- **Anthropic Claude**: Alternative with good reasoning
- **Cohere Command**: Cost-effective option

**Open Source Models**:
- **Llama 2 (7B/13B)**: Meta's open model
- **Mistral 7B**: Efficient European model
- **CodeLlama**: Specialized for technical content

**Prompt Engineering Strategy**:
```
System: You are a GitLab assistant helping users understand GitLab's handbook and direction. 
Context: {retrieved_documents}
User Query: {user_question}
Guidelines: Be accurate, cite sources, admit uncertainty when unsure.
```

### 4. RAG System Architecture

**Query Processing Flow**:
1. User input → Query understanding
2. Query embedding → Vector similarity search
3. Document retrieval → Context assembly
4. LLM generation → Response formatting
5. Response → User interface

**Key Components**:
- **Query Preprocessor**: Intent recognition, keyword extraction
- **Retrieval Engine**: Multi-stage retrieval with reranking
- **Context Manager**: Document relevance scoring, context window management
- **Response Generator**: LLM integration with safety checks

### 5. Backend API Architecture

**Framework**: FastAPI (Python) for async support and automatic docs

**Core Endpoints**:
```python
POST /api/chat
- Input: user_message, conversation_id, user_id
- Output: response, sources, confidence_score

GET /api/search
- Input: query, filters, limit
- Output: relevant_documents, metadata

POST /api/feedback
- Input: message_id, rating, feedback_text
- Output: acknowledgment

GET /api/health
- Output: system_status, database_health, model_availability
```

**Additional Features**:
- **Authentication**: JWT tokens for user sessions
- **Rate Limiting**: Redis-based throttling
- **Caching**: Response caching for common queries
- **Monitoring**: Structured logging, metrics collection

## Updated Section 6: Frontend Interface

Here's the replacement for the current "### 6. Frontend Interface Options" section:

```markdown
### 6. Frontend Interface - Streamlit Implementation

**Framework Decision**: Streamlit (Selected for rapid development and Python-native approach)

**Key Requirements**:
- **Clear Response Display**: Responses displayed with proper formatting and source citations
- **Seamless Follow-up Questions**: Persistent chat history enabling natural conversation flow
- **Basic Error Handling**: Graceful handling of API failures, timeouts, and user input errors
- **Simple, Intuitive Layout**: Accessible design following WCAG guidelines

**Core Features**:

**a. Clear Response Display**:
- **Structured Messages**: User and assistant messages clearly differentiated
- **Source Citations**: Expandable sections showing GitLab page references
- **Formatted Output**: Markdown rendering for better readability
- **Timestamp Display**: Message timing for context

**b. Seamless Follow-up Questions**:
- **Persistent Chat History**: Session-based conversation memory
- **Context Preservation**: Previous questions influence current responses
- **Quick Action Buttons**: Suggested follow-up questions
- **Conversation Threading**: Maintain context across multiple exchanges

**c. Basic Error Handling**:
- **API Timeout Handling**: Graceful degradation when backend is slow
- **Network Error Recovery**: Retry mechanisms with user feedback
- **Input Validation**: Prevent empty queries and malformed requests
- **Rate Limit Management**: Queue requests and inform users of delays
- **Fallback Responses**: Generic help when RAG system fails

**d. Simple, Intuitive Layout**:
- **Clean Interface**: Minimal distractions, focus on conversation
- **Accessibility Features**: Keyboard navigation, screen reader support
- **Responsive Design**: Works on various screen sizes
- **Clear Visual Hierarchy**: Important information prominently displayed
- **Loading Indicators**: User feedback during processing

**Implementation Architecture**:
```python
# Core Streamlit Structure
import streamlit as st
import requests
import time
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="GitLab Handbook Assistant",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Session state management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = generate_conversation_id()
```

**Error Handling Implementation**:
```python
def handle_api_call(user_message: str) -> Dict:
    """Handle API calls with comprehensive error management"""
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://localhost:8000/api/chat",
                json={
                    "message": user_message,
                    "conversation_id": st.session_state.conversation_id,
                    "user_id": st.session_state.get("user_id", "anonymous")
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return {"error": "timeout"}
    
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection error. Please check your internet connection.")
        return {"error": "connection"}
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.error("⏸️ Too many requests. Please wait a moment and try again.")
        else:
            st.error(f"❌ Server error: {e.response.status_code}")
        return {"error": "http_error"}
    
    except Exception as e:
        st.error("❌ An unexpected error occurred. Please try again.")
        st.expander("Error Details").write(str(e))
        return {"error": "unexpected"}
```

**Accessibility Features**:
```python
def render_accessible_chat():
    """Render chat interface with accessibility considerations"""
    
    # Screen reader friendly title
    st.markdown("# GitLab Handbook Assistant")
    st.markdown("*Your AI-powered guide to GitLab's culture, processes, and policies*")
    
    # Skip to main content link
    st.markdown('<a href="#chat-input" class="sr-only">Skip to chat input</a>', 
                unsafe_allow_html=True)
    
    # High contrast mode toggle
    with st.sidebar:
        high_contrast = st.checkbox("High Contrast Mode", key="accessibility_contrast")
    
    # Keyboard navigation hints
    st.markdown("**💡 Tip**: Use Tab to navigate, Enter to send messages")
    
    # Main chat area with ARIA labels
    st.markdown('<div role="main" aria-label="Chat conversation">', 
                unsafe_allow_html=True)
```

**User Experience Enhancements**:
```python
def render_enhanced_chat_interface():
    """Enhanced chat interface with UX improvements"""
    
    # Welcome message for new users
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown("""
            👋 Hello! I'm your GitLab Handbook Assistant. I can help you with:
            - GitLab's culture and values
            - Company policies and procedures
            - Career development guidance
            - Technical processes and workflows
            
            What would you like to know about GitLab?
            """)
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message.get("sources"):
                with st.expander("📚 Sources & References"):
                    for source in message["sources"]:
                        st.markdown(f"- [{source['title']}]({source['url']})")
                        if source.get("relevance_score"):
                            st.caption(f"Relevance: {source['relevance_score']:.2f}")
    
    # Suggested follow-up questions
    if st.session_state.messages:
        last_response = st.session_state.messages[-1]
        if last_response.get("suggested_questions"):
            st.markdown("### 💭 You might also ask:")
            cols = st.columns(min(len(last_response["suggested_questions"]), 3))
            for i, question in enumerate(last_response["suggested_questions"]):
                with cols[i % 3]:
                    if st.button(question, key=f"suggest_{i}"):
                        process_user_input(question)
```

**Input Validation and Safety**:
```python
def validate_user_input(user_input: str) -> bool:
    """Validate and sanitize user input"""
    
    # Check for empty input
    if not user_input.strip():
        st.warning("⚠️ Please enter a question or message.")
        return False
    
    # Check message length
    if len(user_input) > 1000:
        st.warning("⚠️ Message too long. Please keep it under 1000 characters.")
        return False
    
    # Basic content filtering
    prohibited_patterns = ["<script>", "javascript:", "eval("]
    for pattern in prohibited_patterns:
        if pattern.lower() in user_input.lower():
            st.error("❌ Invalid input detected. Please rephrase your question.")
            return False
    
    return True
```

**Tech Stack**:
```python
# Core dependencies
streamlit>=1.28.0
requests>=2.31.0
python-dotenv>=1.0.0
openai>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# Optional enhancements
streamlit-chat>=0.1.0
streamlit-feedback>=0.1.0
streamlit-analytics>=0.4.0
```

**Deployment Configuration**:
```python
# streamlit_config.toml
[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
showErrorDetails = false

[theme]
primaryColor = "#FC6D26"  # GitLab orange
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F0F0"
textColor = "#262626"
```

**Performance Optimizations**:
- **Caching**: `@st.cache_data` for expensive operations
- **Lazy Loading**: Load conversation history on demand
- **Debouncing**: Prevent rapid-fire API calls
- **Connection Pooling**: Reuse HTTP connections
- **Response Streaming**: Show responses as they're generated

**Analytics and Monitoring**:
- **User Interaction Tracking**: Button clicks, query patterns
- **Error Rate Monitoring**: Track and alert on failures
- **Performance Metrics**: Response times, user satisfaction
- **Usage Analytics**: Popular queries, peak usage times


**Cloud Architecture**:
- **Application**: Container orchestration (Kubernetes/Docker Swarm)
- **Database**: Managed PostgreSQL for user data
- **Vector DB**: Managed Pinecone or self-hosted Chroma
- **Cache**: Redis for session management
- **CDN**: CloudFlare for static assets

**Monitoring & Observability**:
- **Logging**: Structured JSON logs
- **Metrics**: Prometheus + Grafana
- **Tracing**: OpenTelemetry for request tracking
- **Alerting**: Critical system failures and performance issues

## Implementation Phases - 1 Week Sprint

### **1-Week Development Plan** (Focused on MVP Gen AI Chatbot)

**Day 1-2: Foundation Setup**
- Project structure and dependencies
- Sample GitLab documentation data
- Basic environment configuration
- OpenAI API integration setup

**Day 3-4: Core RAG System**
- Simple vector database (in-memory/local)
- Basic embedding and similarity search
- LLM integration with OpenAI
- Simple RAG pipeline implementation

**Day 5-6: Streamlit Interface**
- Chat interface with error handling
- Accessibility features implementation
- Source citation system
- Follow-up question functionality

**Day 7: Testing & Polish**
- End-to-end testing
- Code documentation cleanup
- Performance optimization
- Deployment preparation

### **Daily Deliverables**

**Day 1**: 
- ✅ Project setup with requirements.txt
- ✅ Sample GitLab data files
- ✅ Environment configuration (.env)

**Day 2**:
- ✅ OpenAI API integration
- ✅ Basic document processing
- ✅ Simple vector storage

**Day 3**:
- ✅ Embedding generation pipeline
- ✅ Similarity search functionality
- ✅ Basic RAG query processing

**Day 4**:
- ✅ LLM response generation
- ✅ Source citation extraction
- ✅ Context management

**Day 5**:
- ✅ Streamlit chat interface
- ✅ Error handling implementation
- ✅ Input validation

**Day 6**:
- ✅ Accessibility features
- ✅ Follow-up questions
- ✅ UI/UX improvements

**Day 7**:
- ✅ Testing and bug fixes
- ✅ Documentation cleanup
- ✅ Performance optimization

### **Technical Scope (1-Week Version)**

**Included Features**:
- Basic chat interface with Streamlit
- OpenAI GPT-3.5-turbo for responses
- Simple vector search with sample data
- Error handling and input validation
- Source citations and references
- Follow-up question suggestions
- Accessibility features

**Excluded Features (Future Phases)**:
- Real-time GitLab content scraping
- Production vector database
- Advanced authentication
- Complex deployment architecture
- Comprehensive monitoring

### **Success Criteria (1-Week MVP)**

**Functional Requirements**:
- Users can ask questions about GitLab
- System provides relevant responses with sources
- Chat interface works smoothly
- Error handling prevents crashes
- Code is clean and well-documented

**Technical Requirements**:
- Response time < 10 seconds
- 95% uptime during demo
- Clean, readable code with comments
- Basic accessibility compliance
- Simple deployment instructions
