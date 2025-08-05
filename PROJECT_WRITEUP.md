# GitPulseAI - GitLab Handbook Chatbot

## Project Overview

GitPulseAI is an interactive AI chatbot that provides easy access to GitLab's Handbook and Direction pages. The project enables GitLab employees and aspiring employees to easily access and learn about GitLab's culture, processes, and policies through a conversational interface. By combining modern RAG (Retrieval-Augmented Generation) architecture with a clean, user-friendly interface, GitPulseAI makes knowledge discovery engaging and efficient.

## Technical Approach

### Architecture

The project employs a RAG (Retrieval-Augmented Generation) system that combines document retrieval with LLM generation. This architecture was chosen to ensure accurate, up-to-date responses grounded in GitLab's actual documentation, rather than relying solely on an LLM's pre-trained knowledge which might be outdated or inaccurate.

### Core Components

1. **Data Processing Pipeline**:
   - Web scraping modules for GitLab Handbook and Direction pages
   - Comprehensive crawler that follows all internal links for complete coverage
   - Text extraction and cleaning for optimal search quality
   - Document structuring with title, content, URL, and section metadata

2. **Vector Storage Options**:
   - **Local Mode**: In-memory vector store using Hugging Face sentence-transformers
   - **Cloud Mode**: Supabase PostgreSQL with pgvector extension for production deployment
   - TF-IDF fallback mechanism for reliability when semantic search fails

3. **LLM Integration**:
   - Google Gemini integration for both text generation and embeddings
   - System prompts designed for conversational, natural responses
   - Context window optimization to maximize relevant information

4. **User Interface**:
   - Streamlit-based chat interface with native chat components
   - Categorized example questions for easy discovery
   - Source attribution (optional) for transparency
   - Response feedback mechanisms (👍/👎)

## Key Technical Decisions

### 1. RAG Architecture Choice

We implemented a hybrid RAG system that combines:
- Semantic vector search using embeddings
- TF-IDF keyword search as a fallback mechanism
- LLM for synthesizing final responses from retrieved content

This approach balances accuracy with response quality, ensuring the system only provides information it can find in the source documents.

### 2. Technology Stack Selection

- **Frontend**: Streamlit for its simplicity, Python-native approach, and rapid development capabilities
- **LLM**: Google Gemini for its balance of quality, cost-efficiency, and generous free tier
- **Embeddings**: 
  - Local mode: Hugging Face sentence-transformers (offline capability)
  - Cloud mode: Google Gemini embeddings (API-based)
- **Vector Storage**:
  - Development: Simple in-memory vector store for quick iteration
  - Production: Supabase with pgvector for scalability and managed infrastructure

### 3. Deployment Flexibility

The architecture supports two deployment modes:

**Local Development Mode**:
- Uses Hugging Face embeddings that run offline
- Simple in-memory vector store
- Works without cloud dependencies

**Cloud Production Mode**:
- Supabase for PostgreSQL + pgvector storage
- Google Gemini API for embeddings and LLM
- Optimized for Streamlit Cloud deployment

### 4. Data Collection Approach

The initial approach used targeted scraping of specific GitLab pages. However, we found coverage gaps (like missing diversity initiative pages), which led to developing a comprehensive crawler that:

- Starts from the main GitLab Handbook and Direction pages
- Follows every valid internal link recursively
- Filters by domain to stay within GitLab content
- Processes and cleans all discovered content

This ensures complete coverage of GitLab's public documentation, preventing knowledge gaps in the chatbot's responses.

## User Experience Considerations

### 1. Conversational Design

The system is designed to provide natural, conversational responses that sound like they're coming from a helpful GitLab team member rather than formal documentation. This includes:

- Warm, friendly tone that aligns with GitLab's culture
- Removing technical metadata and source citations from responses
- Structuring information in a digestible, conversational format

### 2. Accessibility Features

- Clear categorization of example questions
- Simple, intuitive chat interface
- Response feedback mechanisms
- Copy functionality for easy sharing

### 3. Performance Optimizations

- Query caching for repeated questions
- Optimized context window usage
- Fallback mechanisms when semantic search fails
- Comprehensive error handling with user-friendly messages

## Implementation Challenges and Solutions

### 1. Knowledge Coverage Gaps

**Challenge**: Initial data scraping missed important sections of the GitLab documentation.

**Solution**: Developed a comprehensive crawler that recursively follows all internal links from the main handbook and direction pages, ensuring complete coverage of GitLab's public documentation.

### 2. Response Quality and Citations

**Challenge**: Balancing conversational tone with source attribution.

**Solution**: Implemented a configurable citation system where source links can be displayed separately in the sidebar rather than cluttering the main response. Also added a system prompt that encourages natural, helpful responses.

### 3. Deployment Flexibility

**Challenge**: Supporting both local development and cloud production environments.

**Solution**: Created a modular architecture with interchangeable components for vector storage and embeddings, allowing seamless switching between local and cloud modes via environment variables.

## Technical Architecture Diagram

```
┌───────────────────────────────┐      ┌───────────────────────────┐
│                               │      │                           │
│  Data Collection              │      │  Vector Storage           │
│  ───────────────              │      │  ────────────             │
│  - Web scraping               │      │  - Hugging Face           │
│  - Text extraction            │──────▶  - Supabase pgvector      │
│  - Document structuring       │      │  - TF-IDF fallback        │
│  - Metadata enrichment        │      │                           │
│                               │      └───────────┬───────────────┘
└───────────────────────────────┘                  │
                                                   │
                                                   ▼
┌───────────────────────────────┐      ┌───────────────────────────┐
│                               │      │                           │
│  User Interface               │      │  RAG System               │
│  ─────────────                │      │  ─────────                │
│  - Streamlit chat             │◀─────▶  - Document retrieval     │
│  - Example questions          │      │  - Context formatting     │
│  - Response feedback          │      │  - Google Gemini LLM      │
│  - Citation display           │      │  - Response generation    │
│                               │      │                           │
└───────────────────────────────┘      └───────────────────────────┘
```

## Future Enhancements

1. **Advanced Search Capabilities**:
   - Multi-query reformulation for better semantic matching
   - Query expansion with relevant keywords
   - Learning from user feedback to improve search quality

2. **Enhanced Context Processing**:
   - Dynamic chunk sizing based on content importance
   - Hierarchical document retrieval for better context
   - Entity recognition for improved context selection

3. **UI/UX Improvements**:
   - Custom themes aligned with GitLab branding
   - Mobile optimization for on-the-go access
   - Saved conversations and history browsing

4. **Integration Possibilities**:
   - Slack/Discord bot integration
   - API endpoint for headless usage
   - Integration with GitLab's existing systems

## Conclusion

GitPulseAI successfully delivers on its core promise of making GitLab's extensive documentation accessible and engaging through conversational AI. The project demonstrates the power of combining modern RAG architecture with thoughtful user experience design to create an effective knowledge assistant.

By balancing technical sophistication with practical usability, GitPulseAI provides a valuable tool for GitLab employees and prospective team members to navigate the company's rich documentation and culture in a natural, conversational way.