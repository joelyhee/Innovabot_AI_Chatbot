# InnovaBot: Intelligent Conversational AI with RAG

Enterprise-grade chatbot leveraging Retrieval-Augmented Generation (RAG) and multi-agent architecture to deliver accurate, context-aware responses for EngagePro's AI solutions.

---

## 🎯 High-Level Overview

InnovaBot is an intelligent conversational AI system developed as part of the Generative AI Solutions module at Ngee Ann Polytechnic. The chatbot serves as a virtual assistant for EngagePro, a technology firm specializing in generative AI applications. The system combines advanced natural language processing with ethical guardrails to provide accurate information about EngagePro's products while maintaining robust safety standards.

### Core Capabilities

- **Domain-Specific Knowledge Retrieval**: Uses FAISS vector database to retrieve relevant information from EngagePro's company brochure with semantic search
- **General Knowledge Integration**: Leverages Wikipedia API for technical and general queries outside EngagePro's domain
- **Intelligent Query Routing**: LLM-based classification system determines optimal information source (company knowledge base vs. external sources)
- **Multi-Agent Architecture**: Employs CrewAI framework with specialized agents for product expertise and technical research
- **Comprehensive Guardrails**: Multi-layered safety system including PII detection, prompt injection prevention, content filtering, and nonsense detection

### System Architecture

```
User Query
    ↓
Guardrails Validation (PII, Prompt Injection, Content Safety, Nonsense)
    ↓
Query Routing (LLM-based classification)
    ↓
    ├─→ EngagePro Route → FAISS Vector Retrieval → Product Specialist Agent
    └─→ General Route → Wikipedia API → Technical Researcher Agent
    ↓
Response Generation (with source attribution)
    ↓
User Interface (Streamlit + Follow-up Questions)
```

---

## ✨ What Makes This Project Stand Out

### 1. Hybrid Knowledge Architecture

Unlike basic RAG implementations, InnovaBot employs a **dual-source knowledge system** with intelligent routing:

- **Company Knowledge Base**: FAISS vector store built from EngagePro brochure for domain-specific queries
- **External Knowledge**: Dynamic Wikipedia API integration for general technical queries  
- **Smart Routing**: LLM-powered intent classification ensures queries reach the optimal knowledge source

### 2. Production-Grade Ethical Safeguards

Implements **comprehensive safety measures** rarely seen in academic projects:

| Guardrail | Detection Method | Purpose |
|-----------|------------------|---------|
| **PII Detection** | Regex pattern matching | Prevents exposure of emails, phone numbers, credit cards, SSN/NRIC, bank accounts |
| **Prompt Injection Defense** | Keyword pattern analysis | Blocks attempts to manipulate system prompts ("ignore previous", "act as") |
| **Content Safety** | LLM-based classification | Identifies harmful, illegal, or inappropriate requests |
| **Nonsense Filtering** | Multi-strategy validation | Detects gibberish, keyboard mashing, repeated characters, empty inputs |

All regex-based guardrails operate **without external API dependencies**, ensuring low-latency validation.

### 3. Advanced Prompt Engineering

Employs **sophisticated prompt strategies** optimized for business context:

- **Role-Based System Prompts**: Detailed definitions with response examples and limitation handling
- **Few-Shot Routing**: Context-aware classification with demonstration examples
- **Structured Safety Prompts**: LLM-based content filtering with explicit output formatting
- **Dynamic Follow-Ups**: Contextual question generation for enhanced user engagement

### 4. Multi-Agent Orchestration

Utilizes **CrewAI framework** for specialized task delegation:

- **EngagePro Product Specialist**: Domain expert with deep knowledge of company products and services
- **Technical Information Researcher**: General knowledge specialist with Wikipedia integration
- **Non-Delegating Architecture**: Agents operate independently for deterministic, reliable responses

### 5. Enhanced User Experience

- **🏢 Source Transparency**: Every response includes attribution (Company Knowledge Base or Wikipedia)
- **💬 Suggested Follow-ups**: AI-generated contextually relevant questions after each response
- **🧠 Conversation Memory**: Streamlit session state maintains chat history for coherent multi-turn dialogues
- **🎨 Clean Interface**: Centered layout with minimal distractions for focused interactions

### 6. Robust Error Handling

- Graceful degradation when vector store loading fails
- Exception handling for Wikipedia API timeouts
- Fallback routing when LLM classification is ambiguous
- User-friendly error messages without exposing system internals

---

## 📁 Project Structure

```
innovabot/
│
├── main.py                          # Core Streamlit application with UI and logic
├── build_index.py                   # FAISS vector store builder from PDF documents
├── config.py                        # Configuration and model initialization
├── development.ipynb                # Experimentation notebook for prototyping
├── requirements.txt                 # Python dependencies specification
├── .env                             # Environment variables (API keys) - NOT committed
├── README.md                        # Project documentation (this file)
│
├── data/
│   ├── Company_Brochure.pdf         # Source document for RAG knowledge base
│   └── faiss_index/
│       ├── index.faiss              # FAISS vector embeddings database
│       └── index.pkl                # Document metadata and mappings
│
```

---

## 📄 File Descriptions

### **main.py**
**Purpose**: Core application logic and Streamlit user interface

**Key Components**:
- **Model Initialization**: Cached initialization of GPT-4o-mini and text-embedding-3-small models
- **Vector Store Loading**: FAISS index deserialization with comprehensive error handling
- **Wikipedia Integration**: API wrapper for external knowledge retrieval
- **System Prompt Engineering**: Comprehensive role definition with response examples and guidelines
- **Guardrails Suite**: 
  - PII detection with regex patterns for 6+ sensitive data categories
  - LLM-powered harmful content classifier
  - Prompt injection pattern matching for manipulation attempts
  - Multi-strategy nonsense detection with 7 validation checks
  - Orchestrated safety pipeline with early-exit optimization
- **RAG Retrieval**: FAISS similarity search with configurable distance threshold
- **Query Routing**: LLM-based intent classification with greeting detection fallback
- **Agent Definitions**: CrewAI agent initialization with specialized roles
- **Response Generation**: Crew task execution with source attribution and formatting
- **Streamlit UI**: Interactive chat interface with session state management

**Design Patterns**:
- Cached resource loading to prevent redundant model initialization
- Progressive guardrail validation with early exit on failures
- Threshold-based retrieval filtering (FAISS distance metric)
- Stateful conversation tracking via session state

---

### **build_index.py**
**Purpose**: PDF processing and FAISS vector database construction

**Key Components**:
- **Text Cleaning**: Regex-based normalization for malformed financial figures and formatting artifacts
  - Handles patterns like `35millionin2019to50million` → `$35 million in 2019 to $50 million`
- **Text Chunking**: Overlap-based splitting strategy
  - Default: 1000 characters per chunk with 100 character overlap
  - Maintains context continuity across chunk boundaries
- **PDF Extraction**: pdfplumber-based text extraction with metadata preservation
- **Embedding & Indexing**: OpenAI text-embedding-3-small + FAISS persistence

**Why This Matters**:
- Malformed PDF text can severely degrade RAG retrieval quality
- Overlapping chunks prevent critical information loss at boundaries
- Metadata preservation enables accurate source attribution in responses

---

**Design Rationale**:
- **Low Temperature (0.3)**: Balances creativity with factual accuracy for business queries
- **Alternative Embeddings**: Sentence-transformers enables offline operation without OpenAI API dependency

---

### **development.ipynb**
**Purpose**: Experimentation and prototyping workspace

**Expected Usage**:
- RAG pipeline testing and optimization
- Prompt engineering iterations and A/B testing
- Guardrail validation with edge case scenarios
- Agent behavior analysis and tuning

---

### **Company_Brochure.pdf**
**Purpose**: Primary source document for EngagePro knowledge base

**Contents**:
- Company vision, mission statement, and core values
- Product descriptions: InnovaBot (knowledge management chatbot), CX Transformer (customer service platform)
- Financial performance metrics: $35M (2019) → $50M (2024)
- Team size, location, and contact information

**Usage**: Parsed by `build_index.py` to create FAISS vector store for domain-specific RAG retrieval
---

## 🛠️ Technical Stack

### **Programming Language**
- **Python 3.8+**: Core application development and scripting

### **Large Language Models**
- **OpenAI GPT-4o-mini**: Primary LLM for reasoning tasks
  - Query intent classification and routing
  - Response generation with context awareness
  - Content safety classification
  - Follow-up question generation
  - **Temperature**: 0.3 (balanced creativity and factual accuracy)

- **OpenAI text-embedding-3-small**: Semantic vectorization
  - Document and query embedding generation
  - 1536-dimensional dense vectors
  - Optimized for similarity search tasks

### **Key Frameworks & Libraries**

#### **LLM Orchestration**
- **LangChain 1.2.0**: Abstraction layer for LLM workflows
  - `langchain-openai`: OpenAI model integration
  - `langchain-community`: Vector store and Wikipedia utilities
  - `langchain-core`: Document and message handling
  - `langchain-text-splitters`: Advanced chunking strategies

#### **Multi-Agent System**
- **CrewAI 1.8.1**: Agent orchestration and task delegation
  - Role-based agent specialization
  - Goal-oriented workflow management
  - Non-delegating architecture for deterministic outputs

#### **Vector Database**
- **FAISS-CPU 1.13.2**: Facebook AI Similarity Search
  - Efficient nearest neighbor retrieval at scale
  - CPU-optimized (no GPU dependency)
  - Supports cosine similarity and L2 distance metrics

#### **User Interface**
- **Streamlit 1.32.0**: Rapid web application development
  - Python-native UI components
  - Built-in session state for conversation memory
  - Real-time interactive chat interface

#### **Data Processing**
- **PDFPlumber 0.11.9**: Robust PDF text extraction
- **Pandas 2.2.2**: Data manipulation and analytics
- **NumPy 1.26.4**: Numerical operations for embeddings

#### **External Knowledge**
- **Wikipedia 1.4.0**: General knowledge API wrapper with rate limiting

#### **Utilities**
- **python-dotenv 1.1.1**: Secure environment variable management
- **Regular Expressions (re)**: Pattern matching for guardrails

### **Development Tools**
- **Jupyter Notebook**: Interactive experimentation and prototyping
- **Git**: Version control and collaboration

---

## Future Improvements

### **Proposed Future Enhancements**

1. **Multi-Document Ingestion**:
   - Support for product manuals, FAQs, case studies, technical documentation
   - Automatic re-indexing pipeline for document updates

2. **Conversational RAG**:
   - Implement chat history summarization
   - Context-aware retrieval with conversation memory

3. **Hybrid Search**:
   - Combine dense (FAISS) + sparse (BM25) retrieval
   - Re-ranking with cross-encoder models for improved accuracy

4. **User Feedback Loop**:
   - Thumbs up/down ratings for response quality
   - Active learning to fine-tune retrieval and generation

5. **Advanced Analytics Dashboard**:
   - Query pattern tracking and heatmaps
   - Response quality metrics (latency, retrieval accuracy)
   - User engagement analytics

6. **Voice Interface Integration**:
   - Speech-to-text for accessibility
   - Multi-modal interaction support

7. **Enhanced Guardrails**:
   - Multi-language PII detection
   - Adversarial prompt detection using ML models
   - Toxicity scoring with configurable thresholds

---

## 📝 License & Attribution

**Author**: Joely Lim Kei Cin (S10267773D)  
**Project**: GAIS Assignment 1 - Part 2: Chatbot Applications  
**Submission Date**: 1 February 2026  
**Academic Year**: 2025/26  
**Data Source**: EngagePro Company Brochure (fictional company created for educational purposes)
**Code License**: Educational use only. Not licensed for commercial deployment.
