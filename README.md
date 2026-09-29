# RAG-based AI Chatbot Project

A Retrieval-Augmented Generation (RAG) AI Chatbot built using **Python**, **LangGraph**, **Pinecone**, and **FastAPI**. The system strictly answers user questions based on the knowledge provided in the **Agentic AI eBook**.

---

## 🌟 Architecture & Workflow

The core RAG engine is orchestrated as a stateful graph using **LangGraph**:

```
                       ┌────────────────┐
                       │    [START]     │
                       └───────┬────────┘
                               │
                               ▼
                       ┌────────────────┐
                       │  retrieve_node │ (Queries Pinecone for top-k text chunks)
                       └───────┬────────┘
                               │
                               ▼
                       ┌────────────────┐
                       │  generate_node │ (Strict system prompt LLM generation)
                       └───────┬────────┘
                               │
                               ▼
                       ┌────────────────┐
                       │   grade_node   │ (Evaluates grounded confidence score 0.0-1.0)
                       └───────┬────────┘
                               │
                               ▼
                       ┌────────────────┐
                       │     [END]      │
                       └────────────────┘
```

---

## 📁 Directory Structure

```text
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf        # Downloaded source document
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central environment & model configuration
│   ├── ingestion.py                # PDF parsing, chunking & Pinecone index upload
│   └── graph.py                    # Stateful LangGraph workflow definition
├── app.py                          # FastAPI application interface (/chat endpoint)
├── tests_sample_queries.py         # Script executing benchmark queries
├── requirements.txt                # Python project dependencies
├── .env.example                    # Environment variable template
├── .env                            # Active environment configuration
└── README.md                       # Comprehensive project guide
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

1. **Clone the repository** (or navigate to your project directory):
   ```bash
   cd rag-agentic-ai
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file (refer to `.env.example`):
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_or_openai_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=agentic-ai-index
   ```

---

### 2. PDF Document Ingestion Pipeline

Ensure `data/Ebook-Agentic-AI.pdf` exists, then execute the ETL pipeline to parse the PDF, generate embeddings, and populate your Pinecone vector index:

```bash
python -m src.ingestion
```

---

### 3. Launch the FastAPI Web Server

Start the FastAPI backend server using Uvicorn:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Once running, access the interactive API documentation at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

### 4. Run Benchmark Test Queries

To evaluate the system quality against required sample queries (including ground-truth questions and an out-of-domain validation query), run:

**Direct Graph Invocation**:
```bash
python tests_sample_queries.py direct
```

**REST API Invocation (with uvicorn server running)**:
```bash
python tests_sample_queries.py api
```

---

## 📡 API Reference

### `POST /chat`

**Request Payload**:
```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

**Response Payload**:
```json
{
  "final_answer": "Agentic AI refers to artificial intelligence systems capable of autonomous planning, reasoning, and execution...",
  "retrieved_context": [
    "Text snippet from page 3 of the PDF...",
    "Text snippet from page 4 of the PDF..."
  ],
  "confidence_score": 0.95
}
```

---

## 🛡️ Grounding & Out-of-Domain Guardrails

If a user asks a question that is **not present** in the document context (e.g., *"Who won the 2022 FIFA World Cup?"*), the system:
1. Detects that the retrieved context is insufficient or ungrounded.
2. Refuses to hallucinate and responds with:  
   `"I cannot answer based on the provided document."`
3. Returns a confidence score of `0.0`.
