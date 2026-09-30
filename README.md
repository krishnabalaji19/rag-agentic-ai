# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly based on the provided "Agentic AI" eBook, built with LangGraph, Pinecone, and Google's Gemini API.

## Architecture

User Question -> Retrieve Node (searches Pinecone for top 3 relevant chunks, returns similarity scores) -> Generate Node (sends question + chunks to Gemini, answers ONLY from that context) -> Final Answer + Retrieved Chunks + Confidence Score

The confidence score is derived from the average cosine similarity of the retrieved chunks from Pinecone, rather than a second LLM call. This keeps the pipeline efficient while still reflecting how well-grounded the retrieval was. If the model determines it cannot answer from the context, the score is overridden to 0.0.

## Tech Stack
- LangGraph: orchestrates the retrieve -> generate workflow as a stateful graph
- Pinecone: vector database (serverless, dense, cosine similarity, 768 dimensions)
- Google Gemini: gemini-embedding-001 for embeddings, gemini-3.8-flash for generation
- FastAPI: serves the /chat endpoint
- LangChain: PDF loading and text splitting utilities

Note: The original reference material suggested OpenAI. This implementation uses Google's Gemini API instead, since it offers a genuinely free tier suitable for this project without requiring billing setup.

## Setup

### 1. Clone and create a virtual environment
git clone your-repo-url
cd rag-agentic-ai
python3 -m venv venv
source venv/bin/activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Set up API keys
Copy .env.example to .env and fill in your own keys.
Google API key: from Google AI Studio (aistudio.google.com)
Pinecone API key: from Pinecone (app.pinecone.io), free tier

### 4. Create your Pinecone index
In the Pinecone dashboard, create an index named agentic-ai-index with dimension 768, metric cosine, type Dense.

### 5. Download the eBook
Download the PDF from the provided Google Drive link and save it as data/Ebook-Agentic-AI.pdf

### 6. Run ingestion (one-time step)
python src/ingestion.py
This loads the PDF, splits it into chunks (1000 characters, 100 overlap), embeds them, and stores them in Pinecone.

### 7. Start the API
uvicorn app:app --reload
Visit http://127.0.0.1:8000/docs for interactive API testing.

## Usage

POST /chat
Request: {"query": "What is Agentic AI?"}
Response: {"query": "...", "final_answer": "...", "retrieved_context_chunks": ["...", "...", "..."], "confidence_score": 0.83}

## Testing
tests_sample_queries.py runs 6 validation queries (5 grounded + 1 out-of-scope) and saves results to test_results.json:
python tests_sample_queries.py

## Known Limitations
Running on Gemini's free tier, which has a modest daily request quota. In production, this would use a paid tier or self-hosted models for higher throughput. The confidence score approximates groundedness via retrieval similarity rather than a dedicated hallucination-detection model, as a deliberate efficiency trade-off.