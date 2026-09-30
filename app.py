import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG Chatbot")

# Build the graph once when the server starts, not on every request
graph = build_rag_graph(index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index"))


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0
    }
    result = graph.invoke(initial_state)

    return QueryResponse(
        query=request.query,
        final_answer=result["answer"],
        retrieved_context_chunks=result["context"],
        confidence_score=result["score"]
    )


@app.get("/")
async def root():
    return {"status": "RAG chatbot API is running", "endpoint": "POST /chat with {'query': 'your question'}"}