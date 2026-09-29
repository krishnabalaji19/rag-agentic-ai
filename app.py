import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME

app = FastAPI(
    title="Agentic AI RAG API",
    description="API for Retrieval-Augmented Generation (RAG) chatbot grounded in the Agentic AI eBook.",
    version="1.0.0"
)

# Lazy initialization of the LangGraph execution workflow
graph = None


def get_graph():
    global graph
    if graph is None:
        index_name = os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)
        graph = build_rag_graph(index_name=index_name)
    return graph


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to be answered based on the Agentic AI eBook.")


class QueryResponse(BaseModel):
    final_answer: str = Field(..., description="Generated answer strictly grounded in the document context.")
    retrieved_context: list[str] = Field(..., description="Text chunks retrieved from Pinecone used to answer the query.")
    confidence_score: float = Field(..., description="Grounded confidence or relevance score (0.0 to 1.0).")


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Agentic AI RAG API",
        "index_name": PINECONE_INDEX_NAME,
        "docs_url": "/docs"
    }


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        execution_graph = get_graph()
        initial_state = {
            "question": request.query,
            "context": [],
            "answer": "",
            "score": 0.0
        }

        result = execution_graph.invoke(initial_state)

        return QueryResponse(
            final_answer=result.get("answer", ""),
            retrieved_context=result.get("context", []),
            confidence_score=result.get("score", 0.0)
        )
    except Exception as e:
        print(f"[API Error] Exception during /chat invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
