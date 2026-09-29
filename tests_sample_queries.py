import sys
import time
import json
import requests
from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME

SAMPLE_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?"  # Validation test (should refuse/state missing context)
]


def test_via_direct_graph():
    """Runs benchmark queries directly against compiled LangGraph execution graph."""
    print("=" * 80)
    print("RUNNING BENCHMARK QUERIES (DIRECT LANGGRAPH INVOCATION)")
    print("=" * 80)

    graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)

    for idx, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n--- Query {idx}/{len(SAMPLE_QUERIES)} ---")
        print(f"QUESTION: {query}\n")

        initial_state = {
            "question": query,
            "context": [],
            "answer": "",
            "score": 0.0
        }

        result = graph.invoke(initial_state)

        print(f"ANSWER:\n{result['answer']}\n")
        print(f"CONFIDENCE SCORE: {result['score']}")
        print(f"RETRIEVED CONTEXT CHUNKS: {len(result['context'])}")
        for i, chunk in enumerate(result['context'], 1):
            snippet = chunk.replace('\n', ' ')[:120]
            print(f"  [Chunk {i}]: {snippet}...")
        print("-" * 80)
        time.sleep(5)


def test_via_api(api_url: str = "http://127.0.0.1:8000/chat"):
    """Runs benchmark queries against running FastAPI /chat endpoint."""
    print("=" * 80)
    print(f"RUNNING BENCHMARK QUERIES (FASTAPI REST ENDPOINT: {api_url})")
    print("=" * 80)

    for idx, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n--- Query {idx}/{len(SAMPLE_QUERIES)} ---")
        print(f"QUESTION: {query}\n")

        response = requests.post(api_url, json={"query": query})
        if response.status_code == 200:
            data = response.json()
            print(f"ANSWER:\n{data['final_answer']}\n")
            print(f"CONFIDENCE SCORE: {data['confidence_score']}")
            print(f"RETRIEVED CONTEXT CHUNKS: {len(data['retrieved_context'])}")
        else:
            print(f"API Error ({response.status_code}): {response.text}")
        print("-" * 80)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "direct"
    if mode == "api":
        url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/chat"
        test_via_api(url)
    else:
        test_via_direct_graph()
