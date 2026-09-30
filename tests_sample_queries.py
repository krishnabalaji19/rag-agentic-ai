import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

from src.graph import build_rag_graph

QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?"
]

def run_tests():
    graph = build_rag_graph(index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index"))
    results = []

    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}/6] {q}")
        for attempt in range(3):
            try:
                state = graph.invoke({"question": q, "context": [], "answer": "", "score": 0.0})
                break
            except Exception as e:
                if attempt < 2:
                    print(f"Retrying in 20s due to: {str(e)[:100]}")
                    time.sleep(20)
                else:
                    state = {"answer": f"ERROR: {e}", "context": [], "score": 0.0}

        result = {
            "query": q,
            "final_answer": state["answer"],
            "retrieved_context_chunks": state["context"],
            "confidence_score": state["score"]
        }
        results.append(result)
        print(f"Answer: {result['final_answer'][:150]}...")
        print(f"Score: {result['confidence_score']}")
        time.sleep(15)

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll results saved to test_results.json")

if __name__ == "__main__":
    run_tests()
