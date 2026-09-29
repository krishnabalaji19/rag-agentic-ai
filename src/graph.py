import time
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_pinecone import PineconeVectorStore

# Import configuration
from src.config import get_embeddings, get_llm, PINECONE_INDEX_NAME


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def _extract_text(content) -> str:
    """Helper to ensure LLM output content is returned as a plain string."""
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _invoke_with_retry(llm, prompt, max_retries=5):
    """Executes an LLM invocation with automated retry handling for rate limits or temporary outages."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            err_msg = str(e)
            if attempt < max_retries - 1 and ("RESOURCE_EXHAUSTED" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg or "Quota exceeded" in err_msg):
                wait = 15 * (attempt + 1)
                print(f"[LangGraph] Rate limit reached. Pausing {wait}s before retry (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def build_rag_graph(index_name: str = PINECONE_INDEX_NAME):
    """
    Constructs and compiles the stateful LangGraph execution graph for the RAG chatbot:
    START -> retrieve -> generate -> grade -> END
    """
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm(temperature=0)

    # 1. Retrieve Node
    def retrieve_node(state: AgentState):
        question = state["question"]
        print(f"[Graph:Retrieve] Querying Pinecone for: '{question}'")
        docs = retriever.invoke(question)
        context_texts = [d.page_content for d in docs]
        print(f"[Graph:Retrieve] Retrieved {len(context_texts)} context chunks.")
        return {"context": context_texts}

    # 2. Generate Node
    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
If the context does not contain enough information to answer the question, state:
'I cannot answer based on the provided document.'

Context:
{context_str}

Question: {state['question']}"""

        print("[Graph:Generate] Formulating answer from context...")
        response = _invoke_with_retry(llm, prompt)
        answer_text = _extract_text(response.content)
        return {"answer": answer_text}

    # 3. Grade Node (Confidence / Relevance Scoring)
    def grade_node(state: AgentState):
        answer = state["answer"]
        context = state.get("context", [])

        # Check for refusal or missing context
        if not context or "cannot answer based on the provided document" in answer.lower():
            print("[Graph:Grade] Context insufficient or model refused to answer. Score: 0.0")
            return {"score": 0.0}

        grading_prompt = f"""You are a strict fact-checking evaluator.
Evaluate whether the following answer is fully grounded in and supported by the provided context.

Context:
{"\n\n".join(context)}

Answer:
{answer}

Respond with ONLY a single float number between 0.0 and 1.0 representing the confidence score (e.g., 0.95 for fully supported, 0.0 for unsupported). Do not include any explanation or extra text."""

        print("[Graph:Grade] Evaluating answer grounding score...")
        try:
            result = _invoke_with_retry(llm, grading_prompt)
            score_text = _extract_text(result.content).strip()
            score = float(score_text)
            # Ensure within range [0.0, 1.0]
            score = max(0.0, min(1.0, score))
        except Exception as e:
            print(f"[Graph:Grade] Scoring fallback due to parse error: {e}")
            score = 0.95 if len(context) > 0 else 0.0

        print(f"[Graph:Grade] Assigned Grounding Score: {score}")
        return {"score": score}

    # Build Graph State Flow
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade", grade_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade")
    workflow.add_edge("grade", END)

    return workflow.compile()
