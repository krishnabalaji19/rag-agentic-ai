import os
from dotenv import load_dotenv

load_dotenv()

from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def _extract_text(content):
    """Gemini sometimes returns a list of parts instead of a plain string."""
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


def build_rag_graph(index_name: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768
    )
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    def retrieve_node(state: AgentState):
        # similarity_search_with_score gives us both the chunk text AND
        # a similarity score (0 to 1) for how close it is to the question
        results = vectorstore.similarity_search_with_score(state["question"], k=3)
        context_texts = [doc.page_content for doc, score in results]
        similarity_scores = [score for doc, score in results]
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        return {"context": context_texts, "score": avg_similarity}

    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a helpful assistant that answers ONLY using the context below. Do not use any outside knowledge.

Guidelines:
- If the context contains information relevant to the question, answer using it, even if the context is a partial excerpt.
- Only say 'I cannot answer based on the provided document.' if the context is truly unrelated to the question.

Context:
{context_str}

Question: {state['question']}"""
        response = llm.invoke(prompt)
        answer_text = _extract_text(response.content)

        # If the model couldn't answer, the confidence score should reflect that,
        # regardless of how similar the retrieved chunks looked
        final_score = state["score"]
        if "cannot answer" in answer_text.lower():
            final_score = 0.0

        return {"answer": answer_text, "score": final_score}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
