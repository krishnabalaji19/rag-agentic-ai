import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys & Credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# PDF Source Path
DEFAULT_PDF_PATH = os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf")


def get_embeddings():
    """Returns the configured embedding model (Google Generative AI or OpenAI)."""
    if GOOGLE_API_KEY:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            output_dimensionality=768
        )
    elif OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError("Neither GOOGLE_API_KEY nor OPENAI_API_KEY found in environment variables.")


def get_llm(temperature: float = 0.0):
    """Returns the configured Chat LLM model (Google Gemini or OpenAI)."""
    if GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=temperature
        )
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature
        )
    else:
        raise ValueError("Neither GOOGLE_API_KEY nor OPENAI_API_KEY found in environment variables.")
