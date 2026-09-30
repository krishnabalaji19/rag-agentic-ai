import os
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def run_ingestion(pdf_path: str, index_name: str):
    # 1. Load the PDF — one "document" object per page, each with page metadata
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from the PDF")

    # 2. Split the pages into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # 3. Set up the embedding model (turns text into vectors)
    embeddings = GoogleGenerativeAIEmbeddings(
     model="models/gemini-embedding-001",
     output_dimensionality=768
    )

    # 4. Upload chunks + their embeddings into Pinecone
    # Connect to the existing Pinecone index (doesn't upload anything yet)
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # Upload in small batches, pausing between each to respect the free-tier rate limit
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"Uploaded chunks {i} to {i + len(batch) - 1}")
        time.sleep(20)  # pause so we stay under Gemini's free-tier requests-per-minute limit

    print("Ingestion complete — chunks are now stored in Pinecone")
    return vector_store


if __name__ == "__main__":
    run_ingestion(
        pdf_path="data/Ebook-Agentic-AI.pdf",
        index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    )