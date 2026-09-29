import os
import sys
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Import configuration
from src.config import get_embeddings, PINECONE_INDEX_NAME, DEFAULT_PDF_PATH


def run_ingestion(pdf_path: str = DEFAULT_PDF_PATH, index_name: str = PINECONE_INDEX_NAME):
    """
    Parses the PDF document, splits it into text chunks, generates vector embeddings,
    and indexes them into the specified Pinecone vector index.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    print(f"[ETL] Loading PDF document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[ETL] Successfully loaded {len(docs)} pages.")

    # Chunk the document text
    print("[ETL] Splitting document into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[ETL] Total text chunks created: {len(chunks)}")

    # Initialize embedding model
    print("[ETL] Initializing embedding model...")
    embeddings = get_embeddings()

    # Initialize Pinecone vector store
    print(f"[ETL] Connecting to Pinecone index: '{index_name}'...")
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # Upsert chunks into Pinecone in batches
    batch_size = 20
    total_chunks = len(chunks)
    print(f"[ETL] Uploading {total_chunks} chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"[ETL] Uploaded chunks {i + 1} to {min(i + batch_size, total_chunks)} of {total_chunks}")
        time.sleep(2)  # Short pause between batches to prevent rate limiting

    print("[ETL] Document ingestion complete! All vector embeddings indexed in Pinecone.")
    return vector_store


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    idx = sys.argv[2] if len(sys.argv) > 2 else PINECONE_INDEX_NAME
    run_ingestion(pdf_path=pdf, index_name=idx)