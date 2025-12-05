# ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "data/data.pdf"       # change if your file has a different name
DB_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_split(path: str):
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    return splitter.split_documents(docs)


def build_vector_store(docs):
    """Embed all chunks and build a local FAISS index."""
    print("Loading local embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    print("Building FAISS index...")
    vectordb = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model,
    )

    vectordb.save_local(DB_PATH)
    print(f"FAISS index saved in: {DB_PATH}/")


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    print("Loading and splitting PDF...")
    documents = load_and_split(PDF_PATH)
    print(f"Chunks: {len(documents)}")

    build_vector_store(documents)
    print("Ingest complete.")
