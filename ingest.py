"""
ingest.py - Stage 1: Document Ingestion for Designs Law Assistant

This script handles the document ingestion pipeline:
1. Loads the source PDF ('document.pdf') using PyPDFLoader.
2. Splits the document into 800-character chunks with a 150-character overlap.
3. Preserves metadata including source filename and page numbers.
4. Generates vector embeddings using Google Gemini's 'models/gemini-embedding-001'.
5. Persists the embeddings and metadata in a local ChromaDB vector store ('db' folder).
"""

import os
import sys
import time
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress minor deprecation warnings for cleaner beginner-friendly console output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure stdout handles UTF-8 safely on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# --- Configuration Constants ---
PDF_FILE_PATH = "document.pdf"
CHROMA_PERSIST_DIRECTORY = "db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "models/gemini-embedding-001"


def load_and_validate_env() -> str:
    """Loads environment variables and validates the Google Gemini API key."""
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key.strip() == "" or api_key.strip() == "your_gemini_api_key_here":
        print("\n[ERROR] Missing or placeholder GOOGLE_API_KEY.")
        print("Please follow these steps:")
        print("  1. Copy '.env.example' to '.env'")
        print("  2. Open '.env' and paste your Gemini API key from Google AI Studio:")
        print("     https://aistudio.google.com/")
        print("  3. Save the file and run ingest.py again.\n")
        sys.exit(1)

    return api_key.strip()


def validate_pdf_file(file_path: str) -> Path:
    """Checks whether the PDF file exists and is accessible."""
    pdf_path = Path(file_path)
    if not pdf_path.is_file():
        print(f"\n[ERROR] PDF file not found at: '{pdf_path.resolve()}'")
        print("Please ensure your PDF is named 'document.pdf' and placed in the project root directory.\n")
        sys.exit(1)
    return pdf_path


def run_ingestion():
    """Executes the ingestion workflow step-by-step with clear feedback."""
    print("=" * 60)
    print("Designs Law Assistant - Document Ingestion (Stage 1)")
    print("=" * 60)

    # 1. Check API Key
    api_key = load_and_validate_env()
    print("[OK] GOOGLE_API_KEY loaded successfully.")

    # 2. Check PDF File
    pdf_path = validate_pdf_file(PDF_FILE_PATH)
    print(f"[OK] Found document: {pdf_path.name}")

    # 3. Load PDF
    print("\n[Step 1/4] Loading PDF document...")
    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
    except Exception as e:
        print(f"\n[ERROR] Failed to load PDF '{pdf_path.name}': {e}")
        sys.exit(1)

    page_count = len(documents)
    if page_count == 0:
        print(f"\n[ERROR] PDF loaded but contains 0 pages or no extractable text.")
        sys.exit(1)
    print(f"[OK] Total pages loaded: {page_count}")

    # 4. Split into chunks with overlap
    print("\n[Step 2/4] Splitting document into text chunks...")
    print(f"  - Chunk size: {CHUNK_SIZE} characters")
    print(f"  - Chunk overlap: {CHUNK_OVERLAP} characters")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    try:
        chunks = text_splitter.split_documents(documents)
    except Exception as e:
        print(f"\n[ERROR] Failed to split documents: {e}")
        sys.exit(1)

    chunk_count = len(chunks)
    if chunk_count == 0:
        print("\n[ERROR] Text splitting produced 0 chunks. Please check if the document contains readable text.")
        sys.exit(1)
    print(f"[OK] Total chunks created: {chunk_count}")

    # Verify metadata preservation (source and page)
    sample_chunk = chunks[0]
    print(f"[OK] Preserved metadata example (chunk #1): {sample_chunk.metadata}")

    # 5. Initialize Google Gemini Embeddings
    print(f"\n[Step 3/4] Initializing Gemini Embeddings ({EMBEDDING_MODEL})...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize Gemini embeddings: {e}")
        sys.exit(1)
    print("[OK] Embeddings model initialized.")

    # 6. Embed and Persist to ChromaDB
    print(f"\n[Step 4/4] Generating embeddings and saving to './{CHROMA_PERSIST_DIRECTORY}'...")
    try:
        BATCH_SIZE = 50
        total_chunks = len(chunks)
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )

        # Reset collection to avoid duplicate vectors on re-runs
        try:
            vector_store.delete_collection()
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
            )
        except Exception:
            pass

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  - Ingesting batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            # Handle rate limits gracefully (Gemini free tier has 100 requests/minute)
            for attempt in range(5):
                try:
                    vector_store.add_documents(batch)
                    break
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        print("    [Rate limit reached, pausing 15s for quota reset...]")
                        time.sleep(15)
                    else:
                        raise e
    except Exception as e:
        print(f"\n[ERROR] Failed to generate embeddings or save to ChromaDB: {e}")
        print("Tip: Check your internet connection and verify that your Gemini API key has quota enabled.")
        sys.exit(1)

    # 7. Print summary and confirmation
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE & DATABASE CONFIRMATION")
    print("=" * 60)
    print(f"• Source Document   : {pdf_path.name}")
    print(f"• Total Pages       : {page_count}")
    print(f"• Total Chunks      : {chunk_count}")
    print(f"• Embedding Model   : {EMBEDDING_MODEL}")
    print(f"• Database Location : {Path(CHROMA_PERSIST_DIRECTORY).resolve()}")
    print("• Database Status   : Successfully saved and ready for Stage 2 querying!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_ingestion()
