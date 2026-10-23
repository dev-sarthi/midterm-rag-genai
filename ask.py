"""
ask.py - Stage 2: Question Answering for Designs Law Assistant

Loads the persistent ChromaDB database from './db', retrieves the 4 most relevant
chunks using 'models/gemini-embedding-001', and generates an answer strictly
from the retrieved context using Gemini. If not explicitly supported, returns 'Not found'.
"""

import os
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress minor deprecation warnings for cleaner console output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure stdout handles UTF-8 safely on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# LangChain components
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --- Configuration Constants ---
CHROMA_PERSIST_DIRECTORY = "db"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
TOP_K = 4

_vector_store = None
_embeddings = None


def load_and_validate_env() -> str:
    """Loads environment variables and validates the Google Gemini API key."""
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key.strip() == "" or api_key.strip() == "your_gemini_api_key_here":
        raise ValueError("Missing or placeholder GOOGLE_API_KEY in environment or .env file.")

    return api_key.strip()


def validate_database(db_path: str) -> Path:
    """Verifies that the ChromaDB directory exists."""
    path = Path(db_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Database directory not found at: '{path.resolve()}'. Run ingest.py first.")
    return path


def extract_text(content) -> str:
    """Extracts plain text from LangChain / Google message content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)
    return str(content)


def get_vector_store():
    """Initializes and caches the persistent Chroma vector store."""
    global _vector_store, _embeddings
    if _vector_store is None:
        api_key = load_and_validate_env()
        validate_database(CHROMA_PERSIST_DIRECTORY)
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )
        _vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=_embeddings,
        )
    return _vector_store


def query_rag(question: str) -> dict:
    """
    Executes the full RAG pipeline for a given question.
    Returns a dictionary with 'question', 'answer', and 'sources'.
    """
    cleaned_question = question.strip() if question else ""
    if not cleaned_question:
        return {
            "question": "",
            "answer": "Not found",
            "sources": []
        }

    api_key = load_and_validate_env()
    vector_store = get_vector_store()

    # Verify vector store has data
    if vector_store._collection.count() == 0:
        raise ValueError("The Chroma database is empty. Please run ingest.py to populate it.")

    # 1. Retrieve Top-4 most relevant chunks
    docs = vector_store.similarity_search(cleaned_question, k=TOP_K)
    if not docs:
        return {
            "question": cleaned_question,
            "answer": "Not found",
            "sources": []
        }

    # 2. Build structured source metadata and context
    sources = []
    context_parts = []
    for i, doc in enumerate(docs, 1):
        page_num = doc.metadata.get("page_label") or (doc.metadata.get("page", 0) + 1)
        source_file = doc.metadata.get("source", "document.pdf")
        section = doc.metadata.get("title") or "General Section"
        snippet = doc.page_content.strip()

        sources.append({
            "chunk_index": i,
            "source": source_file,
            "page": page_num,
            "section": section,
            "content": snippet
        })

        context_parts.append(
            f"[Source: {source_file} | Page: {page_num} | Section: {section}]\n{snippet}"
        )

    formatted_context = "\n\n---\n\n".join(context_parts)

    # 3. Formulate strict prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a legal assistant for the Designs Law document.
Answer the user's question STRICTLY and ONLY based on the context provided below.

Rules:
1. If the context does not contain sufficient information to explicitly answer the question, output EXACTLY "Not found" and nothing else.
2. If the context does answer the question, provide a clear, accurate answer and cite the source PDF page numbers and relevant section headings mentioned in the context.
3. Do not assume or extrapolate beyond the provided text.

Context:
{context}

Question:
{question}

Answer:"""
    )

    formatted_prompt = prompt_template.format(context=formatted_context, question=cleaned_question)

    # 4. Invoke LLM
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.0,
    )
    response = llm.invoke(formatted_prompt)
    answer = extract_text(response.content).strip()

    # If the LLM returned Not found, normalize it
    if answer.lower() == "not found" or answer.lower().startswith("not found."):
        answer = "Not found"

    return {
        "question": cleaned_question,
        "answer": answer,
        "sources": sources
    }


def ask_question(question: str):
    """CLI wrapper for terminal testing."""
    try:
        result = query_rag(question)
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(result["answer"])
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        try:
            question = input("Enter your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not question:
        print("No question entered. Exiting.")
        sys.exit(0)

    ask_question(question)


if __name__ == "__main__":
    main()
