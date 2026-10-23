"""
main.py - FastAPI Backend for Designs Law Assistant

Serves the frontend interface and provides the /api/ask endpoint for RAG queries.
Reuses the verified pipeline in ask.py without modifying or duplicating ingestion.
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ask import query_rag, get_vector_store

app = FastAPI(
    title="Designs Law Assistant API",
    description="RAG-powered legal research assistant for Designs Law",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static folder exists and mount it
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class QuestionRequest(BaseModel):
    question: str = Field(..., description="The user's query about Designs Law")


class SourceItem(BaseModel):
    chunk_index: int
    source: str
    page: int | str
    section: str
    content: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]


@app.on_event("startup")
def startup_event():
    """Warm up the vector store on startup so queries execute quickly."""
    try:
        get_vector_store()
    except Exception as e:
        print(f"[WARNING] Vector store warmup skipped: {e}")


@app.get("/", response_class=FileResponse)
def serve_index():
    """Serves the main frontend single-page application."""
    index_path = static_dir / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_path))


@app.post("/api/ask", response_model=AnswerResponse)
def ask_endpoint(payload: QuestionRequest):
    """Processes a user question through the RAG pipeline."""
    raw_question = payload.question.strip() if payload.question else ""

    if not raw_question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty. Please provide a valid inquiry."
        )

    try:
        result = query_rag(raw_question)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=500, detail="Database files not found. Ingestion must be run first.")
    except Exception as e:
        # Protect against leaking internal stack traces or secrets
        print(f"[ERROR in /api/ask]: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your legal query. Please verify your connection and API key quota, then try again."
        )


@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "ok", "service": "Designs Law Assistant"}
