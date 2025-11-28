"""
FastAPI backend exposing RAG capabilities.
"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import get_settings
from .rag_pipeline import RAGPipeline
from .utils import cleanup_file, save_upload_file_tmp

settings = get_settings()
pipeline = RAGPipeline(settings)

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbedRequest(BaseModel):
    texts: List[str]


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF, create embeddings, and refresh vector store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    temp_path = save_upload_file_tmp(file)

    try:
        documents = pipeline.load_documents(temp_path)
        chunks = pipeline.split_documents(documents)
        pipeline.build_vector_store(chunks)
        return {
            "message": "Document indexed successfully",
            "chunks_indexed": len(chunks),
        }
    except Exception as exc:  # pragma: no cover - surfaced via API
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        cleanup_file(temp_path)


@app.post("/embed")
async def embed_texts(request: EmbedRequest):
    """Return embeddings for ad-hoc text input."""
    try:
        embeddings = pipeline.embed_texts(request.texts)
        return {"embeddings": embeddings, "count": len(embeddings)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
async def query_rag(request: QueryRequest):
    """Ask a question to the RAG pipeline."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        response = pipeline.query(request.question)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


__all__ = ["app"]


