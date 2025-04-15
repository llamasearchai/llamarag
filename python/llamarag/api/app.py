"""
FastAPI application for llamarag.
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from llamarag.core.chunking import (
    Chunker,
    FixedSizeChunker,
    ParagraphChunker,
    SentenceChunker,
)
from llamarag.core.embeddings import EmbeddingModel, HuggingFaceEmbeddings
from llamarag.core.retrieval import Document, SearchResult, VectorRetriever

# API configuration from environment variables
EMBEDDING_MODEL = os.environ.get("LLAMARAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
API_KEY = os.environ.get("LLAMARAG_API_KEY")
API_KEY_HEADER = os.environ.get("LLAMARAG_API_KEY_HEADER", "X-API-Key")
ENABLE_CORS = os.environ.get("LLAMARAG_ENABLE_CORS", "0") == "1"
CORS_ORIGINS = (
    os.environ.get("LLAMARAG_CORS_ORIGINS", "").split(",")
    if os.environ.get("LLAMARAG_CORS_ORIGINS")
    else []
)

# Initialize API key security
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

app = FastAPI(
    title="LlamaRAG API",
    description="API for Retrieval-Augmented Generation with LLMs",
    version="0.1.0",
)

# Configure CORS
if ENABLE_CORS:
    origins = ["*"] if not CORS_ORIGINS else CORS_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# API key security dependency
async def verify_api_key(api_key: str = Security(api_key_header)):
    if API_KEY and (api_key != API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
    return api_key


# Models for API requests and responses
class ChunkRequest(BaseModel):
    text: str
    chunker_type: str = "paragraph"
    max_chunk_size: Optional[int] = None
    overlap_size: Optional[int] = None


class ChunkResponse(BaseModel):
    chunks: List[str]


class DocumentModel(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 3
    rerank: bool = False


class SearchResultModel(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int


# Global instances
chunkers = {
    "paragraph": ParagraphChunker(),
    "sentence": SentenceChunker(),
    "fixed": FixedSizeChunker(),
}

embedding_model = None
documents = []
retriever = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global embedding_model

    # Initialize default embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@app.get("/")
async def root():
    """Root endpoint providing API info."""
    return {
        "name": "LlamaRAG API",
        "version": "0.1.0",
        "endpoints": [
            "/chunk - Chunk text",
            "/documents - Add or retrieve documents",
            "/search - Search for relevant documents",
        ],
    }


@app.post("/chunk", response_model=ChunkResponse)
async def chunk_text(request: ChunkRequest, api_key: str = Depends(verify_api_key)):
    """Chunk text using specified chunker."""
    chunker_type = request.chunker_type.lower()

    if chunker_type not in chunkers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown chunker type: {chunker_type}. Available options: {', '.join(chunkers.keys())}",
        )

    # Use default chunker or create custom one
    if request.max_chunk_size is None:
        chunker = chunkers[chunker_type]
    else:
        if chunker_type == "paragraph":
            chunker = ParagraphChunker(max_paragraphs_per_chunk=request.max_chunk_size)
        elif chunker_type == "sentence":
            chunker = SentenceChunker(max_sentences_per_chunk=request.max_chunk_size)
        elif chunker_type == "fixed":
            overlap = request.overlap_size or 200
            chunker = FixedSizeChunker(
                chunk_size=request.max_chunk_size, chunk_overlap=overlap
            )

    # Chunk the text
    chunks = chunker.chunk(request.text)
    return ChunkResponse(chunks=chunks)


@app.post("/documents", response_model=List[str])
async def add_documents(
    documents_list: List[DocumentModel], api_key: str = Depends(verify_api_key)
):
    """Add documents to the vector store."""
    global retriever, documents, embedding_model

    # Convert to internal Document format
    docs = [
        Document(content=doc.content, metadata=doc.metadata, id=doc.id)
        for doc in documents_list
    ]

    # Add documents to global list
    documents.extend(docs)

    # Recreate retriever with updated documents
    if embedding_model is not None:
        # Embed the new documents
        for doc in docs:
            if doc.embedding is None:
                doc.embedding = embedding_model.embed(doc.content)

        # Create or update retriever
        retriever = VectorRetriever(documents, embedding_model)

    # Return document IDs
    return [doc.id for doc in docs]


@app.get("/documents", response_model=List[DocumentModel])
async def get_documents(api_key: str = Depends(verify_api_key)):
    """Get all documents."""
    global documents

    return [
        DocumentModel(content=doc.content, metadata=doc.metadata, id=doc.id)
        for doc in documents
    ]


@app.delete("/documents")
async def clear_documents(api_key: str = Depends(verify_api_key)):
    """Clear all documents."""
    global documents, retriever

    documents = []
    retriever = None

    return {"status": "success", "message": "All documents cleared"}


@app.post("/search", response_model=List[SearchResultModel])
async def search(request: SearchRequest, api_key: str = Depends(verify_api_key)):
    """Search for relevant documents."""
    global retriever, documents, embedding_model

    if not documents:
        raise HTTPException(status_code=400, detail="No documents available for search")

    if retriever is None:
        if embedding_model is None:
            raise HTTPException(
                status_code=500, detail="Embedding model not initialized"
            )

        # Create retriever
        retriever = VectorRetriever(documents, embedding_model)

    # Retrieve documents
    results = retriever.retrieve(request.query, k=request.k)

    # Rerank if requested
    if request.rerank:
        from llamarag.core.reranking import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        results = reranker.rerank(request.query, results)

    # Convert to API format
    return [
        SearchResultModel(
            content=result.document.content,
            metadata=result.document.metadata,
            score=float(result.score),  # Ensure score is a Python float
            rank=result.rank,
        )
        for result in results
    ]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "embedding_model": EMBEDDING_MODEL,
        "documents_count": len(documents),
        "api_auth_enabled": API_KEY is not None,
    }
