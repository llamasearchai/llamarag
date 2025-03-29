#!/usr/bin/env python3
"""
Document Q&A Example

This example demonstrates a complete document question-answering system using LlamaRAG.
It loads a text file, chunks it, embeds the chunks, and then answers questions using
retrieval-augmented generation.

Usage:
    python document_qa.py <document_file> <question>

Example:
    python document_qa.py my_document.txt "What are the key points in this document?"
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path for running the example directly
sys.path.append(str(Path(__file__).resolve().parent.parent / "python"))

from llamarag.core.chunking import OverlappingChunker
from llamarag.core.embeddings import HuggingFaceEmbeddings
from llamarag.core.retrieval import Document, VectorRetriever, SearchResult
from llamarag.core.reranking import CrossEncoderReranker, RerankingRetriever
from llamarag.utils.retrieval_utils import batch_embed_documents, format_retrieval_results

# Optional: Use OpenAI for the LLM if available
try:
    import openai
    has_openai = True
except ImportError:
    has_openai = False
    print("OpenAI package not found. Will print prompts instead of generating answers.")


def load_document(file_path: str) -> str:
    """Load document from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_document(text: str) -> List[Document]:
    """Chunk the document into smaller pieces."""
    # Use overlapping chunker for better context handling
    chunker = OverlappingChunker(
        chunk_size=1000,
        overlap_size=200,
        split_tokens=["\n\n", "\n", ".", " "]
    )
    
    chunks = chunker.chunk(text)
    
    # Convert chunks to Document objects
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            content=chunk,
            metadata={
                "chunk_id": i,
                "start_char": text.find(chunk),
                "chunk_index": i,
            }
        )
        documents.append(doc)
    
    return documents


def setup_retriever(documents: List[Document]) -> RerankingRetriever:
    """Set up a retriever with reranking capabilities."""
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Embed documents
    documents = batch_embed_documents(documents, embedding_model)
    
    # Create base retriever
    base_retriever = VectorRetriever(documents, embedding_model)
    
    # Add reranker for better results
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Create reranking retriever
    retriever = RerankingRetriever(
        base_retriever=base_retriever,
        reranker=reranker,
        candidate_multiplier=3  # Get 3x more candidates before reranking
    )
    
    return retriever


def answer_question(question: str, context: str, openai_api_key: Optional[str] = None) -> str:
    """Answer question using OpenAI API with context."""
    if not has_openai:
        return "OpenAI package not installed. Cannot generate answer."
    
    if openai_api_key:
        openai.api_key = openai_api_key
    
    # Check if API key is available
    if not openai.api_key:
        return "OpenAI API key not provided. Cannot generate answer."
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question based only on the provided context. If the answer cannot be found in the context, say 'I don't have enough information to answer this question.'"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"


def create_context_from_results(results: List[SearchResult]) -> str:
    """Create a context string from search results."""
    context_parts = []
    
    for i, result in enumerate(results):
        context_parts.append(f"[Document {i+1}]\n{result.document.content}\n")
    
    return "\n".join(context_parts)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Document Q&A using LlamaRAG")
    parser.add_argument("document_file", help="Path to document file to process")
    parser.add_argument("question", help="Question to answer about the document")
    parser.add_argument("--openai-key", help="OpenAI API key (optional)")
    args = parser.parse_args()
    
    # Load document
    print(f"Loading document from {args.document_file}...")
    try:
        text = load_document(args.document_file)
    except Exception as e:
        print(f"Error loading document: {e}")
        return 1
    
    # Chunk document
    print("Chunking document...")
    documents = chunk_document(text)
    print(f"Document chunked into {len(documents)} pieces")
    
    # Set up retriever
    print("Setting up retriever...")
    retriever = setup_retriever(documents)
    
    # Process question
    question = args.question
    print(f"\nQuestion: {question}")
    
    # Retrieve relevant chunks
    print("\nRetrieving relevant context...")
    results = retriever.retrieve(question, k=3)
    
    # Print retrieval results
    print("\nRetrieved Context:")
    print("=================")
    print(format_retrieval_results(results))
    
    # Create context for LLM
    context = create_context_from_results(results)
    
    # Answer question
    if has_openai:
        print("\nGenerating answer...")
        answer = answer_question(question, context, args.openai_key)
        
        print("\nAnswer:")
        print("=======")
        print(answer)
    else:
        print("\nPrompt for LLM:")
        print("==============")
        prompt = f"""
Context:
{context}

Question: {question}

Answer:
"""
        print(prompt)
        print("\nTo generate answers, install OpenAI package with: pip install openai")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 