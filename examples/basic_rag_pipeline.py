#!/usr/bin/env python3
"""
Basic RAG Pipeline Example

This example demonstrates a simple Retrieval-Augmented Generation pipeline using LlamaRAG.
It creates sample documents, embeds them, performs retrieval, and prepares a prompt for an LLM.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for running the example directly
sys.path.append(str(Path(__file__).resolve().parent.parent / "python"))

from llamarag.core.chunking import SentenceChunker
from llamarag.core.embeddings import HuggingFaceEmbeddings
from llamarag.core.retrieval import Document, VectorRetriever, SearchResult
from llamarag.core.reranking import CrossEncoderReranker
from llamarag.utils.retrieval_utils import format_retrieval_results

# Sample knowledge base documents
documents = [
    Document(
        "LlamaRAG is a Python package for building Retrieval-Augmented Generation systems.",
        metadata={"source": "documentation", "section": "introduction"}
    ),
    Document(
        "Retrieval-Augmented Generation (RAG) combines retrieval from a knowledge base with text generation.",
        metadata={"source": "documentation", "section": "concepts"}
    ),
    Document(
        "RAG enhances LLM outputs by providing relevant context from a knowledge base.",
        metadata={"source": "documentation", "section": "benefits"}
    ),
    Document(
        "LlamaRAG supports various chunking strategies like sentence, paragraph, and fixed-size chunking.",
        metadata={"source": "documentation", "section": "features"}
    ),
    Document(
        "Vector retrieval uses embeddings to find documents similar to a query.",
        metadata={"source": "documentation", "section": "retrieval"}
    ),
    Document(
        "Reranking improves retrieval quality by applying more sophisticated relevance models to initial results.",
        metadata={"source": "documentation", "section": "reranking"}
    ),
    Document(
        "LlamaRAG integrates with popular embedding models including those from HuggingFace.",
        metadata={"source": "documentation", "section": "integrations"}
    ),
    Document(
        "The LlamaDB integration provides efficient vector storage and retrieval capabilities.",
        metadata={"source": "documentation", "section": "database"}
    )
]

def main():
    print("LlamaRAG Basic Example")
    print("======================\n")
    
    # Initialize the embedding model
    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create a retriever
    print("Creating retriever with sample documents...")
    retriever = VectorRetriever(documents, embedding_model)
    
    # Optional: Add a reranker
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Run a sample query
    query = "What is RAG and how does it enhance LLMs?"
    print(f"\nQuery: {query}")
    
    # Retrieve relevant documents
    print("\nRetrieving documents...")
    results = retriever.retrieve(query, k=3)
    
    # Print results
    print("\nRetrieval Results:")
    print("=================")
    print(format_retrieval_results(results))
    
    # Optional: Rerank results
    print("\nReranking results...")
    reranked_results = reranker.rerank(query, results)
    
    print("\nReranked Results:")
    print("================")
    print(format_retrieval_results(reranked_results))
    
    # Prepare prompt for an LLM (this is where you would use an LLM API)
    context = "\n".join([f"Context {i+1}:\n{result.document.content}" 
                        for i, result in enumerate(reranked_results)])
    
    prompt = f"""Answer the following question based on the provided context:

Context:
{context}

Question: {query}

Answer:"""
    
    print("\nPrompt for LLM:")
    print("==============")
    print(prompt)
    
    # In a real application, you would now send this prompt to an LLM
    print("\nIn a complete application, this prompt would be sent to an LLM API.")

if __name__ == "__main__":
    main() 