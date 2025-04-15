#!/usr/bin/env python3
"""
Command-line interface for LlamaRAG.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from llamarag.core.chunking import FixedSizeChunker, ParagraphChunker, SentenceChunker
from llamarag.core.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from llamarag.core.reranking import CrossEncoderReranker
from llamarag.core.retrieval import Document, VectorRetriever
from llamarag.utils.retrieval_utils import (
    batch_embed_documents,
    format_retrieval_results,
    load_documents_from_json,
    save_documents_to_json,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LlamaRAG - Retrieval-Augmented Generation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Chunk documents")
    chunk_parser.add_argument("input_file", help="Input file path (text or JSON)")
    chunk_parser.add_argument("output_file", help="Output JSON file path")
    chunk_parser.add_argument(
        "--method",
        choices=["sentence", "paragraph", "fixed"],
        default="paragraph",
        help="Chunking method",
    )
    chunk_parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Chunk size (sentences, paragraphs, or characters depending on method)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap size (only for fixed size chunking)",
    )

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed documents")
    embed_parser.add_argument("input_file", help="Input JSON file with documents")
    embed_parser.add_argument("output_file", help="Output JSON file path")
    embed_parser.add_argument(
        "--model",
        choices=["huggingface", "openai"],
        default="huggingface",
        help="Embedding model provider",
    )
    embed_parser.add_argument(
        "--model-name", default="all-MiniLM-L6-v2", help="Model name or path"
    )
    embed_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embedding"
    )
    embed_parser.add_argument(
        "--api-key", help="API key for the embedding service (if needed)"
    )

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve documents")
    retrieve_parser.add_argument(
        "documents_file", help="JSON file with embedded documents"
    )
    retrieve_parser.add_argument("--query", help="Query text")
    retrieve_parser.add_argument("--query-file", help="File containing query text")
    retrieve_parser.add_argument(
        "--top-k", type=int, default=3, help="Number of documents to retrieve"
    )
    retrieve_parser.add_argument(
        "--rerank", action="store_true", help="Use cross-encoder reranking"
    )
    retrieve_parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking",
    )
    retrieve_parser.add_argument(
        "--output-file", help="Output file for results (JSON format)"
    )

    return parser


def chunk_command(args: argparse.Namespace) -> int:
    """Execute the chunk command."""
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # Determine input type and load content
    if input_path.suffix.lower() == ".json":
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                # Assume list of documents
                if all(isinstance(d, dict) and "content" in d for d in data):
                    documents = load_documents_from_json(str(input_path))
                    texts = [doc.content for doc in documents]
                else:
                    # Just a list of strings
                    texts = data
            elif isinstance(data, dict) and "texts" in data:
                texts = data["texts"]
            else:
                print(f"Error: Unsupported JSON format in {input_path}")
                return 1
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {input_path}")
            return 1
    else:
        # Assume plain text file
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
            texts = [text]
        except Exception as e:
            print(f"Error reading file {input_path}: {e}")
            return 1

    # Create chunker based on method
    if args.method == "sentence":
        chunker = SentenceChunker(max_sentences=args.size)
    elif args.method == "paragraph":
        chunker = ParagraphChunker(max_paragraphs=args.size)
    else:  # fixed
        chunker = FixedSizeChunker(chunk_size=args.size, overlap_size=args.overlap)

    # Chunk documents
    chunked_documents = []
    for i, text in enumerate(texts):
        chunks = chunker.chunk(text)
        for j, chunk in enumerate(chunks):
            doc = Document(
                content=chunk,
                metadata={
                    "source": input_path.name,
                    "document_index": i,
                    "chunk_index": j,
                    "chunking_method": args.method,
                },
            )
            chunked_documents.append(doc)

    # Save chunked documents
    save_documents_to_json(chunked_documents, str(output_path))
    print(f"Chunked {len(texts)} documents into {len(chunked_documents)} chunks")
    print(f"Saved chunks to {output_path}")

    return 0


def embed_command(args: argparse.Namespace) -> int:
    """Execute the embed command."""
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # Load documents
    try:
        documents = load_documents_from_json(str(input_path))
    except Exception as e:
        print(f"Error loading documents from {input_path}: {e}")
        return 1

    # Create embedding model
    if args.model == "huggingface":
        embedding_model = HuggingFaceEmbeddings(model_name=args.model_name)
    elif args.model == "openai":
        if not args.api_key:
            print("Error: OpenAI embeddings require an API key")
            return 1
        embedding_model = OpenAIEmbeddings(
            api_key=args.api_key, model_name=args.model_name
        )
    else:
        print(f"Error: Unsupported embedding model {args.model}")
        return 1

    # Embed documents
    print(f"Embedding {len(documents)} documents with {args.model} model...")
    documents = batch_embed_documents(
        documents, embedding_model, batch_size=args.batch_size
    )

    # Save embedded documents
    save_documents_to_json(documents, str(output_path))
    print(f"Embedded {len(documents)} documents")
    print(f"Saved embedded documents to {output_path}")

    return 0


def retrieve_command(args: argparse.Namespace) -> int:
    """Execute the retrieve command."""
    documents_path = Path(args.documents_file)

    # Load documents
    try:
        documents = load_documents_from_json(str(documents_path))
    except Exception as e:
        print(f"Error loading documents from {documents_path}: {e}")
        return 1

    # Check if documents have embeddings
    if any(doc.embedding is None for doc in documents):
        print("Warning: Some documents don't have embeddings")
        documents = [doc for doc in documents if doc.embedding is not None]
        if not documents:
            print("Error: No documents with embeddings found")
            return 1

    # Get query
    if args.query:
        query = args.query
    elif args.query_file:
        try:
            with open(args.query_file, "r", encoding="utf-8") as f:
                query = f.read().strip()
        except Exception as e:
            print(f"Error reading query file {args.query_file}: {e}")
            return 1
    else:
        print("Error: Either --query or --query-file must be provided")
        return 1

    # Create retriever
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever = VectorRetriever(documents, embedding_model)

    # Retrieve documents
    print(f"Retrieving documents for query: {query}")
    results = retriever.retrieve(query, k=args.top_k)

    # Rerank if requested
    if args.rerank:
        print("Reranking results...")
        reranker = CrossEncoderReranker(args.reranker_model)
        results = reranker.rerank(query, results)

    # Display results
    print("\nResults:")
    print("--------")
    print(format_retrieval_results(results))

    # Save results if requested
    if args.output_file:
        output = {
            "query": query,
            "results": [
                {
                    "content": r.document.content,
                    "metadata": r.document.metadata,
                    "score": r.score,
                }
                for r in results
            ],
        }

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")

    return 0


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "chunk":
        return chunk_command(args)
    elif args.command == "embed":
        return embed_command(args)
    elif args.command == "retrieve":
        return retrieve_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
