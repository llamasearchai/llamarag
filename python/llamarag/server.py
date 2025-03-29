#!/usr/bin/env python3
"""
Server runner for LlamaRAG API.

This script provides a convenient way to start the LlamaRAG API server
with various configuration options.

Usage:
    python -m llamarag.server [options]
"""

import argparse
import os
import sys
import uvicorn
from typing import Dict, Any, Optional

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LlamaRAG API Server")
    
    # Server configuration
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    # API configuration
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", 
                      help="HuggingFace embedding model to use")
    parser.add_argument("--enable-cors", action="store_true",
                      help="Enable CORS for all origins")
    parser.add_argument("--cors-origins", nargs="+", 
                      help="List of allowed origins for CORS")
    
    # Authentication
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--api-key-header", default="X-API-Key",
                      help="Header name for API key authentication")
    
    return parser.parse_args()

def configure_server(args: argparse.Namespace) -> Dict[str, Any]:
    """Configure server based on command line arguments."""
    # Set environment variables for API configuration
    if args.embedding_model:
        os.environ["LLAMARAG_EMBEDDING_MODEL"] = args.embedding_model
    
    if args.api_key:
        os.environ["LLAMARAG_API_KEY"] = args.api_key
        os.environ["LLAMARAG_API_KEY_HEADER"] = args.api_key_header
    
    if args.enable_cors:
        os.environ["LLAMARAG_ENABLE_CORS"] = "1"
    
    if args.cors_origins:
        os.environ["LLAMARAG_CORS_ORIGINS"] = ",".join(args.cors_origins)
    
    # Configure uvicorn settings
    uvicorn_config = {
        "app": "llamarag.api.app:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "workers": args.workers,
        "log_level": "info",
    }
    
    return uvicorn_config

def main() -> int:
    """Run the server."""
    try:
        args = parse_args()
        config = configure_server(args)
        
        print(f"Starting LlamaRAG API server at http://{args.host}:{args.port}")
        print(f"Using embedding model: {args.embedding_model}")
        
        uvicorn.run(**config)
        return 0
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 