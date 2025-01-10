# LlamaRAG

A comprehensive Python package for building Retrieval-Augmented Generation (RAG) systems with large language models.

## Features

- **Flexible Chunking Strategies**: Various methods to split documents into optimal chunks
- **Vector-based Retrieval**: Efficient similarity search using embeddings
- **Reranking Capabilities**: Improve retrieval quality with cross-encoders and LLMs
- **Multiple Embedding Models**: Support for Anthropic, OpenAI, and HuggingFace embeddings
- **LlamaDB Integration**: Seamless connection with LlamaDB for vector storage
- **API Server**: Ready-to-use FastAPI application for embedding and retrieval

## Installation

```bash
# Basic installation
pip install llamarag

# With specific features
pip install llamarag[huggingface,anthropic,api]

# Full installation
pip install llamarag[dev,anthropic,openai,huggingface,llamadb,api]
```

## Quick Start

### Basic RAG Pipeline

```python
from llamarag.core.chunking import ParagraphChunker
from llamarag.core.embeddings import HuggingFaceEmbeddings
from llamarag.core.retrieval import Document, VectorRetriever

# Create sample documents
documents = [
    Document("LlamaRAG is a Python package for Retrieval-Augmented Generation."),
    Document("It provides tools for chunking, embedding, and retrieving text."),
    Document("RAG enhances LLM outputs by providing relevant context from a knowledge base.")
]

# Initialize components
chunker = ParagraphChunker()
embedding_model = HuggingFaceEmbeddings()
retriever = VectorRetriever(documents, embedding_model)

# Retrieve relevant documents
results = retriever.retrieve("What is RAG?", k=2)

# Print results
for result in results:
    print(f"Score: {result.score:.4f}, Content: {result.document.content}")
```

### Using LlamaDB Integration

```python
from llamarag.core.embeddings import HuggingFaceEmbeddings
from llamarag.integrations.llamadb import LlamaDBVectorStore, LlamaDBRetriever
from llamarag.core.retrieval import Document

# Initialize components
embedding_model = HuggingFaceEmbeddings()
vector_store = LlamaDBVectorStore(collection_name="my_documents")
retriever = LlamaDBRetriever(vector_store, embedding_model)

# Create and add documents
documents = [
    Document("Document 1 content", metadata={"source": "book1", "page": 10}),
    Document("Document 2 content", metadata={"source": "book2", "page": 15})
]
doc_ids = retriever.add_documents(documents)

# Retrieve documents
results = retriever.retrieve("query text", k=2)

# Print results
for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Content: {result.document.content}")
    print(f"Metadata: {result.document.metadata}")
```

## Advanced Features

### Custom Chunking

```python
from llamarag.core.chunking import OverlappingChunker

# Create chunker with custom settings
chunker = OverlappingChunker(
    chunk_size=500,  # Characters per chunk
    overlap_size=100,  # Overlap between chunks
    split_tokens=["\n\n", "\n", ".", " "]  # Priority for split points
)

# Chunk a document
text = "Your long document text here..."
chunks = chunker.chunk(text)
```

### Reranking

```python
from llamarag.core.retrieval import Document, VectorRetriever
from llamarag.core.embeddings import HuggingFaceEmbeddings
from llamarag.core.reranking import CrossEncoderReranker, RerankingRetriever

# Initialize components
documents = [Document("...") for _ in range(10)]
embedding_model = HuggingFaceEmbeddings()
vector_retriever = VectorRetriever(documents, embedding_model)

# Create reranker
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Create reranking retriever
retriever = RerankingRetriever(
    base_retriever=vector_retriever,
    reranker=reranker,
    candidate_multiplier=3  # Get 3x more candidates before reranking
)

# Retrieve and rerank in one step
results = retriever.retrieve("query", k=3)
```

## API Server

Run the built-in API server:

```bash
pip install llamarag[api]
uvicorn llamarag.api.app:app --reload
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
# Updated in commit 1 - 2025-04-04 17:32:10

# Updated in commit 9 - 2025-04-04 17:32:10

# Updated in commit 17 - 2025-04-04 17:32:11

# Updated in commit 25 - 2025-04-04 17:32:13

# Updated in commit 1 - 2025-04-05 14:36:01

# Updated in commit 9 - 2025-04-05 14:36:01

# Updated in commit 17 - 2025-04-05 14:36:02

# Updated in commit 25 - 2025-04-05 14:36:02

# Updated in commit 1 - 2025-04-05 15:22:31

# Updated in commit 9 - 2025-04-05 15:22:32

# Updated in commit 17 - 2025-04-05 15:22:32

# Updated in commit 25 - 2025-04-05 15:22:32

# Updated in commit 1 - 2025-04-05 15:56:50

# Updated in commit 9 - 2025-04-05 15:56:50
