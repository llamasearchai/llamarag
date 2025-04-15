"""
Integrations with external systems for LlamaRAG.
"""

try:
    from llamarag.integrations.llamadb import LlamaDBRetriever, LlamaDBVectorStore

    has_llamadb = True
except ImportError:
    has_llamadb = False

__all__ = []

if has_llamadb:
    __all__.extend(["LlamaDBVectorStore", "LlamaDBRetriever"])
