"""
API server for LlamaRAG.
"""

try:
    from llamarag.api.app import app
    has_fastapi = True
except ImportError:
    has_fastapi = False

__all__ = []

if has_fastapi:
    __all__.append("app") 