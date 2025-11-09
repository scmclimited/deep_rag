"""
REST API route modules.
"""
from inference.routes.health import router as health_router
from inference.routes.ask import router as ask_router
from inference.routes.ask_graph import router as ask_graph_router
from inference.routes.infer import router as infer_router
from inference.routes.infer_graph import router as infer_graph_router
from inference.routes.ingest import router as ingest_router
from inference.routes.diagnostics import router as diagnostics_router
from inference.routes.graph_export import router as graph_export_router
from inference.routes.threads import router as threads_router
from inference.routes.documents import router as documents_router

__all__ = [
    'health_router',
    'ask_router',
    'ask_graph_router',
    'infer_router',
    'infer_graph_router',
    'ingest_router',
    'diagnostics_router',
    'graph_export_router',
    'threads_router',
    'documents_router',
]

