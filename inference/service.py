"""
FastAPI service - Main application entry point.
"""
from fastapi import FastAPI
from inference.routes import (
    health_router,
    ask_router,
    ask_graph_router,
    infer_router,
    infer_graph_router,
    ingest_router,
    diagnostics_router,
    graph_export_router
)

app = FastAPI(title="Deep RAG API", version="0.1.0")

# Include all route routers
app.include_router(health_router)
app.include_router(ask_router)
app.include_router(ask_graph_router)
app.include_router(infer_router)
app.include_router(infer_graph_router)
app.include_router(ingest_router)
app.include_router(diagnostics_router)
app.include_router(graph_export_router)
