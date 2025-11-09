"""
FastAPI service - Main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inference.routes import (
    health_router,
    ask_router,
    ask_graph_router,
    infer_router,
    infer_graph_router,
    ingest_router,
    diagnostics_router,
    graph_export_router,
    threads_router,
    documents_router
)

app = FastAPI(title="Deep RAG API", version="0.1.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all route routers
app.include_router(health_router)
app.include_router(ask_router)
app.include_router(ask_graph_router)
app.include_router(infer_router)
app.include_router(infer_graph_router)
app.include_router(ingest_router)
app.include_router(diagnostics_router)
app.include_router(graph_export_router)
app.include_router(threads_router)
app.include_router(documents_router)
