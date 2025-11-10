"""
FastAPI service - Main application entry point.
"""
import logging
import sys
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

# Configure verbose logging for Docker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout for Docker
    ],
    force=True  # Override any existing configuration
)

# Set all relevant loggers to INFO for verbose output
for logger_name in [
    'inference.graph',
    'inference.graph.graph_wrapper',
    'inference.graph.nodes',
    'inference.graph.nodes.planner',
    'inference.graph.nodes.retriever',
    'inference.graph.nodes.compressor',
    'inference.graph.nodes.critic',
    'inference.graph.nodes.synthesizer',
    'inference.graph.nodes.refine_retrieve',
    'inference.routes',
    'retrieval',
    'uvicorn.access'
]:
    logging.getLogger(logger_name).setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("Deep RAG API starting with VERBOSE logging enabled")
logger.info("=" * 80)

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
