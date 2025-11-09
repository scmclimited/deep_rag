"""
Graph export route - Export LangGraph visualization.
"""
import logging
from fastapi import APIRouter, HTTPException
from inference.graph.graph_viz import export_graph_png

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/graph")
def graph_export(out: str = "inference/graph/artifacts/deep_rag_graph.png"):
    """
    Export the LangGraph as PNG (Graphviz) or Mermaid (.mmd fallback).
    
    Returns the path to the generated graph file.
    """
    try:
        path = export_graph_png(out)
        return {
            "status": "success",
            "path": path,
            "format": "png" if path.endswith(".png") else "mermaid"
        }
    except Exception as e:
        logger.error(f"Error in /graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

