from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
from pathlib import Path
import logging

from inference.agent_loop import run_deep_rag
from inference.graph.graph_wrapper import ask_with_graph
from ingestion.ingest import ingest as ingest_pdf
from ingestion.ingest_text import ingest_text_file
from ingestion.ingest_image import ingest_image
from retrieval.diagnostics import inspect_document

logger = logging.getLogger(__name__)

app = FastAPI(title="Deep RAG API", version="0.1.0")

class AskBody(BaseModel):
    question: str

class InferBody(BaseModel):
    question: str
    title: Optional[str] = None

class AskGraphBody(BaseModel):
    question: str
    thread_id: Optional[str] = "default"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(body: AskBody):
    """
    Query existing documents in the vector database using direct pipeline.
    Assumes documents have already been ingested.
    Uses agent_loop.py (direct pipeline without LangGraph).
    """
    try:
        answer = run_deep_rag(body.question)
        return {"answer": answer, "mode": "query_only", "pipeline": "direct"}
    except Exception as e:
        logger.error(f"Error in /ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/diagnostics/document")
def diagnostics(doc_title: Optional[str] = None, doc_id: Optional[str] = None):
    """
    Inspect what chunks and pages are stored for a document.
    Useful for debugging ingestion and retrieval issues.
    
    Query params:
        doc_title: Document title to search for (partial match)
        doc_id: Document ID (UUID) - if provided, doc_title is ignored
    """
    try:
        result = inspect_document(doc_title=doc_title, doc_id=doc_id)
        return result
    except Exception as e:
        logger.error(f"Error in diagnostics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-graph")
def ask_graph(body: AskGraphBody):
    """
    Query existing documents using LangGraph pipeline with conditional routing.
    
    The graph allows agents to decide if they have sufficient evidence
    or need to iterate over query refinement and refine_retrieve options.
    """
    try:
        answer = ask_with_graph(body.question, thread_id=body.thread_id)
        return {
            "answer": answer,
            "mode": "query_only",
            "pipeline": "langgraph",
            "thread_id": body.thread_id
        }
    except Exception as e:
        logger.error(f"Error in /ask-graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer")
async def infer(
    question: str = Form(...),
    attachment: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None)
):
    """
    Combined ingestion and query endpoint using direct pipeline.
    
    If attachment is provided:
    - Ingest the attachment (PDF, TXT, PNG, JPEG)
    - Then query using the question
    
    If no attachment:
    - Just query existing documents (same as /ask)
    
    Uses agent_loop.py (direct pipeline without LangGraph).
    
    Supported file types:
    - PDF: Full text extraction with OCR fallback
    - TXT: Direct text ingestion
    - PNG/JPEG: Image captioning (extracts text via OCR/vision)
    """
    try:
        # If no attachment, just query existing documents
        if not attachment:
            answer = run_deep_rag(question)
            return {
                "answer": answer,
                "mode": "query_only",
                "pipeline": "direct",
                "attachment_processed": False
            }
        
        # Process attachment
        file_ext = Path(attachment.filename).suffix.lower() if attachment.filename else ""
        content_type = attachment.content_type or ""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await attachment.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Process based on file type
            if file_ext in ['.pdf'] or 'pdf' in content_type:
                ingest_pdf(tmp_path, title=title)
                logger.info(f"Ingested PDF: {attachment.filename}")
                
            elif file_ext in ['.txt'] or 'text/plain' in content_type:
                ingest_text_file(tmp_path, title=title or attachment.filename)
                logger.info(f"Ingested text file: {attachment.filename}")
                
            elif file_ext in ['.png', '.jpg', '.jpeg'] or 'image' in content_type:
                ingest_image(tmp_path, title=title or attachment.filename)
                logger.info(f"Ingested image: {attachment.filename}")
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG"
                )
            
            # After ingestion, run the query
            answer = run_deep_rag(question)
            
            return {
                "answer": answer,
                "mode": "ingest_and_query",
                "pipeline": "direct",
                "attachment_processed": True,
                "filename": attachment.filename,
                "file_type": file_ext or content_type
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /infer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer-graph")
async def infer_graph(
    question: str = Form(...),
    attachment: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    thread_id: Optional[str] = Form("default")
):
    """
    Combined ingestion and query endpoint using LangGraph pipeline.
    
    If attachment is provided:
    - Ingest the attachment (PDF, TXT, PNG, JPEG)
    - Then query using the question with LangGraph
    
    If no attachment:
    - Just query existing documents with LangGraph (same as /ask-graph)
    
    Supported file types:
    - PDF: Full text extraction with OCR fallback
    - TXT: Direct text ingestion
    - PNG/JPEG: Image captioning (extracts text via OCR/vision)
    """
    file_ext = None
    content_type = None
    
    try:
        # Process attachment if provided
        if attachment:
            file_ext = Path(attachment.filename).suffix.lower() if attachment.filename else ""
            content_type = attachment.content_type or ""
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                content = await attachment.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Process based on file type
                if file_ext in ['.pdf'] or 'pdf' in content_type:
                    ingest_pdf(tmp_path, title=title)
                    logger.info(f"Ingested PDF: {attachment.filename}")
                    
                elif file_ext in ['.txt'] or 'text/plain' in content_type:
                    ingest_text_file(tmp_path, title=title or attachment.filename)
                    logger.info(f"Ingested text file: {attachment.filename}")
                    
                elif file_ext in ['.png', '.jpg', '.jpeg'] or 'image' in content_type:
                    ingest_image(tmp_path, title=title or attachment.filename)
                    logger.info(f"Ingested image: {attachment.filename}")
                    
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG"
                    )
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        # Run the query with LangGraph
        answer = ask_with_graph(question, thread_id=thread_id)
        
        return {
            "answer": answer,
            "mode": "ingest_and_query" if attachment else "query_only",
            "pipeline": "langgraph",
            "thread_id": thread_id,
            "attachment_processed": attachment is not None,
            "filename": attachment.filename if attachment else None,
            "file_type": file_ext or content_type if attachment else None
        }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /infer-graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_endpoint(
    attachment: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    """
    Ingest a document into the vector database without querying.
    Supports PDF, TXT, PNG, JPEG.
    
    This endpoint only handles embedding and upsert - no agentic reasoning.
    Use /infer for ingestion + query with agentic reasoning.
    """
    try:
        file_ext = Path(attachment.filename).suffix.lower() if attachment.filename else ""
        content_type = attachment.content_type or ""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await attachment.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Process based on file type
            if file_ext in ['.pdf'] or 'pdf' in content_type:
                ingest_pdf(tmp_path, title=title)
                logger.info(f"Ingested PDF: {attachment.filename}")
                
            elif file_ext in ['.txt'] or 'text/plain' in content_type:
                ingest_text_file(tmp_path, title=title or attachment.filename)
                logger.info(f"Ingested text file: {attachment.filename}")
                
            elif file_ext in ['.png', '.jpg', '.jpeg'] or 'image' in content_type:
                ingest_image(tmp_path, title=title or attachment.filename)
                logger.info(f"Ingested image: {attachment.filename}")
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Supported: PDF, TXT, PNG, JPEG"
                )
            
            return {
                "status": "success",
                "filename": attachment.filename,
                "file_type": file_ext or content_type,
                "title": title or attachment.filename
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /ingest: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
def graph_export(out: str = "deep_rag_graph.png"):
    """
    Export the LangGraph as PNG (Graphviz) or Mermaid (.mmd fallback).
    
    Returns the path to the generated graph file.
    """
    try:
        from inference.graph.graph_viz import export_graph_png
        path = export_graph_png(out)
        return {
            "status": "success",
            "path": path,
            "format": "png" if path.endswith(".png") else "mermaid"
        }
    except Exception as e:
        logger.error(f"Error in /graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
