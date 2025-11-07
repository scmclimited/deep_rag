"""
Main pipeline for direct agent loop.
"""
import logging
from typing import Optional
from inference.agents.state import State
from inference.agents.planner import planner
from inference.agents.retriever import retriever_agent
from inference.agents.compressor import compressor
from inference.agents.critic import critic
from inference.agents.synthesizer import synthesizer

logger = logging.getLogger(__name__)


def run_deep_rag(question: str, doc_id: Optional[str] = None, cross_doc: bool = False) -> str:
    """
    Main entry point for Deep RAG pipeline.
    
    Args:
        question: The question to ask
        doc_id: Optional document ID to filter retrieval to a specific document
        cross_doc: If True, enable cross-document retrieval (two-stage when doc_id provided)
        
    Returns:
        The answer string
    """
    logger.info("-" * 40)
    logger.info("DEEP RAG PIPELINE STARTED")
    logger.info("-" * 40)
    logger.info(f"Question: {question}")
    if doc_id:
        logger.info(f"Document filter: {doc_id[:8]}...")
    logger.info("")
    
    state: State = {
        "question": question, 
        "plan": "", 
        "evidence": [], 
        "notes": "", 
        "answer": "", 
        "confidence": 0.0, 
        "iterations": 0,
        "doc_ids": [],
        "cross_doc": cross_doc
    }
    if doc_id:
        state["doc_id"] = doc_id
    
    # Execute pipeline stages
    pipeline_stages = [
        ("Planner", planner),
        ("Retriever", retriever_agent),
        ("Compressor", compressor),
        ("Critic", critic),
        ("Synthesizer", synthesizer)
    ]
    
    for stage_name, stage_fn in pipeline_stages:
        logger.info(f"\n>>> Stage: {stage_name}")
        try:
            state = stage_fn(state)
        except Exception as e:
            logger.error(f"Error in {stage_name} stage: {e}", exc_info=True)
            raise
    
    logger.info("")
    logger.info("-" * 40)
    logger.info("DEEP RAG PIPELINE COMPLETED")
    logger.info("-" * 40)
    logger.info(f"Final Confidence: {state['confidence']:.2f}")
    logger.info(f"Total Iterations: {state['iterations']}")
    logger.info(f"Total Evidence Chunks: {len(state['evidence'])}")
    logger.info("-" * 40)
    
    return state["answer"]

