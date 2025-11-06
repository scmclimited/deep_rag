from inference.graph.graph import build_app
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def ask_with_graph(question: str, thread_id: str = "default", doc_id: Optional[str] = None) -> str:
    """
    Query using LangGraph pipeline with conditional routing.
    
    The graph allows agents to decide if they have sufficient evidence
    or need to iterate over query refinement and refine_retrieve options.
    
    Args:
        question: The question to ask
        thread_id: Optional thread ID for conversation state (default: "default")
        doc_id: Optional document ID to filter retrieval to a specific document
        
    Returns:
        The final answer from the graph pipeline
    """
    logger.info("-" * 40)
    logger.info("LANGRAPH PIPELINE STARTED")
    logger.info("-" * 40)
    logger.info(f"Question: {question}")
    logger.info(f"Thread ID: {thread_id}")
    if doc_id:
        logger.info(f"Document filter: {doc_id[:8]}...")
    
    app = build_app()  # uses ./langgraph_state.sqlite
    # thread_id lets you keep state per ongoing conversation (optional for this pipeline)
    initial_state = {
        "question": question, 
        "plan": "", 
        "evidence": [], 
        "notes": "", 
        "answer": "", 
        "confidence": 0.0, 
        "iterations": 0, 
        "refinements": []
    }
    if doc_id:
        initial_state["doc_id"] = doc_id
    
    resp = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}}
    )
    
    # Log final state
    logger.info("-" * 40)
    logger.info("LANGRAPH PIPELINE COMPLETED")
    logger.info("-" * 40)
    logger.info(f"Final Confidence: {resp.get('confidence', 0.0):.2f}")
    logger.info(f"Total Iterations: {resp.get('iterations', 0)}")
    logger.info(f"Total Evidence Chunks: {len(resp.get('evidence', []))}")
    
    # Log page distribution in final evidence
    evidence = resp.get('evidence', [])
    if evidence:
        pages_found = sorted(set([h.get('p0', 0) for h in evidence]))
        logger.info(f"Pages in final evidence: {pages_found}")
    logger.info("-" * 40)
    
    # app.invoke returns the final state; pull the answer:
    return resp.get("answer", "")

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What does the document say about model architecture?"
    print(ask_with_graph(q, thread_id="cli-demo"))
