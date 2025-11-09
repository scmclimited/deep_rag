from inference.graph.builder import build_app
import logging
import unicodedata
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

def ask_with_graph(question: str, thread_id: str = "default", doc_id: Optional[str] = None, 
                  selected_doc_ids: Optional[list[str]] = None, cross_doc: bool = False) -> dict:
    """
    Query using LangGraph pipeline with conditional routing.
    
    The graph allows agents to decide if they have sufficient evidence
    or need to iterate over query refinement and refine_retrieve options.
    
    Args:
        question: The question to ask
        thread_id: Optional thread ID for conversation state (default: "default")
        doc_id: Optional document ID to filter retrieval to a specific document
        cross_doc: If True, enable cross-document retrieval (two-stage when doc_id provided)
        
    Returns:
        Dictionary with answer, confidence, action, and other metadata
    """
    logger.info("-" * 40)
    logger.info("LANGRAPH PIPELINE STARTED")
    logger.info("-" * 40)
    logger.info(f"Question: {question}")
    logger.info(f"Thread ID: {thread_id}")
    
    # Handle multi-document selection or single doc_id
    # Priority: selected_doc_ids (explicit user selection) > doc_id (from ingestion/previous query)
    # If both are provided, combine them (user may have selected other docs in addition to ingested doc)
    # CRITICAL: Track if selected_doc_ids was explicitly provided (even if empty) to override persisted state
    selected_doc_ids_explicitly_provided = selected_doc_ids is not None
    doc_ids_to_use = None
    if selected_doc_ids is not None:
        # selected_doc_ids was explicitly provided (could be empty list)
        if len(selected_doc_ids) > 0:
            doc_ids_to_use = list(selected_doc_ids)  # Make a copy to avoid modifying original
            
            # If doc_id is also provided and not already in selected_doc_ids, add it
            # This handles the case where user ingested a doc AND selected other docs
            if doc_id and doc_id not in doc_ids_to_use:
                doc_ids_to_use.append(doc_id)
                logger.info(f"Combining selected_doc_ids with doc_id: {len(doc_ids_to_use)} document(s) total")
        # If empty list, doc_ids_to_use stays None (user deselected all)
    elif doc_id:
        # Fallback to doc_id only if selected_doc_ids was not provided (None)
        doc_ids_to_use = [doc_id]
    
    if doc_ids_to_use:
        if len(doc_ids_to_use) > 1:
            logger.info(f"Multi-document selection: {len(doc_ids_to_use)} document(s)")
        else:
            logger.info(f"Document filter: {doc_ids_to_use[0]}...")
    elif not cross_doc:
        logger.info("No documents selected and cross_doc=False - will return empty results")
    if cross_doc:
        logger.info("Cross-document retrieval enabled")
    
    app = build_app()  # uses ./langgraph_state.sqlite
    # thread_id lets you keep state per ongoing conversation (optional for this pipeline)
    # CRITICAL: Explicitly clear doc_id and selected_doc_ids to prevent using persisted state
    # LangGraph persists state between queries, so we must explicitly set these to None/[] 
    # to prevent using documents from previous queries
    initial_state = {
        "question": question, 
        "plan": "", 
        "evidence": [], 
        "notes": "", 
        "answer": "", 
        "confidence": 0.0, 
        "iterations": 0, 
        "refinements": [],
        "doc_ids": [],
        "cross_doc": cross_doc,
        "doc_id": None,  # Explicitly clear to prevent using persisted doc_id
        "selected_doc_ids": None  # Explicitly clear to prevent using persisted selected_doc_ids
    }
    # Set doc_id and selected_doc_ids based on doc_ids_to_use
    # CRITICAL: Explicitly set these values to override any persisted state from previous queries
    # LangGraph merges initial_state with persisted state, so we must explicitly set values
    if doc_ids_to_use and len(doc_ids_to_use) > 0:
        initial_state["doc_id"] = doc_ids_to_use[0]
        initial_state["selected_doc_ids"] = doc_ids_to_use  # New multi-document support
    elif selected_doc_ids_explicitly_provided:
        # User explicitly provided empty selected_doc_ids (deselected all)
        # Explicitly set to empty list to override persisted state
        initial_state["selected_doc_ids"] = []  # Explicitly empty
        initial_state["doc_id"] = None  # Explicitly clear doc_id
    # If doc_ids_to_use is None and selected_doc_ids was not explicitly provided, 
    # both doc_id and selected_doc_ids remain None (explicitly cleared in initial_state above)
    
    resp = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}}
    )
    
    # Log final state
    logger.info("-" * 40)
    logger.info("LANGRAPH PIPELINE COMPLETED")
    logger.info("-" * 40)
    final_confidence = resp.get('confidence', 0.0)
    iterations = resp.get('iterations', 0)
    refinements = resp.get('refinements', [])
    logger.info(f"User-facing confidence (percentage): {final_confidence:.2f}%")
    logger.info(f"Total Iterations Executed: {iterations}")
    logger.info(f"Refinement prompts issued: {len(refinements)}")
    if refinements:
        logger.info(f"Refinement history: {refinements}")
    logger.info(f"Total Evidence Chunks: {len(resp.get('evidence', []))}")
    
    # Log page distribution in final evidence
    evidence = resp.get('evidence', [])
    if evidence:
        pages_found = sorted(set([h.get('p0', 0) for h in evidence]))
        logger.info(f"Pages in final evidence: {pages_found}")
    logger.info("-" * 40)
    
    # Extract page references and doc_ids from evidence
    evidence = resp.get('evidence', [])
    pages: List[str] = []
    doc_order: List[str] = []
    doc_counts: Dict[str, int] = {}
    if evidence:
        for ev in evidence[:10]:  # Top 10 chunks for better coverage
            p0 = ev.get('p0')
            p1 = ev.get('p1')
            ev_doc_id = ev.get('doc_id')
            if ev_doc_id:
                doc_counts[ev_doc_id] = doc_counts.get(ev_doc_id, 0) + 1
                if ev_doc_id not in doc_order:
                    doc_order.append(ev_doc_id)
            # Add page range if available
            if p0 is not None:
                if p1 is not None and p1 != p0:
                    pages.append(f"{p0}-{p1}")
                else:
                    pages.append(str(p0))

    ranked_doc_ids: List[str] = []
    if doc_counts:
        ranked_doc_ids = sorted(
            doc_counts.keys(),
            key=lambda doc: (
                -doc_counts.get(doc, 0),
                doc_order.index(doc) if doc in doc_order else len(doc_order)
            )
        )
    else:
        ranked_doc_ids = list(dict.fromkeys(doc_order))

    if doc_ids_to_use:
        for doc_id in doc_ids_to_use:
            if doc_id and doc_id not in ranked_doc_ids:
                ranked_doc_ids.append(doc_id)

    max_docs_to_report = 5
    final_doc_ids = ranked_doc_ids[:max_docs_to_report]
    if not final_doc_ids and doc_ids_to_use:
        final_doc_ids = [doc for doc in doc_ids_to_use if doc][:max_docs_to_report]

    primary_doc_id = final_doc_ids[0] if final_doc_ids else None
    logger.info(f"Document ranking (top {max_docs_to_report}): {final_doc_ids}")
    logger.info(f"Final graph action: {resp.get('action', 'answer')} with iterations={iterations}")

    answer_text = resp.get("answer", "")
    normalized_answer = unicodedata.normalize("NFKD", answer_text or "")
    answer_normalized = normalized_answer.strip().lower()
    answer_ascii = (
        normalized_answer.encode("ascii", "ignore").decode("ascii").strip().lower()
        if normalized_answer
        else ""
    )
    if (
        answer_normalized.startswith("i don't know")
        or answer_normalized.startswith("i do not know")
        or "i don't know" in answer_normalized.splitlines()[0]
        or answer_ascii.startswith("i dont know")
        or "i dont know" in answer_ascii.splitlines()[0]
    ):
        final_doc_ids = []
        primary_doc_id = None
        pages = []
    
    # Return full state with answer, confidence, action, and metadata
    return {
        "answer": answer_text,
        "confidence": resp.get("confidence", 0.0),
        "action": resp.get("action", "answer"),
        "doc_id": primary_doc_id,
        "doc_ids": final_doc_ids,
        "pages": sorted(set(pages)) if pages else []  # Unique sorted pages
    }

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "In summary, what does the document say?"
    result = ask_with_graph(q, thread_id="cli-demo")
    print(result.get("answer", ""))
