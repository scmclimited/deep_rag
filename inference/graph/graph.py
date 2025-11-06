from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional
import logging

from langgraph.graph import StateGraph, END  # type: ignore[import-untyped]
from inference.graph.agent_logger import get_agent_logger

logger = logging.getLogger(__name__)
agent_log = get_agent_logger()  # Initialize comprehensive agent logger

# Try to import SqliteSaver, fallback to None if not available
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    try:
        # Alternative import path for some langgraph versions
        from langgraph.checkpoint import SqliteSaver
    except ImportError:
        # If SQLite checkpoint not available, use in-memory or None
        SqliteSaver = None

from retrieval.retrieval import retrieve_hybrid
from inference.llm import call_llm

# ---------- State definition ----------
class GraphState(TypedDict, total=False):
    question: str
    plan: str
    evidence: List[Dict[str, Any]]
    notes: str
    answer: str
    confidence: float
    iterations: int
    refinements: List[str]
    doc_id: Optional[str]  # Optional document ID for document-specific retrieval

MAX_ITERS = 3
THRESH = 0.30   # matches your CE/lex+vec heuristic

# ---------- Node implementations ----------
def node_planner(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Planner - Decomposing question into sub-goals")
    logger.info("-" * 40)
    logger.info(f"Question: {state['question']}")
    doc_id = state.get('doc_id')
    if doc_id:
        logger.info(f"Planning for specific document: {doc_id[:8]}...")
    
    # Include doc_id context in prompt if available
    doc_context = ""
    if doc_id:
        doc_context = f"\n\nNote: This question is about a specific document that was just ingested. Focus your planning on this document's content."
    
    prompt = f"""You are a planner. Decompose the user's question into 1-3 concrete sub-goals
that can be answered ONLY from the provided assets such as PDFs, images, or other documents. Prefer explicit nouns and constraints.
Question: {state['question']}{doc_context}"""
    plan = call_llm("You plan tasks.", [{"role": "user", "content": prompt}], max_tokens=200, temperature=0.2)
    plan_text = plan.strip()
    
    logger.info(f"Generated Plan: {plan_text}")
    logger.info("-" * 40)
    
    # Log to agent logger for future training
    agent_log.log_step(
        node="planner",
        action="plan_generation",
        question=state['question'],
        plan=plan_text,
        iterations=state.get("iterations", 0)
    )
    
    return {"plan": plan_text, "iterations": state.get("iterations", 0)}

def node_retriever(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Retriever - Fetching relevant chunks")
    logger.info("-" * 40)
    q = f"{state['question']}  {state.get('plan','')}"
    doc_id = state.get('doc_id')
    logger.info(f"Query: {q}")
    if doc_id:
        logger.info(f"Filtering to document: {doc_id[:8]}...")
    logger.info(f"Retrieval parameters: k=8, k_lex=40, k_vec=40")
    
    hits = retrieve_hybrid(q, k=8, k_lex=40, k_vec=40, doc_id=doc_id)
    # Merge with any prior evidence (e.g., from refinement loops)
    seen, merged = set(), []
    for h in (state.get("evidence", []) + hits):
        if h["chunk_id"] in seen:
            continue
        seen.add(h["chunk_id"]); merged.append(h)
    
    logger.info(f"Retrieved {len(hits)} new chunks, {len(merged)} total after merge")
    for i, hit in enumerate(merged[:10], 1):  # Log top 10 for better visibility
        logger.info(f"  [{i}] Chunk ID: {hit.get('chunk_id', 'N/A')[:8]}...")
        logger.info(f"      Pages: {hit.get('p0', 'N/A')}-{hit.get('p1', 'N/A')}")
        logger.info(f"      Content Type: {hit.get('content_type', 'N/A')}")
        logger.info(f"      Scores: lex={hit.get('lex', 0):.4f}, vec={hit.get('vec', 0):.4f}, ce={hit.get('ce', 0):.4f}")
        # Show text preview (first 200 chars) to understand what was retrieved
        text_preview = hit.get('text', '')[:200] if hit.get('text') else 'N/A'
        logger.info(f"      Text preview: {text_preview}...")
    if len(merged) > 10:
        logger.info(f"  ... and {len(merged) - 10} more chunks")
    # Log page distribution to see if all pages are represented
    pages_found = sorted(set([h.get('p0', 0) for h in merged]))
    logger.info(f"Pages represented in retrieved chunks: {pages_found}")
    logger.info("-" * 40)
    
    # Log to agent logger with detailed retrieval info
    agent_log.log_step(
        node="retriever",
        action="retrieve",
        query=q,
        num_chunks=len(merged),
        pages=pages_found,
        metadata={
            "new_chunks": len(hits),
            "total_chunks": len(merged),
            "top_scores": [
                {
                    "lex": h.get('lex', 0),
                    "vec": h.get('vec', 0),
                    "ce": h.get('ce', 0)
                } for h in merged[:5]
            ]
        }
    )
    
    # Log detailed retrieval results for analysis
    agent_log.log_retrieval_details(
        session_id="current",
        query=q,
        chunks=merged
    )
    
    return {"evidence": merged}

def node_compressor(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Compressor - Summarizing evidence")
    logger.info("-" * 40)
    logger.info(f"Compressing {len(state.get('evidence', []))} chunks into notes...")
    
    snippets = "\n\n".join([f"[p{h['p0']}–{h['p1']}] {h['text'][:1200]}" for h in state.get("evidence", [])])
    prompt = f"""Summarize the following snippets into crisp notes with bullets.
Retain numbers and proper nouns verbatim. Avoid speculation.
Snippets:\n{snippets}"""
    notes = call_llm("You compress evidence.", [{"role": "user", "content": prompt}], max_tokens=300, temperature=0.2)
    notes_text = notes.strip()
    
    logger.info(f"Compressed Notes:\n{notes_text}")
    logger.info("-" * 40)
    
    # Log compression step
    agent_log.log_step(
        node="compressor",
        action="compress",
        num_chunks=len(state.get('evidence', [])),
        metadata={"notes_length": len(notes_text)}
    )
    
    return {"notes": notes_text}

def node_critic(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Critic - Evaluating evidence quality")
    logger.info("-" * 40)
    
    ev = state.get("evidence", [])
    strong = sum(1 for h in ev if float(h.get("ce", 0.0)) > THRESH or (h.get("lex",0)>0 and h.get("vec",0)>0))
    conf = min(0.9, 0.4 + 0.1*strong)

    result: GraphState = {"confidence": conf, "iterations": state.get("iterations", 0)}

    logger.info(f"Strong chunks: {strong}/{len(ev)}")
    logger.info(f"Confidence score: {conf:.2f}")
    logger.info(f"Iterations: {state.get('iterations', 0)}/{MAX_ITERS}")
    
    # Log critic evaluation
    agent_log.log_step(
        node="critic",
        action="evaluate",
        confidence=conf,
        iterations=state.get('iterations', 0),
        metadata={
            "strong_chunks": strong,
            "total_chunks": len(ev),
            "threshold": THRESH
        }
    )

    # If weak confidence and not at loop cap, propose refinements (sub-queries)
    if conf < 0.6 and state.get("iterations", 0) < MAX_ITERS:
        logger.info(f"Confidence {conf:.2f} < 0.6 threshold - Requesting refinement...")
        prompt = f"""Given the plan:\n{state.get('plan','')}\nAnd notes:\n{state.get('notes','')}\n
Propose refined sub-queries (max 2) to retrieve missing evidence. Short, 1 line each.

IMPORTANT: Write queries as natural language questions without special characters like &, *, |, !, :, or quotes. 
Use plain text only. For example, write "Hygiene and DX" instead of "Hygiene & DX"."""
        refinements = call_llm("You suggest refinements.", [{"role":"user","content":prompt}], max_tokens=120, temperature=0.0)
        lines = [ln.strip("-• ").strip() for ln in refinements.splitlines() if ln.strip()]
        # Additional sanitization: remove any remaining special characters
        sanitized_lines = []
        for line in lines:
            # Remove leading special chars and normalize
            cleaned = line.strip()
            # Replace & with "and" if present
            cleaned = cleaned.replace('&', ' and ')
            # Remove other problematic characters
            import re
            cleaned = re.sub(r'[\!\|\:\*\"]', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                sanitized_lines.append(cleaned)
        
        result["refinements"] = sanitized_lines[:2] if sanitized_lines else []
        result["iterations"] = state.get("iterations", 0) + 1
        
        logger.info(f"Refinements: {result['refinements']}")
        logger.info("Routing to refine_retrieve node")
        
        # Log refinement decision
        agent_log.log_step(
            node="critic",
            action="request_refinement",
            confidence=conf,
            iterations=result["iterations"],
            refinements=result["refinements"]
        )
    else:
        result["refinements"] = []
        if conf >= 0.6:
            logger.info(f"Confidence {conf:.2f} >= 0.6 - Routing to synthesizer")
        else:
            logger.warning(f"Max iterations ({MAX_ITERS}) reached with confidence {conf:.2f}")
    logger.info("-" * 40)
    return result

def node_refine_retrieve(state: GraphState) -> GraphState:
    """Optional additional retrieve step driven by critic's refinements."""
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Refine Retriever - Fetching additional chunks from refinements")
    logger.info("-" * 40)
    
    refinements = state.get("refinements", [])
    if not refinements:
        logger.info("No refinements provided, skipping refinement retrieval")
        logger.info("-" * 40)
        return {}
    
    logger.info(f"Refinement queries: {refinements}")
    doc_id = state.get('doc_id')
    hits_all: List[Dict[str, Any]] = []
    for rq in refinements:
        logger.info(f"Retrieving for: {rq}")
        hits = retrieve_hybrid(rq, k=6, k_lex=30, k_vec=30, doc_id=doc_id)
        hits_all.extend(hits)
        
        # Log each refinement query
        agent_log.log_step(
            node="refine_retrieve",
            action="refine_query",
            query=rq,
            num_chunks=len(hits),
            pages=sorted(set([h.get('p0', 0) for h in hits]))
        )
    
    logger.info(f"Retrieved {len(hits_all)} additional chunks from refinements")
    
    # Log retrieved chunks with text preview
    for i, hit in enumerate(hits_all[:5], 1):
        logger.info(f"  Refinement [{i}] Pages: {hit.get('p0', 'N/A')}-{hit.get('p1', 'N/A')}")
        text_preview = hit.get('text', '')[:150] if hit.get('text') else 'N/A'
        logger.info(f"      Text preview: {text_preview}...")
    
    # Merge with existing evidence
    seen, merged = set(), []
    for h in (state.get("evidence", []) + hits_all):
        if h["chunk_id"] in seen:
            continue
        seen.add(h["chunk_id"]); merged.append(h)
    
    logger.info(f"Total evidence after merge: {len(merged)} chunks")
    # Log page distribution after merge
    pages_found = sorted(set([h.get('p0', 0) for h in merged]))
    logger.info(f"Pages represented after merge: {pages_found}")
    logger.info("Routing back to compressor for re-compression")
    logger.info("-" * 40)
    
    # Log refinement retrieval summary
    agent_log.log_step(
        node="refine_retrieve",
        action="merge_results",
        num_chunks=len(merged),
        pages=pages_found,
        metadata={
            "refinement_chunks": len(hits_all),
            "total_after_merge": len(merged)
        }
    )
    
    return {"evidence": merged}

def node_synthesizer(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Synthesizer - Generating final answer")
    logger.info("-" * 40)
    logger.info(f"Using top {min(5, len(state.get('evidence', [])))} chunks for synthesis")
    
    doc_id = state.get('doc_id')
    if doc_id:
        logger.info(f"Synthesizing answer for specific document: {doc_id[:8]}...")
    
    ctx_evs = state.get("evidence", [])[:5]
    
    # Identify doc_ids from retrieved chunks if not already set
    if not doc_id and ctx_evs:
        doc_ids_found = set(h.get('doc_id') for h in ctx_evs if h.get('doc_id'))
        if doc_ids_found:
            logger.info(f"Identified {len(doc_ids_found)} document(s) from retrieved chunks: {[d[:8] for d in doc_ids_found]}")
            # Use the most common doc_id if multiple found
            if len(doc_ids_found) == 1:
                doc_id = list(doc_ids_found)[0]
                logger.info(f"Using document ID: {doc_id[:8]}...")
    
    citations = [f"[{i}] p{h['p0']}–{h['p1']}" for i, h in enumerate(ctx_evs, 1)]
    # Log which chunks are being used for synthesis
    logger.info("Chunks used for synthesis:")
    for i, h in enumerate(ctx_evs, 1):
        chunk_doc_id = h.get('doc_id', 'N/A')
        logger.info(f"  [{i}] Doc: {chunk_doc_id[:8] if chunk_doc_id != 'N/A' else 'N/A'}... Pages {h['p0']}–{h['p1']}: {h.get('text', '')[:100]}...")
    context = "\n\n".join([f"[{i}] {h['text'][:1200]}" for i, h in enumerate(ctx_evs, 1)])
    
    # Include doc_id context in prompt if available
    doc_context = ""
    if doc_id:
        doc_context = f"\n\nNote: This answer is based on a specific document that was recently ingested or identified from the knowledge base. Focus your answer on this document's content."
    
    prompt = f"""Answer the question using ONLY the context.
If insufficient evidence, or the result is likely not in the context, say "I don't know."
Add bracket citations like [1], [2] that map to the provided context blocks and snippets of text used from source documents.
Which can include exact verbatim text from source documents or image descriptions.{doc_context}

Question: {state['question']}

Context:
{context}
"""
    ans = call_llm("You write precise, sourced answers.", [{"role":"user","content":prompt}], max_tokens=500, temperature=0.2)
    out = ans.strip()
    if citations:
        out += "\n\nSources: " + ", ".join(citations)
    
    # Update state with doc_id if identified
    result = {"answer": out}
    if doc_id:
        result["doc_id"] = doc_id
        logger.info(f"Answer generated for document: {doc_id[:8]}...")
    
    logger.info(f"Generated Answer:\n{out}")
    logger.info("-" * 40)
    
    # Log final synthesis
    agent_log.log_step(
        node="synthesizer",
        action="synthesize",
        question=state['question'],
        answer=out,
        num_chunks=len(ctx_evs),
        pages=sorted(set([h['p0'] for h in ctx_evs])),
        confidence=state.get('confidence', 0.0),
        iterations=state.get('iterations', 0),
        metadata={
            "citations": citations,
            "answer_length": len(out),
            "doc_id": doc_id[:8] if doc_id else None
        }
    )
    
    return result

# ---------- Conditional routing ----------
def should_refine(state: GraphState) -> str:
    """Edge function: either loop to refine_retrieve or end at synthesizer."""
    if state.get("confidence", 0.0) < 0.6 and state.get("iterations", 0) <= MAX_ITERS and state.get("refinements"):
        return "refine"
    return "synthesize"

# ---------- Build the graph ----------
def build_app(sqlite_path: str = "langgraph_state.sqlite"):
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("planner", node_planner)
    graph.add_node("retriever", node_retriever)
    graph.add_node("compressor", node_compressor)
    graph.add_node("critic", node_critic)
    graph.add_node("refine_retrieve", node_refine_retrieve)
    graph.add_node("synthesizer", node_synthesizer)

    # Edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "compressor")
    graph.add_edge("compressor", "critic")
    graph.add_conditional_edges("critic", should_refine, {
        "refine": "refine_retrieve",
        "synthesize": "synthesizer"
    })
    # After refine retrieval, go back to compressor → critic again
    graph.add_edge("refine_retrieve", "compressor")
    graph.add_edge("synthesizer", END)

    # Persistence (per-thread history/checkpoints)
    if SqliteSaver is not None:
        try:
            checkpointer = SqliteSaver.from_conn_string(sqlite_path)
            app = graph.compile(checkpointer=checkpointer)
        except Exception as e:
            # Fallback to no checkpoint if SQLite fails
            print(f"Warning: Could not initialize SQLite checkpoint: {e}. Using in-memory mode.")
            app = graph.compile()
    else:
        # No checkpoint available, use in-memory mode
        app = graph.compile()
    return app
