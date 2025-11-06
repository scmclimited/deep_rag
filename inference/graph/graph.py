from __future__ import annotations

from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass
import logging

from langgraph.graph import StateGraph, END  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

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

MAX_ITERS = 3
THRESH = 0.30   # matches your CE/lex+vec heuristic

# ---------- Node implementations ----------
def node_planner(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Planner - Decomposing question into sub-goals")
    logger.info("-" * 40)
    logger.info(f"Question: {state['question']}")
    
    prompt = f"""You are a planner. Decompose the user's question into 1-3 concrete sub-goals
that can be answered ONLY from the provided PDFs or assets. Prefer explicit nouns and constraints.
Question: {state['question']}"""
    plan = call_llm("You plan tasks.", [{"role": "user", "content": prompt}], max_tokens=200, temperature=0.2)
    plan_text = plan.strip()
    
    logger.info(f"Generated Plan: {plan_text}")
    logger.info("-" * 40)
    return {"plan": plan_text, "iterations": state.get("iterations", 0)}

def node_retriever(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Retriever - Fetching relevant chunks")
    logger.info("-" * 40)
    q = f"{state['question']}  {state.get('plan','')}"
    logger.info(f"Query: {q}")
    logger.info(f"Retrieval parameters: k=8, k_lex=40, k_vec=40")
    
    hits = retrieve_hybrid(q, k=8, k_lex=40, k_vec=40)
    # Merge with any prior evidence (e.g., from refinement loops)
    seen, merged = set(), []
    for h in (state.get("evidence", []) + hits):
        if h["chunk_id"] in seen:
            continue
        seen.add(h["chunk_id"]); merged.append(h)
    
    logger.info(f"Retrieved {len(hits)} new chunks, {len(merged)} total after merge")
    for i, hit in enumerate(merged[:5], 1):  # Log top 5
        logger.info(f"  [{i}] Chunk ID: {hit.get('chunk_id', 'N/A')[:8]}...")
        logger.info(f"      Pages: {hit.get('p0', 'N/A')}-{hit.get('p1', 'N/A')}")
        logger.info(f"      Scores: lex={hit.get('lex', 0):.4f}, vec={hit.get('vec', 0):.4f}, ce={hit.get('ce', 0):.4f}")
    logger.info("-" * 40)
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
    hits_all: List[Dict[str, Any]] = []
    for rq in refinements:
        logger.info(f"Retrieving for: {rq}")
        hits_all.extend(retrieve_hybrid(rq, k=6, k_lex=30, k_vec=30))
    
    logger.info(f"Retrieved {len(hits_all)} additional chunks from refinements")
    
    # Merge with existing evidence
    seen, merged = set(), []
    for h in (state.get("evidence", []) + hits_all):
        if h["chunk_id"] in seen:
            continue
        seen.add(h["chunk_id"]); merged.append(h)
    
    logger.info(f"Total evidence after merge: {len(merged)} chunks")
    logger.info("Routing back to compressor for re-compression")
    logger.info("-" * 40)
    return {"evidence": merged}

def node_synthesizer(state: GraphState) -> GraphState:
    logger.info("-" * 40)
    logger.info("GRAPH NODE: Synthesizer - Generating final answer")
    logger.info("-" * 40)
    logger.info(f"Using top {min(5, len(state.get('evidence', [])))} chunks for synthesis")
    
    ctx_evs = state.get("evidence", [])[:5]
    citations = [f"[{i}] p{h['p0']}–{h['p1']}" for i, h in enumerate(ctx_evs, 1)]
    context = "\n\n".join([f"[{i}] {h['text'][:1200]}" for i, h in enumerate(ctx_evs, 1)])
    prompt = f"""Answer the question using ONLY the context.
If insufficient evidence, say "I don't know."
Add bracket citations like [1], [2] that map to the provided context blocks.

Question: {state['question']}

Context:
{context}
"""
    ans = call_llm("You write precise, sourced answers.", [{"role":"user","content":prompt}], max_tokens=500, temperature=0.2)
    out = ans.strip()
    if citations:
        out += "\n\nSources: " + ", ".join(citations)
    
    logger.info(f"Generated Answer:\n{out}")
    logger.info("-" * 40)
    return {"answer": out}

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
