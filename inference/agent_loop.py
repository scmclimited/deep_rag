# agent_graph.py
from typing import TypedDict, List, Optional
import logging
from retrieval.retrieval import retrieve_hybrid
# from sentence_transformers import CrossEncoder
from inference.llm import call_llm  
# Assume you provide an LLM function like:
# def call_llm(system: str, messages: List[dict], max_tokens=512) -> str: ...

logger = logging.getLogger(__name__)

class State(TypedDict, total=False):
    question: str
    plan: str
    evidence: List[dict]
    notes: str
    answer: str
    confidence: float
    iterations: int
    doc_id: Optional[str]  # Optional document ID for document-specific retrieval

MAX_ITERS = 3
THRESH = 0.30  # require ≥2 strong chunks via CE or lex/vec hybrid

def planner(state: State) -> State:
    """Planner agent: Decomposes the question into sub-goals."""
    logger.info("-" * 40)
    logger.info("AGENT: Planner - Decomposing question into sub-goals")
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
that can be answered ONLY from the provided context. Prefer explicit nouns and constraints.
Question: {state['question']}{doc_context}"""
    plan = call_llm("You plan tasks for the given question.", [{"role":"user","content":prompt}], max_tokens=350)
    state["plan"] = plan.strip()
    
    logger.info(f"Generated Plan: {state['plan']}")
    logger.info("-" * 80)
    return state

def retriever_agent(state: State) -> State:
    """Retriever agent: Fetches relevant chunks from the vector database."""
    logger.info("-" * 40)
    logger.info("AGENT: Retriever - Fetching relevant chunks")
    logger.info("-" * 40)
    q = f"{state['question']}  {state['plan']}"
    doc_id = state.get('doc_id')
    logger.info(f"Query: {q}")
    if doc_id:
        logger.info(f"Filtering to document: {doc_id[:8]}...")
    logger.info(f"Retrieval parameters: k=8, k_lex=40, k_vec=40")
    
    hits = retrieve_hybrid(q, k=8, k_lex=40, k_vec=40, doc_id=doc_id)
    state["evidence"] = hits
    
    logger.info(f"Retrieved {len(hits)} chunks:")
    for i, hit in enumerate(hits[:10], 1):  # Log top 10 for better visibility
        logger.info(f"  [{i}] Chunk ID: {hit.get('chunk_id', 'N/A')[:8]}...")
        logger.info(f"      Pages: {hit.get('p0', 'N/A')}-{hit.get('p1', 'N/A')}")
        logger.info(f"      Content Type: {hit.get('content_type', 'N/A')}")
        logger.info(f"      Scores: lex={hit.get('lex', 0):.4f}, vec={hit.get('vec', 0):.4f}, ce={hit.get('ce', 0):.4f}")
        # Show more text preview (200 chars) to understand what was retrieved
        text_preview = hit.get('text', '')[:200] if hit.get('text') else 'N/A'
        logger.info(f"      Text preview: {text_preview}...")
    if len(hits) > 10:
        logger.info(f"  ... and {len(hits) - 10} more chunks")
    # Log page distribution to see if all pages are represented
    pages_found = sorted(set([h.get('p0', 0) for h in hits]))
    logger.info(f"Pages represented in retrieved chunks: {pages_found}")
    logger.info("-" * 40)
    return state

def compressor(state: State) -> State:
    """Compressor agent: Summarizes retrieved evidence into concise notes."""
    logger.info("-" * 40)
    logger.info("AGENT: Compressor - Summarizing evidence")
    logger.info("-" * 40)
    logger.info(f"Compressing {len(state['evidence'])} chunks into notes...")
    
    # Map-reduce style compression of top evidence
    snippets = "\n\n".join([f"[p{h['p0']}–{h['p1']}] {h['text'][:1200]}" for h in state["evidence"]])
    prompt = f"""Summarize the following snippets into crisp notes with bullets.
Retain numbers and proper nouns verbatim. Avoid speculation.
Snippets:\n{snippets}"""
    notes = call_llm("You compress evidence.", [{"role":"user","content":prompt}], max_tokens=300)
    state["notes"] = notes.strip()
    
    logger.info(f"Compressed Notes:\n{state['notes']}")
    logger.info("-" * 80)
    return state

def critic(state: State) -> State:
    """Critic agent: Evaluates evidence quality and triggers refinement if needed."""
    logger.info("-" * 40)
    logger.info("AGENT: Critic - Evaluating evidence quality")
    logger.info("-" * 40)
    
    ev = state["evidence"]
    strong = sum(1 for h in ev if h.get("ce", 0.0) > THRESH or (h["lex"]>0 and h["vec"]>0))
    conf = min(0.9, 0.4 + 0.1*strong)  # toy heuristic; plug in your own
    state["confidence"] = conf
    
    logger.info(f"Strong chunks: {strong}/{len(ev)}")
    logger.info(f"Confidence score: {conf:.2f}")
    logger.info(f"Iterations: {state['iterations']}/{MAX_ITERS}")
    
    if conf < 0.6 and state["iterations"] < MAX_ITERS:
        logger.info(f"Confidence {conf:.2f} < 0.6 threshold - Requesting refinement...")
        # Ask for refinement: new sub-questions or different keywords
        prompt = f"""Given the plan:\n{state['plan']}\nAnd notes:\n{state['notes']}\n
Propose refined sub-queries (max 2) to retrieve missing evidence. Short, 1 line each.

IMPORTANT: Write queries as natural language questions without special characters like &, *, |, !, :, or quotes. 
Use plain text only. For example, write "Hygiene and DX" instead of "Hygiene & DX"."""
        refinements = call_llm("You suggest refinements for the given question and plan.", [{"role":"user","content":prompt}], max_tokens=120)
        # Re-query once with the first refinement
        rq_raw = refinements.splitlines()[0].strip("-• ").strip()
        # Sanitize the refinement query
        import re
        rq = rq_raw.replace('&', ' and ')
        rq = re.sub(r'[\!\|\:\*\"]', ' ', rq)
        rq = re.sub(r'\s+', ' ', rq).strip()
        logger.info(f"Refinement query: {rq}")
        
        doc_id = state.get('doc_id')
        hits = retrieve_hybrid(rq, k=6, k_lex=30, k_vec=30, doc_id=doc_id)
        logger.info(f"Retrieved {len(hits)} additional chunks from refinement")
        
        # Merge and dedup by chunk_id
        seen, merged = set(), []
        for h in state["evidence"] + hits:
            if h["chunk_id"] in seen: continue
            seen.add(h["chunk_id"]); merged.append(h)
        state["evidence"] = merged
        state["iterations"] += 1
        
        logger.info(f"Total evidence after merge: {len(state['evidence'])} chunks")
        logger.info("-" * 80)
        return critic(state)   # one self-loop step (bounded by MAX_ITERS)
    else:
        if conf >= 0.6:
            logger.info(f"Confidence {conf:.2f} >= 0.6 - Proceeding to synthesis")
        else:
            logger.warning(f"Max iterations ({MAX_ITERS}) reached with confidence {conf:.2f}")
    logger.info("-" * 80)
    return state

def synthesizer(state: State) -> State:
    """Synthesizer agent: Generates final answer from evidence."""
    logger.info("-" * 40)
    logger.info("AGENT: Synthesizer - Generating final answer")
    logger.info("-" * 40)
    logger.info(f"Using top {min(5, len(state['evidence']))} chunks for synthesis")
    
    doc_id = state.get('doc_id')
    if doc_id:
        logger.info(f"Synthesizing answer for specific document: {doc_id[:8]}...")
    
    citations = []
    chunks_used = state["evidence"][:5]
    
    # Identify doc_ids from retrieved chunks if not already set
    if not doc_id and chunks_used:
        doc_ids_found = set(h.get('doc_id') for h in chunks_used if h.get('doc_id'))
        if doc_ids_found:
            logger.info(f"Identified {len(doc_ids_found)} document(s) from retrieved chunks: {[d[:8] for d in doc_ids_found]}")
            # Use the most common doc_id if multiple found
            if len(doc_ids_found) == 1:
                doc_id = list(doc_ids_found)[0]
                logger.info(f"Using document ID: {doc_id[:8]}...")
                state["doc_id"] = doc_id
    
    # Log which chunks are being used for synthesis
    logger.info("Chunks used for synthesis:")
    for i, h in enumerate(chunks_used, 1):
        chunk_doc_id = h.get('doc_id', 'N/A')
        logger.info(f"  [{i}] Doc: {chunk_doc_id[:8] if chunk_doc_id != 'N/A' else 'N/A'}... Pages {h['p0']}–{h['p1']}: {h.get('text', '')[:100]}...")
    for i, h in enumerate(chunks_used, 1):
        citations.append(f"[{i}] p{h['p0']}–{h['p1']}")
    context = "\n\n".join([f"[{i}] {h['text'][:1200]}" for i, h in enumerate(chunks_used, 1)])
    
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
    ans = call_llm("You write precise, sourced answers.", [{"role":"user","content":prompt}], max_tokens=500)
    state["answer"] = ans.strip() + "\n\nSources: " + ", ".join(citations)
    
    logger.info(f"Generated Answer:\n{state['answer']}")
    logger.info("-" * 40)
    return state

def run_deep_rag(question: str, doc_id: Optional[str] = None) -> str:
    """
    Main entry point for Deep RAG pipeline.
    
    Args:
        question: The question to ask
        doc_id: Optional document ID to filter retrieval to a specific document
        
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
        "iterations": 0
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
