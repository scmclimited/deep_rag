# agent_graph.py
from typing import TypedDict, List
from dataclasses import dataclass
import logging
from retrieval.retrieval import retrieve_hybrid
# from sentence_transformers import CrossEncoder
from inference.llm import call_llm  
# Assume you provide an LLM function like:
# def call_llm(system: str, messages: List[dict], max_tokens=512) -> str: ...

logger = logging.getLogger(__name__)

class State(TypedDict):
    question: str
    plan: str
    evidence: List[dict]
    notes: str
    answer: str
    confidence: float
    iterations: int

MAX_ITERS = 3
THRESH = 0.30  # require ≥2 strong chunks via CE or lex/vec hybrid

def planner(state: State) -> State:
    """Planner agent: Decomposes the question into sub-goals."""
    logger.info("-" * 40)
    logger.info("AGENT: Planner - Decomposing question into sub-goals")
    logger.info("-" * 40)
    logger.info(f"Question: {state['question']}")
    
    prompt = f"""You are a planner. Decompose the user's question into 1-3 concrete sub-goals
that can be answered ONLY from the provided context. Prefer explicit nouns and constraints.
Question: {state['question']}"""
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
    logger.info(f"Query: {q}")
    logger.info(f"Retrieval parameters: k=8, k_lex=40, k_vec=40")
    
    hits = retrieve_hybrid(q, k=8, k_lex=40, k_vec=40)
    state["evidence"] = hits
    
    logger.info(f"Retrieved {len(hits)} chunks:")
    for i, hit in enumerate(hits[:5], 1):  # Log top 5
        logger.info(f"  [{i}] Chunk ID: {hit.get('chunk_id', 'N/A')[:8]}...")
        logger.info(f"      Pages: {hit.get('p0', 'N/A')}-{hit.get('p1', 'N/A')}")
        logger.info(f"      Content Type: {hit.get('content_type', 'N/A')}")
        logger.info(f"      Scores: lex={hit.get('lex', 0):.4f}, vec={hit.get('vec', 0):.4f}, ce={hit.get('ce', 0):.4f}")
        logger.info(f"      Text preview: {hit.get('text', '')[:100]}...")
    if len(hits) > 5:
        logger.info(f"  ... and {len(hits) - 5} more chunks")
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
        
        hits = retrieve_hybrid(rq, k=6, k_lex=30, k_vec=30)
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
    
    citations = []
    for i, h in enumerate(state["evidence"][:5], 1):
        citations.append(f"[{i}] p{h['p0']}–{h['p1']}")
    context = "\n\n".join([f"[{i}] {h['text'][:1200]}" for i, h in enumerate(state["evidence"][:5], 1)])
    prompt = f"""Answer the question using ONLY the context.
If insufficient evidence, say "I don't know."
Add bracket citations like [1], [2] that map to the provided context blocks.

Question: {state['question']}

Context:
{context}
"""
    ans = call_llm("You write precise, sourced answers.", [{"role":"user","content":prompt}], max_tokens=500)
    state["answer"] = ans.strip() + "\n\nSources: " + ", ".join(citations)
    
    logger.info(f"Generated Answer:\n{state['answer']}")
    logger.info("-" * 40)
    return state

def run_deep_rag(question: str) -> str:
    """Main entry point for Deep RAG pipeline."""
    logger.info("-" * 40)
    logger.info("DEEP RAG PIPELINE STARTED")
    logger.info("-" * 40)
    logger.info(f"Question: {question}")
    logger.info("")
    
    state: State = {"question": question, "plan":"", "evidence":[], "notes":"", "answer":"", "confidence":0.0, "iterations":0}
    
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
