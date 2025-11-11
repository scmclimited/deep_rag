"""
Graph builder for LangGraph pipeline.
"""
from langgraph.graph import StateGraph, END  # type: ignore[import-untyped]
from inference.graph.state import GraphState
from inference.graph.nodes import (
    node_planner,
    node_retriever,
    node_compressor,
    node_critic,
    node_refine_retrieve,
    node_synthesizer,
    node_citation_pruner
)
from inference.graph.routing import should_refine

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


def build_app(sqlite_path: str = "langgraph_state.sqlite"):
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("planner", node_planner)
    graph.add_node("retriever", node_retriever)
    graph.add_node("compressor", node_compressor)
    graph.add_node("critic", node_critic)
    graph.add_node("refine_retrieve", node_refine_retrieve)
    graph.add_node("synthesizer", node_synthesizer)
    graph.add_node("citation_pruner", node_citation_pruner)

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
    graph.add_edge("synthesizer", "citation_pruner")
    graph.add_edge("citation_pruner", END)

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

