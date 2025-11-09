from __future__ import annotations
import argparse
from pathlib import Path

from inference.graph.builder import build_app

def export_graph_png(png_path: str = "inference/graph/artifacts/deep_rag_graph.png") -> str:
    """
    Export the compiled LangGraph to a PNG using Graphviz.
    Falls back to Mermaid if Graphviz rendering isn't available.
    Returns the path to the created file.
    """
    app = build_app()
    g = app.get_graph()
    out = Path(png_path)
    # Create parent directory if it doesn't exist
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        # LangGraph exposes a draw_png helper if graphviz is installed
        g.draw_png(str(out))
        return str(out)
    except Exception as e:
        # Fallback: write a Mermaid diagram
        mermaid = g.draw_mermaid()
        mmd_out = out.with_suffix(".mmd")
        mmd_out.write_text(mermaid, encoding="utf-8")
        print(
            "[graph-viz] Graphviz rendering failed; wrote Mermaid instead.\n"
            f"Reason: {e}\n"
            f"Mermaid file: {mmd_out}"
        )
        return str(mmd_out)

def main():
    parser = argparse.ArgumentParser(description="Export Deep RAG LangGraph visualization.")
    parser.add_argument("--out", default="inference/graph/artifacts/deep_rag_graph.png", help="Output PNG (or .mmd if Graphviz missing)")
    args = parser.parse_args()
    path = export_graph_png(args.out)
    print(f"[graph-viz] Wrote: {path}")

if __name__ == "__main__":
    main()
