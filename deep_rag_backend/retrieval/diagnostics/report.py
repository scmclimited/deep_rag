"""
Print human-readable inspection reports.
"""
from retrieval.diagnostics.inspect import inspect_document


def print_inspection_report(doc_title: str = None, doc_id: str = None):
    """Print a human-readable inspection report."""
    result = inspect_document(doc_title, doc_id)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if "documents" in result:
        # Already printed document list
        return
    
    doc = result["document"]
    stats = result["statistics"]
    pages = result["page_distribution"]
    
    print("=" * 80)
    print(f"Document Inspection Report")
    print("=" * 80)
    print(f"Title: {doc['title']}")
    print(f"Source: {doc['source_path']}")
    print(f"Document ID: {doc['doc_id']}")
    print()
    print(f"Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique pages: {stats['unique_pages']}")
    print(f"  Page range: {stats['page_range']}")
    print(f"  Avg text length: {stats['avg_text_length']} chars")
    print()
    print(f"Page Distribution:")
    print("-" * 80)
    
    for page_key in sorted(pages.keys(), key=lambda x: pages[x]["page_start"]):
        page_info = pages[page_key]
        print(f"  Pages {page_info['page_start']}-{page_info['page_end']}:")
        print(f"    Chunks: {page_info['chunk_count']}")
        print(f"    Total text length: {page_info['total_text_length']} chars")
        print(f"    Sample chunks:")
        for i, chunk in enumerate(page_info['chunks'][:3], 1):  # Show first 3 chunks per page
            print(f"      [{i}] {chunk['chunk_id']} ({chunk['content_type']}, {chunk['text_length']} chars)")
            print(f"          Preview: {chunk['text_preview']}...")
        if len(page_info['chunks']) > 3:
            print(f"      ... and {len(page_info['chunks']) - 3} more chunks")
        print()
    
    print("=" * 80)

