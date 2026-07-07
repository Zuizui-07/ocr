import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from project.vector_db_storage.vector_store import VectorStore


def search_documents(query: str, n_results=10):
    """
    Search across all processed documents
    """
    print("=" * 80)
    print("🔍 DOCUMENT SEARCH")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print(f"Results to show: {n_results}\n")
    
    # Initialize store
    store = VectorStore()
    
    # Check if database has documents
    stats = store.get_stats()
    if stats['total_documents'] == 0:
        print("⚠️  Database is empty!")
        print("   Run: python batch_ocr_to_db.py")
        return
    
    print(f"📊 Searching {stats['total_documents']} documents...\n")
    
    # Search
    results = store.search(query, n_results=n_results)
    
    if not results['documents'][0]:
        print("⚠️  No results found!")
        return
    
    print("=" * 80)
    print(f"📋 TOP {len(results['documents'][0])} RESULTS")
    print("=" * 80)
    
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity_score = 1 - dist
        
        # Determine relevance
        if similarity_score > 0.8:
            relevance = "🟢 Highly Relevant"
        elif similarity_score > 0.6:
            relevance = "🟡 Moderately Relevant"
        else:
            relevance = "🟠 Somewhat Relevant"
        
        print(f"\n{'─' * 80}")
        print(f"Result #{i} | {relevance} | Score: {similarity_score:.2%}")
        print(f"{'─' * 80}")
        print(f"📄 Source: {meta.get('source', 'N/A')}")
        print(f"📑 Page: {meta.get('page', 'N/A')}")
        print(f"🏷️  Type: {meta.get('type', 'N/A')}")
        
        if meta.get('confidence'):
            print(f"✨ OCR Confidence: {meta.get('confidence'):.2%}")
        
        print(f"\n📝 Text:")
        # Show text with word wrap
        text_preview = doc if len(doc) <= 300 else doc[:300] + "..."
        print(f"   {text_preview}")
    
    print("\n" + "=" * 80)
    print("💡 TIP: Increase results with: python search_db.py 'query' 20")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    else:
        query = input("Enter search query: ").strip()
        n_results = 10
    
    if query:
        search_documents(query, n_results)
    else:
        print("❌ No query provided!")