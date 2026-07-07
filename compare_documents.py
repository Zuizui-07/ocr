import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from project.vector_db_storage.vector_store import VectorStore


def find_similar_documents(query: str, threshold: float = 0.7):
    """
    Find documents similar to each other based on query
    """
    print("=" * 80)
    print("🔄 DOCUMENT SIMILARITY ANALYSIS")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print(f"Similarity threshold: {threshold:.0%}\n")
    
    store = VectorStore()
    results = store.search(query, n_results=20)
    
    if not results['documents'][0]:
        print("⚠️  No results found!")
        return
    
    # Group by source document
    doc_groups = {}
    
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        similarity = 1 - dist
        
        if similarity >= threshold:
            source = meta.get('source', 'unknown')
            
            if source not in doc_groups:
                doc_groups[source] = {
                    'chunks': [],
                    'max_similarity': 0,
                    'avg_similarity': 0
                }
            
            doc_groups[source]['chunks'].append({
                'text': doc,
                'similarity': similarity,
                'page': meta.get('page', 'N/A')
            })
            
            doc_groups[source]['max_similarity'] = max(
                doc_groups[source]['max_similarity'],
                similarity
            )
    
    # Calculate averages
    for source, data in doc_groups.items():
        data['avg_similarity'] = sum(c['similarity'] for c in data['chunks']) / len(data['chunks'])
    
    # Sort by max similarity
    sorted_docs = sorted(doc_groups.items(), key=lambda x: x[1]['max_similarity'], reverse=True)
    
    print(f"📊 Found {len(sorted_docs)} relevant document(s):\n")
    
    for rank, (source, data) in enumerate(sorted_docs, 1):
        print(f"{'═' * 80}")
        print(f"Rank #{rank} - {source}")
        print(f"{'═' * 80}")
        print(f"🎯 Max Similarity: {data['max_similarity']:.2%}")
        print(f"📊 Avg Similarity: {data['avg_similarity']:.2%}")
        print(f"📄 Matching Chunks: {len(data['chunks'])}")
        
        print(f"\n📝 Top matching content:")
        for i, chunk in enumerate(data['chunks'][:3], 1):  # Show top 3
            print(f"\n   [{i}] Page {chunk['page']} | Score: {chunk['similarity']:.2%}")
            preview = chunk['text'][:150]
            print(f"   {preview}{'...' if len(chunk['text']) > 150 else ''}")
        
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    else:
        query = input("Enter search query: ").strip()
        threshold = 0.7
    
    if query:
        find_similar_documents(query, threshold)
    else:
        print("❌ No query provided!")
