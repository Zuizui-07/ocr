import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from project.vector_db_storage.vector_store import VectorStore


def show_database_stats():
    """Display detailed database statistics"""
    store = VectorStore()
    stats = store.get_stats()
    
    print("=" * 70)
    print("📊 ChromaDB Statistics")
    print("=" * 70)
    print(f"\n📦 Collection: {stats['collection_name']}")
    print(f"📁 Location: {stats['persist_directory']}")
    print(f"📄 Total Documents: {stats['total_documents']}")
    
    if stats['total_documents'] > 0:
        # Get all documents
        all_docs = store.collection.get()
        
        # Count by source
        sources = {}
        pages = {}
        types = {}
        
        for meta in all_docs['metadatas']:
            source = meta.get('source', 'unknown')
            page = meta.get('page', 0)
            doc_type = meta.get('type', 'unknown')
            
            sources[source] = sources.get(source, 0) + 1
            pages[page] = pages.get(page, 0) + 1
            types[doc_type] = types.get(doc_type, 0) + 1
        
        print(f"\n📚 By Source:")
        for source, count in sources.items():
            print(f"   • {source}: {count} documents")
        
        print(f"\n📄 By Page:")
        for page, count in sorted(pages.items()):
            print(f"   • Page {page}: {count} documents")
        
        print(f"\n🏷️  By Type:")
        for doc_type, count in types.items():
            print(f"   • {doc_type}: {count} documents")
        
        # Average confidence
        confidences = [m.get('confidence', 0) for m in all_docs['metadatas'] if m.get('confidence')]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            print(f"\n📈 Average Confidence: {avg_conf:.2%}")
    else:
        print("\n⚠️  Database is empty!")
        print("   Run: python project\\ocr_to_vector_pipeline.py")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    show_database_stats()