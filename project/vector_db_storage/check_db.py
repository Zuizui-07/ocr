import sys
from pathlib import Path

# Add project root to path
project_root = Path(r"C:\Users\APNMUM07\docling_ocr_project")
sys.path.insert(0, str(project_root))

from project.vector_db_storage.vector_store import VectorStore
from dotenv import load_dotenv
import json

load_dotenv()


def main():
    print("=" * 70)
    print("🔍 ChromaDB Inspector")
    print("=" * 70)
    
    # Initialize vector store
    store = VectorStore()
    
    # Get statistics
    stats = store.get_stats()
    
    print(f"\n📊 Database Statistics:")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total Documents: {stats['total_documents']}")
    print(f"   Location: {stats['persist_directory']}")
    
    if stats['total_documents'] > 0:
        # Peek at some documents
        print(f"\n📄 Sample Documents (first 5):")
        sample = store.collection.peek(limit=5)
        
        for i, (doc_id, document, metadata) in enumerate(zip(
            sample['ids'],
            sample['documents'],
            sample['metadatas']
        ), 1):
            print(f"\n  Document {i}:")
            print(f"    ID: {doc_id}")
            print(f"    Text: {document[:100]}{'...' if len(document) > 100 else ''}")
            print(f"    Metadata: {json.dumps(metadata, indent=6)}")
        
        # Test search
        print("\n" + "=" * 70)
        print("🔍 Test Search")
        print("=" * 70)
        
        test_query = input("\nEnter search query (or press Enter to skip): ").strip()
        
        if test_query:
            print(f"\nSearching for: '{test_query}'")
            results = store.search(test_query, n_results=3)
            
            print(f"\n📋 Top {len(results['documents'][0])} Results:")
            for i, (doc, meta, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                print(f"\n  Result {i}:")
                print(f"    Text: {doc[:150]}{'...' if len(doc) > 150 else ''}")
                print(f"    Page: {meta.get('page', 'N/A')}")
                print(f"    Source: {meta.get('source', 'N/A')}")
                print(f"    Similarity Score: {1 - distance:.4f}")
    else:
        print("\n⚠️  Collection is empty!")
        print("   Run 'python ocr_to_vector_pipeline.py' to add documents")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()