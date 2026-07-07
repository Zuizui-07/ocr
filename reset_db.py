import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from project.vector_db_storage.vector_store import VectorStore


def reset_database():
    """⚠️ Delete all data from ChromaDB"""
    store = VectorStore()
    
    stats = store.get_stats()
    
    print("=" * 70)
    print("⚠️  DATABASE RESET WARNING")
    print("=" * 70)
    print(f"\nCollection: {stats['collection_name']}")
    print(f"Current Documents: {stats['total_documents']}")
    print(f"Location: {stats['persist_directory']}")
    
    confirm = input("\n⚠️  Delete ALL data? Type 'YES' to confirm: ")
    
    if confirm == "YES":
        store.reset_collection()
        print("\n✅ Database reset complete!")
        print("   All documents have been deleted.")
    else:
        print("\n❌ Reset cancelled")


if __name__ == "__main__":
    reset_database()