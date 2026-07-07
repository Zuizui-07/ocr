import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from project.text_preparation.text_processing import extract_chunks_from_json
from project.vector_db_storage.vector_store import VectorStore
import json


def process_all_jsons_in_folder(folder_path: str = "output"):
    """
    Process all JSON files in output folder and store in ChromaDB
    """
    folder = Path(folder_path) if Path(folder_path).is_absolute() else project_root / folder_path
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return
    
    # Find all JSON files
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"⚠️  No JSON files found in {folder}")
        return
    
    print("=" * 70)
    print(f"📄 Processing {len(json_files)} JSON file(s)")
    print("=" * 70)
    
    # Initialize vector store once
    store = VectorStore()
    
    total_chunks = 0
    successful = 0
    
    for i, json_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] {json_path.name}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc_json = json.load(f)
            
            chunks = extract_chunks_from_json(doc_json)
            
            if chunks:
                store.add_chunks(chunks, source_name=json_path.stem)
                total_chunks += len(chunks)
                successful += 1
                print(f"   ✅ Added {len(chunks)} chunks")
            else:
                print(f"   ⚠️  No chunks extracted")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Show final stats
    stats = store.get_stats()
    print("\n" + "=" * 70)
    print("📊 Final Statistics")
    print("=" * 70)
    print(f"✅ Processed successfully: {successful}/{len(json_files)}")
    print(f"📦 Total chunks added: {total_chunks}")
    print(f"💾 Total in database: {stats['total_documents']}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "output"
    
    process_all_jsons_in_folder(folder)