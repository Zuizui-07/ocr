from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from project.ocr_pipeline.docling_paddle_hybrid import DoclingPaddleHybrid
from project.text_preparation.text_processing import extract_chunks_from_json
from project.vector_db_storage.vector_store import VectorStore


def process_all_documents_to_db(input_folder: str = None, file_patterns: list = None):
    """
    Complete pipeline: Multiple files → OCR → Vector DB
    """
    print("=" * 80)
    print("🚀 BATCH OCR TO VECTOR DATABASE PIPELINE")
    print("=" * 80)
    
    # Setup paths
    if input_folder:
        source_dir = Path(input_folder)
    else:
        source_dir = Path.cwd()  # Current directory (project root)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Find all supported files
    if file_patterns:
        document_files = []
        for pattern in file_patterns:
            document_files.extend(source_dir.glob(pattern))
    else:
        # Default: find all PDFs and images
        supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        document_files = []
        for ext in supported_extensions:
            document_files.extend(source_dir.glob(f"*{ext}"))
    
    if not document_files:
        print(f"❌ No documents found in: {source_dir}")
        print(f"Supported: PDF, PNG, JPG, JPEG, BMP, TIFF")
        return
    
    # Show found files
    pdfs = [f for f in document_files if f.suffix.lower() == '.pdf']
    images = [f for f in document_files if f.suffix.lower() != '.pdf']
    
    print(f"\n📂 Source: {source_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"\n📊 Found {len(document_files)} file(s):")
    
    if pdfs:
        print(f"\n  📄 PDFs ({len(pdfs)}):")
        for pdf in pdfs[:5]:  # Show first 5
            print(f"     • {pdf.name}")
        if len(pdfs) > 5:
            print(f"     ... and {len(pdfs) - 5} more")
    
    if images:
        print(f"\n  🖼️  Images ({len(images)}):")
        for img in images[:5]:  # Show first 5
            print(f"     • {img.name}")
        if len(images) > 5:
            print(f"     ... and {len(images) - 5} more")
    
    print("\n" + "=" * 80)
    proceed = input(f"Process all {len(document_files)} files and store in database? (y/n): ").strip().lower()
    
    if proceed != 'y':
        print("❌ Cancelled")
        return
    
    # STEP 1: Initialize OCR processor
    print("\n" + "=" * 80)
    print("STEP 1: INITIALIZING OCR SYSTEM")
    print("=" * 80)
    processor = DoclingPaddleHybrid(lang="en", use_gpu=False)
    print("✅ OCR system ready\n")
    
    # STEP 2: Process all documents with OCR
    print("=" * 80)
    print("STEP 2: OCR PROCESSING")
    print("=" * 80)
    
    processed_files = []
    failed_files = []
    
    for i, doc_file in enumerate(document_files, 1):
        print(f"\n[{i}/{len(document_files)}] Processing: {doc_file.name}")
        
        try:
            # Run OCR
            result = processor.process_document(str(doc_file))
            
            # Save JSON output
            output_name = doc_file.stem
            json_path = output_dir / f"{output_name}.json"
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Count text blocks
            total_blocks = result.get('metadata', {}).get('total_text_blocks', 0)
            total_pages = result.get('metadata', {}).get('total_pages', 1)
            
            print(f"   ✅ OCR complete: {total_pages} page(s), {total_blocks} text blocks")
            print(f"   💾 Saved: {json_path.name}")
            
            processed_files.append({
                'file': doc_file.name,
                'json_path': json_path,
                'pages': total_pages,
                'blocks': total_blocks
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed_files.append(doc_file.name)
    
    if not processed_files:
        print("\n❌ No files were successfully processed!")
        return
    
    # STEP 3: Extract chunks and store in Vector DB
    print("\n" + "=" * 80)
    print("STEP 3: STORING IN VECTOR DATABASE")
    print("=" * 80)
    
    # Initialize vector store
    store = VectorStore()
    
    total_chunks = 0
    
    for file_info in processed_files:
        print(f"\n📄 Processing: {file_info['file']}")
        
        try:
            # Load JSON
            with open(file_info['json_path'], 'r', encoding='utf-8') as f:
                doc_json = json.load(f)
            
            # Extract text chunks
            chunks = extract_chunks_from_json(doc_json)
            
            if chunks:
                # Store in ChromaDB
                source_name = file_info['json_path'].stem
                store.add_chunks(chunks, source_name=source_name)
                
                total_chunks += len(chunks)
                print(f"   ✅ Added {len(chunks)} chunks to database")
            else:
                print(f"   ⚠️  No chunks extracted")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # STEP 4: Summary
    print("\n" + "=" * 80)
    print("📊 PIPELINE SUMMARY")
    print("=" * 80)
    
    stats = store.get_stats()
    
    print(f"\n✅ Successfully processed: {len(processed_files)}")
    if failed_files:
        print(f"❌ Failed: {len(failed_files)}")
        for failed in failed_files:
            print(f"   • {failed}")
    
    print(f"\n💾 Vector Database:")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Chunks added this run: {total_chunks}")
    
    print(f"\n📁 Outputs:")
    print(f"   JSON files: {output_dir}/")
    print(f"   Database: {stats['persist_directory']}")
    
    print("\n" + "=" * 80)
    print("✨ PIPELINE COMPLETE!")
    print("=" * 80)
    print("\n🔍 Next steps:")
    print("   • Search database: python search_db.py 'your query'")
    print("   • View stats: python view_stats.py")
    print("   • Check database: python project/vector_db_storage/check_db.py")


if __name__ == "__main__":
    import sys
    
    # Check for folder argument
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = None  # Use current directory
    
    process_all_documents_to_db(input_folder=folder)