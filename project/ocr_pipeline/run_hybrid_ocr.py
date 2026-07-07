from pathlib import Path
import sys
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project.ocr_pipeline.docling_paddle_hybrid import DoclingPaddleHybrid


def process_single_file(input_file: Path, output_dir: Path, processor):
    """Process a single file (PDF or image)"""
    print("\n" + "=" * 70)
    print(f"📄 Processing: {input_file.name}")
    print(f"📋 Type: {input_file.suffix.upper()}")
    print("=" * 70)
    
    try:
        # Process document
        result = processor.process_document(str(input_file))
        
        # Create output filename based on input
        output_name = input_file.stem
        
        # Save JSON
        json_path = output_dir / f"{output_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON saved: {json_path.name}")
        
        # Save text
        if "pages" in result and result["pages"]:
            all_text = []
            
            # Extract text from all pages
            for page_idx, page in enumerate(result["pages"], 1):
                extracted_text = page.get("extracted_text", "")
                if extracted_text:
                    all_text.append(f"=== Page {page_idx} ===\n{extracted_text}\n")
            
            if all_text:
                text_path = output_dir / f"{output_name}.txt"
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_text))
                print(f"✅ Text saved: {text_path.name}")
        
        # Show stats
        total_blocks = result.get('metadata', {}).get('total_text_blocks', 0)
        total_pages = len(result.get('pages', []))
        print(f"📊 Pages: {total_pages}, Text blocks: {total_blocks}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # File paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Get all supported files in project root
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    document_files = []
    
    for ext in supported_extensions:
        document_files.extend(project_root.glob(f"*{ext}"))
    
    if not document_files:
        print("❌ No supported files found in project root!")
        print(f"📂 Looking in: {project_root}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return
    
    # Organize by type
    pdfs = [f for f in document_files if f.suffix.lower() == '.pdf']
    images = [f for f in document_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']]
    
    print("=" * 70)
    print("🚀 Docling + PaddleOCR - Batch Processing")
    print("=" * 70)
    print(f"\n📂 Found {len(document_files)} file(s):")
    
    if pdfs:
        print(f"\n  📄 PDFs ({len(pdfs)}):")
        for pdf in pdfs:
            print(f"     • {pdf.name}")
    
    if images:
        print(f"\n  🖼️  Images ({len(images)}):")
        for img in images:
            print(f"     • {img.name}")
    
    print(f"\n📁 Output directory: {output_dir}")
    print("=" * 70)
    
    # Ask for confirmation
    proceed = input(f"\nProcess all {len(document_files)} files? (y/n): ").strip().lower()
    if proceed != 'y':
        print("❌ Cancelled")
        return
    
    # Initialize processor once
    print("\n🔧 Initializing Docling + PaddleOCR...")
    processor = DoclingPaddleHybrid(lang="en", use_gpu=False)
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, doc_file in enumerate(document_files, 1):
        print(f"\n[{i}/{len(document_files)}]", end=" ")
        if process_single_file(doc_file, output_dir, processor):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Processing Summary")
    print("=" * 70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()