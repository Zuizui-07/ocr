from pathlib import Path
import sys
import json
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent))

from project.ocr_pipeline.docling_paddle_hybrid import DoclingPaddleHybrid
from project.text_preparation.text_processing import extract_chunks_from_json
from project.vector_db_storage.vector_store import VectorStore


def extract_images_from_pdf(pdf_path: Path, output_folder: Path):
    """
    Extract images from PDF and save to output folder
    Returns: List of extracted image info
    """
    doc = fitz.open(pdf_path)
    extracted_images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save in output folder with document name prefix
            image_filename = f"{pdf_path.stem}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
            image_path = output_folder / image_filename
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            extracted_images.append({
                'filename': image_filename,
                'page': page_num + 1,
                'path': str(image_path)
            })
    
    doc.close()
    return extracted_images


def process_documents_with_images():
    """
    Complete pipeline: PDFs → Extract Images → OCR → Database
    All outputs (JSON, TXT, Images) saved in output/ folder
    """
    print("=" * 80)
    print("🚀 BATCH OCR WITH IMAGE EXTRACTION")
    print("=" * 80)
    
    project_root = Path.cwd()
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Find PDFs and images
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    document_files = []
    
    for ext in supported_extensions:
        document_files.extend(project_root.glob(f"*{ext}"))
    
    if not document_files:
        print("❌ No documents found!")
        return
    
    pdfs = [f for f in document_files if f.suffix.lower() == '.pdf']
    images = [f for f in document_files if f.suffix.lower() != '.pdf']
    
    print(f"\n📂 Source: {project_root}")
    print(f"📁 Output: {output_dir}")
    print(f"\n📊 Found:")
    print(f"  📄 PDFs: {len(pdfs)}")
    print(f"  🖼️  Images: {len(images)}")
    
    proceed = input(f"\nProcess all files and extract images? (y/n): ").strip().lower()
    if proceed != 'y':
        print("❌ Cancelled")
        return
    
    # STEP 1: Extract images from PDFs
    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTING IMAGES FROM PDFs")
    print("=" * 80)
    
    total_extracted_images = 0
    extraction_summary = {}
    
    for pdf in pdfs:
        print(f"\n📄 {pdf.name}")
        try:
            extracted = extract_images_from_pdf(pdf, output_dir)
            extraction_summary[pdf.stem] = extracted
            total_extracted_images += len(extracted)
            
            if extracted:
                print(f"  ✅ Extracted {len(extracted)} images:")
                for img_info in extracted:
                    print(f"     • {img_info['filename']}")
            else:
                print(f"  ℹ️  No images found")
                
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    
    # STEP 2: OCR all documents
    print("\n" + "=" * 80)
    print("STEP 2: OCR PROCESSING")
    print("=" * 80)
    
    processor = DoclingPaddleHybrid(lang="en", use_gpu=False)
    
    processed_files = []
    
    for i, doc_file in enumerate(document_files, 1):
        print(f"\n[{i}/{len(document_files)}] 📄 {doc_file.name}")
        
        try:
            result = processor.process_document(str(doc_file))
            
            # Add extracted images info to result if it's a PDF
            if doc_file.stem in extraction_summary:
                if 'metadata' not in result:
                    result['metadata'] = {}
                result['metadata']['extracted_images'] = extraction_summary[doc_file.stem]
            
            # Save JSON
            json_path = output_dir / f"{doc_file.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Save Text
            if "pages" in result and result["pages"]:
                all_text = []
                for page_idx, page in enumerate(result["pages"], 1):
                    extracted_text = page.get("extracted_text", "")
                    if extracted_text:
                        all_text.append(f"=== Page {page_idx} ===\n{extracted_text}\n")
                
                if all_text:
                    text_path = output_dir / f"{doc_file.stem}.txt"
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(all_text))
            
            processed_files.append(json_path)
            
            total_blocks = result.get('metadata', {}).get('total_text_blocks', 0)
            total_pages = result.get('metadata', {}).get('total_pages', 1)
            print(f"  ✅ OCR complete: {total_pages} page(s), {total_blocks} text blocks")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 3: Store in database
    print("\n" + "=" * 80)
    print("STEP 3: STORING IN DATABASE")
    print("=" * 80)
    
    store = VectorStore()
    total_chunks = 0
    
    for json_path in processed_files:
        print(f"\n📄 {json_path.stem}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                doc_json = json.load(f)
            
            chunks = extract_chunks_from_json(doc_json)
            if chunks:
                store.add_chunks(chunks, source_name=json_path.stem)
                total_chunks += len(chunks)
                print(f"  ✅ Added {len(chunks)} chunks")
            else:
                print(f"  ⚠️  No chunks extracted")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PROCESSING SUMMARY")
    print("=" * 80)
    print(f"\n✅ Documents processed: {len(document_files)}")
    print(f"   • PDFs: {len(pdfs)}")
    print(f"   • Images: {len(images)}")
    
    print(f"\n🖼️  Images extracted: {total_extracted_images}")
    
    if extraction_summary:
        print(f"\n📋 Extraction details:")
        for doc_name, images in extraction_summary.items():
            if images:
                print(f"   • {doc_name}: {len(images)} images")
    
    stats = store.get_stats()
    print(f"\n💾 Vector Database:")
    print(f"   • Collection: {stats['collection_name']}")
    print(f"   • Total documents: {stats['total_documents']}")
    print(f"   • Chunks added: {total_chunks}")
    
    print(f"\n📁 All outputs saved in: {output_dir}/")
    print(f"   • JSON files (OCR data)")
    print(f"   • TXT files (extracted text)")
    print(f"   • Image files (extracted from PDFs)")
    
    print("\n" + "=" * 80)
    print("✨ PIPELINE COMPLETE!")
    print("=" * 80)
    print("\n🔍 Next steps:")
    print("   • View outputs: ls output/")
    print("   • Search database: python search_db.py 'query'")
    print("   • View stats: python view_stats.py")


if __name__ == "__main__":
    process_documents_with_images()
