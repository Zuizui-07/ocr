from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io


def extract_images_from_pdf(pdf_path: str, output_folder: str = "extracted_images"):
    """
    Extract all images from a PDF file
    """
    pdf_file = Path(pdf_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)
    
    if not pdf_file.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    print(f"📄 Processing: {pdf_file.name}")
    print(f"📁 Output: {output_dir}\n")
    
    # Open PDF
    doc = fitz.open(pdf_file)
    
    image_count = 0
    
    # Iterate through pages
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        print(f"Page {page_num + 1}: Found {len(image_list)} image(s)")
        
        # Extract each image
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get image extension
            image_ext = base_image["ext"]
            
            # Save image
            image_filename = f"{pdf_file.stem}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
            image_path = output_dir / image_filename
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            print(f"  ✅ Saved: {image_filename}")
            image_count += 1
    
    doc.close()
    
    print(f"\n✅ Extracted {image_count} images from {pdf_file.name}")
    print(f"📁 Images saved in: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter PDF path: ").strip()
    
    extract_images_from_pdf(pdf_path)