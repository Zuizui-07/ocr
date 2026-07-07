# Docling + PaddleOCR Document Processing System

A complete **offline document processing pipeline** using:

-   **Docling** -- layout & structure analysis\
-   **PaddleOCR** -- high-accuracy OCR\
-   **ChromaDB** -- vector database for semantic search

------------------------------------------------------------------------

## Features

-   PDF & Image OCR (PNG, JPG, JPEG, TIFF, BMP)
-   Image extraction from PDFs
-   Layout detection (headers, sections, tables)
-   High-accuracy OCR using PaddleOCR
-   Semantic search using ChromaDB
-   Local persistent vector database
-   Batch processing support
-   Fully offline after model download

------------------------------------------------------------------------

## System Requirements

-   Python **3.10.x** (required)
-   Windows / Linux / macOS
-   RAM: 8 GB recommended
-   Disk: 5 GB free

------------------------------------------------------------------------

## Installation

### 1. Create Virtual Environment

``` powershell
python -m venv venv310
.\venv310\Scripts\activate
```

### 2. Install Dependencies

``` powershell
pip install -r requirements.txt
```

### 3. Install System Dependencies (Windows only)

``` powershell
winget install poppler
winget install Microsoft.VisualStudio.2022.BuildTools
```

Restart your PC after installation.

### 4. Verify Installation

``` powershell
python -c "import paddleocr, docling, chromadb; print('OK')"
```

------------------------------------------------------------------------

## Configuration

Create a `.env` file in the project root:

``` env
CHROMA_PERSIST_DIR=project/vector_db_storage/chroma_db
CHROMA_COLLECTION_NAME=docling-paddle
```

------------------------------------------------------------------------

## Project Structure

    docling_ocr_project/
    ├── output/
    ├── project/
    │   ├── ocr_pipeline/
    │   │   ├── docling_paddle_hybrid.py
    │   │   └── run_hybrid_ocr.py
    │   ├── text_preparation/
    │   │   └── text_processing.py
    │   ├── vector_db_storage/
    │   │   ├── chroma_db/
    │   │   ├── check_db.py
    │   │   └── vector_store.py
    │   └── ocr_to_vector_pipeline.py
    ├── batch_ocr_to_db.py
    ├── batch_ocr_with_images.py
    ├── compare_documents.py
    ├── extract_pdf_images.py
    ├── search_db.py
    ├── view_stats.py
    ├── reset_db.py
    ├── requirements.txt
    ├── .env
    ├── readme.md
    ├── IMG_2677.pdf
    ├── invoice.png
    └── sample.png

------------------------------------------------------------------------

## Usage

## One-Command Full Pipeline (Recommended)

You only need to run **one command** to perform the complete workflow:

```powershell
python batch_ocr_with_images.py


### Run OCR Pipeline

``` powershell
python -m project.ocr_pipeline.run_hybrid_ocr
```

### Store OCR Output into Vector DB

``` powershell
python project/ocr_to_vector_pipeline.py
```

### Batch OCR + Store

``` powershell
python batch_ocr_to_db.py

```

### Search Documents

``` powershell
python search_db.py "invoice total"
```

### View Database Statistics

``` powershell
python view_stats.py
```

### Reset Database

``` powershell
python reset_db.py
```

------------------------------------------------------------------------

## Database Location

    project/vector_db_storage/chroma_db/

------------------------------------------------------------------------

## Supported Formats

-   PDF
-   PNG
-   JPG / JPEG
-   BMP
-   TIFF

------------------------------------------------------------------------

## GPU Acceleration (Optional)

``` powershell
pip uninstall paddlepaddle
pip install paddlepaddle-gpu
```

Enable in code:

``` python
use_gpu = True
```

------------------------------------------------------------------------

## Common Fixes

### NumPy

``` powershell
pip install "numpy<2.0"
```

### Protobuf

``` powershell
pip install protobuf==3.20.3
```

### Empty Database

``` powershell
python project/ocr_to_vector_pipeline.py
```

------------------------------------------------------------------------

## Notes

-   All processing is local & offline
-   ChromaDB is persistent
-   Docling provides layout & structure
-   PaddleOCR provides accurate text recognition
-   Vector DB enables semantic search

------------------------------------------------------------------------

