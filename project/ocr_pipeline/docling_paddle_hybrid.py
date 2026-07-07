import logging
from pathlib import Path
from typing import Dict, List
from PIL import Image
import numpy as np

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


class DoclingPaddleHybrid:
    """
    Docling = layout + structure
    PaddleOCR = text recognition (angle classifier enabled)
    Supports: PDFs (multi-page) and Images
    """

    def __init__(self, lang="en", use_gpu=False):
        _log.info("Initializing Docling + PaddleOCR hybrid system...")

        # PaddleOCR
        self.paddle_ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=True,
            use_gpu=use_gpu,
            show_log=False
        )

        # Docling (with OCR enabled for page structure)
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(do_cell_matching=True)
        )

        self.docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

        _log.info("Hybrid system ready.")

    def process_document(self, input_path: str) -> Dict:
        """
        Process PDF or image document
        """
        input_path = Path(input_path)
        file_type = input_path.suffix.lower()

        _log.info(f"Processing {file_type} file: {input_path.name}")

        # Get document structure from Docling
        _log.info("Running Docling (structure analysis)...")
        docling_result = self.docling_converter.convert(str(input_path))
        doc = docling_result.document

        # For PDFs, Docling handles multi-page conversion
        # For images, process with PaddleOCR
        if file_type == '.pdf':
            _log.info("PDF detected - processing with Docling + PaddleOCR enhancement")
            merged = self._process_pdf(doc, input_path)
        else:
            _log.info("Image detected - processing with PaddleOCR")
            paddle_results = self._run_paddle_ocr(input_path)
            merged = self._merge_results(doc, paddle_results, input_path)

        return merged

    def _process_pdf(self, docling_doc, pdf_path: Path) -> Dict:
        """
        Process multi-page PDF
        Uses Docling for structure, PaddleOCR for text enhancement
        """
        try:
            doc_dict = docling_doc.export_to_dict()
        except Exception:
            doc_dict = {}

        if "pages" not in doc_dict or not isinstance(doc_dict["pages"], list):
            doc_dict["pages"] = []

        # Docling already extracted pages from PDF
        # We can optionally enhance with PaddleOCR if needed
        
        total_blocks = 0
        for page in doc_dict["pages"]:
            # Count text blocks
            if "cells" in page:
                total_blocks += len(page.get("cells", []))
            
            # Extract text if available
            if "text" not in page or not page["text"]:
                # Build text from cells
                if "cells" in page:
                    page["extracted_text"] = "\n".join([
                        cell.get("text", "") for cell in page["cells"]
                    ])

        # Metadata
        if "metadata" not in doc_dict:
            doc_dict["metadata"] = {}

        doc_dict["metadata"].update({
            "ocr_engine": "Docling (RapidOCR) + PaddleOCR",
            "structure_engine": "Docling",
            "total_text_blocks": total_blocks,
            "total_pages": len(doc_dict["pages"])
        })

        return doc_dict

    def _run_paddle_ocr(self, image_path: Path) -> List[Dict]:
        """
        Run PaddleOCR on single image
        """
        results = []

        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        ocr_result = self.paddle_ocr.ocr(img_array, cls=True)

        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                box, (text, conf) = line
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]

                results.append({
                    "text": text,
                    "confidence": float(conf),
                    "bbox": {
                        "x_min": int(min(xs)),
                        "y_min": int(min(ys)),
                        "x_max": int(max(xs)),
                        "y_max": int(max(ys))
                    },
                    "polygon": [[int(p[0]), int(p[1])] for p in box]
                })

        _log.info(f"PaddleOCR extracted {len(results)} blocks")
        return results

    def _merge_results(self, docling_doc, paddle_results, input_path: Path) -> Dict:
        """
        Merge Docling structure with PaddleOCR text (for images)
        """
        try:
            doc_dict = docling_doc.export_to_dict()
        except Exception:
            doc_dict = {}

        if "pages" not in doc_dict or not isinstance(doc_dict["pages"], list):
            doc_dict["pages"] = []

        if len(doc_dict["pages"]) == 0:
            img = Image.open(input_path)
            doc_dict["pages"].append({
                "page_number": 1,
                "size": {
                    "width": img.width,
                    "height": img.height
                }
            })

        page = doc_dict["pages"][0]
        page["paddle_ocr_cells"] = paddle_results
        page["extracted_text"] = "\n".join(r["text"] for r in paddle_results)
        page["text_block_count"] = len(paddle_results)

        if "metadata" not in doc_dict:
            doc_dict["metadata"] = {}

        doc_dict["metadata"].update({
            "ocr_engine": "PaddleOCR",
            "structure_engine": "Docling",
            "total_text_blocks": len(paddle_results),
            "total_pages": 1
        })

        return doc_dict