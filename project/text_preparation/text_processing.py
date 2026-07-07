import re


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", "")
    return text.strip()


def chunk_text(text: str, chunk_size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def extract_chunks_from_json(doc_json):
    chunks = []

    # -------------------------
    # 1️⃣ PaddleOCR per page
    # -------------------------
    for page in doc_json.get("pages", []):
        page_num = page.get("page_number", 1)
        page_chunks_added = False

        for cell in page.get("paddle_ocr_cells", []):
            text = cell.get("text", "").strip()
            if not text:
                continue

            cleaned = clean_text(text)
            sub_chunks = chunk_text(cleaned)

            for c in sub_chunks:
                chunks.append({
                    "text": c,
                    "metadata": {
                        "page": page_num,
                        "type": "paddle_ocr",
                        "confidence": cell.get("confidence"),
                        "bbox": cell.get("bbox")
                    }
                })

            page_chunks_added = True

        # fallback
        if not page_chunks_added and "extracted_text" in page:
            cleaned = clean_text(page["extracted_text"])
            for c in chunk_text(cleaned):
                chunks.append({
                    "text": c,
                    "metadata": {
                        "page": page_num,
                        "type": "full_page"
                    }
                })

    # -------------------------
    # 2️⃣ Docling layout blocks
    # -------------------------
    for item in doc_json.get("texts", []):
        text = item.get("text", "").strip()
        if not text:
            continue

        label = item.get("label", "docling")

        page_no = 1
        prov = item.get("prov")
        if prov and isinstance(prov, list):
            page_no = prov[0].get("page_no", 1)

        cleaned = clean_text(text)

        for c in chunk_text(cleaned):
            chunks.append({
                "text": c,
                "metadata": {
                    "page": page_no,
                    "type": f"docling_{label}"
                }
            })

    return chunks
