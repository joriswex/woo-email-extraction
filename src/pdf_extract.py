"""
PDF text extraction for Woo dossier PDFs.

Two extraction modes:
    extract_text()          — fast, uses the PDF text layer via pdfplumber
    extract_text_with_ocr() — slower, renders pages as images and runs Surya OCR;
                              use for PDFs with garbled/custom-encoded text layers

Both return the same (text, page_map) tuple so the rest of the pipeline is
unaffected by which mode was used.
"""

from __future__ import annotations
from pathlib import Path


def extract_text(pdf_path: str | Path) -> tuple[str, dict[int, tuple[int, int]]]:
    """
    Extract all text from a PDF using the native text layer (pdfplumber).

    Parameters
    ----------
    pdf_path : path to the PDF file

    Returns
    -------
    text     : concatenated text of all pages, each followed by a newline
    page_map : {page_number: (start_char, end_char)} — 1-indexed
    """
    import pdfplumber

    text_parts: list[str] = []
    page_map: dict[int, tuple[int, int]] = {}
    cursor = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if not page_text.strip():
                print(
                    f"Warning: page {page.page_number} has no text layer "
                    f"(image-only or custom encoding). Consider extract_text_with_ocr()."
                )
            page_text_nl = page_text + "\n"
            start = cursor
            end = cursor + len(page_text_nl)
            page_map[page.page_number] = (start, end)
            text_parts.append(page_text_nl)
            cursor = end

    return "".join(text_parts), page_map


def extract_text_with_ocr(
    pdf_path: str | Path,
    dpi: int = 200,
) -> tuple[str, dict[int, tuple[int, int]]]:
    """
    Extract text by rendering each page as an image and running Tesseract OCR.

    Use this for PDFs where extract_text() produces garbled characters due to
    custom font encodings. Tesseract handles Dutch and English well and skips
    redaction rectangles cleanly.

    Parameters
    ----------
    pdf_path : path to the PDF file
    dpi      : rendering resolution (200 is a good balance of speed and accuracy)

    Returns
    -------
    text     : concatenated OCR text of all pages, each followed by a newline
    page_map : {page_number: (start_char, end_char)} — 1-indexed
    """
    import fitz
    import pytesseract
    from PIL import Image
    import io

    doc = fitz.open(str(pdf_path))
    text_parts: list[str] = []
    page_map: dict[int, tuple[int, int]] = {}
    cursor = 0
    matrix = fitz.Matrix(dpi / 72, dpi / 72)

    print(f"Running Tesseract OCR on {len(doc)} pages …")
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        page_text = pytesseract.image_to_string(img, lang="nld+eng")

        if (page_num + 1) % 10 == 0 or page_num == 0:
            print(f"  page {page_num + 1}/{len(doc)} done")

        page_text_nl = page_text + "\n"
        start = cursor
        end = cursor + len(page_text_nl)
        page_map[page_num + 1] = (start, end)
        text_parts.append(page_text_nl)
        cursor = end

    doc.close()
    return "".join(text_parts), page_map


def char_offset_to_page(char_offset: int, page_map: dict[int, tuple[int, int]]) -> int:
    """Return the page number containing the given character offset."""
    for page_num, (start, end) in page_map.items():
        if start <= char_offset < end:
            return page_num
    raise ValueError(f"Character offset {char_offset} is out of range.")
