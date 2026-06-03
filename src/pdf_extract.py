"""
PDF text extraction for Woo dossier PDFs.

extract_text() renders every page via PyMuPDF and runs Tesseract OCR (nld+eng).
This handles both image-only pages and PDFs with custom font encodings that
produce garbled output from native text-layer extraction.

extract_text_native() is kept as a fast fallback for cases where the PDF is
known to have a clean, standard text layer.

Both return the same (text, page_map) tuple.
"""

from __future__ import annotations
from pathlib import Path


def extract_text(
    pdf_path: str | Path,
    dpi: int = 200,
) -> tuple[str, dict[int, tuple[int, int]]]:
    """
    Extract text from a PDF by rendering each page and running Tesseract OCR.

    Handles image-only pages and custom-encoded text layers (common in Dutch
    Woo dossiers where pdfplumber returns \\x01 garbage).

    Parameters
    ----------
    pdf_path : path to the PDF file
    dpi      : rendering resolution (200 balances speed and accuracy)

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

    n = len(doc)
    print(f"OCR: {Path(pdf_path).name}  ({n} pages) …")
    for page_num in range(n):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        page_text = pytesseract.image_to_string(img, lang="nld+eng")

        if (page_num + 1) % 20 == 0 or page_num == n - 1:
            print(f"  {page_num + 1}/{n} pages done")

        page_text_nl = page_text + "\n"
        start = cursor
        end = cursor + len(page_text_nl)
        page_map[page_num + 1] = (start, end)
        text_parts.append(page_text_nl)
        cursor = end

    doc.close()
    return "".join(text_parts), page_map


def extract_text_native(pdf_path: str | Path) -> tuple[str, dict[int, tuple[int, int]]]:
    """
    Extract text using the PDF's native text layer via pdfplumber.

    Fast, but fails silently on image-only pages and produces garbage for PDFs
    with custom font encodings. Only use this when you are certain the PDF has
    a clean, standard text layer.

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
            page_text_nl = page_text + "\n"
            start = cursor
            end = cursor + len(page_text_nl)
            page_map[page.page_number] = (start, end)
            text_parts.append(page_text_nl)
            cursor = end

    return "".join(text_parts), page_map


# Backwards-compatibility alias
extract_text_with_ocr = extract_text


def char_offset_to_page(char_offset: int, page_map: dict[int, tuple[int, int]]) -> int:
    """Return the page number containing the given character offset."""
    for page_num, (start, end) in page_map.items():
        if start <= char_offset < end:
            return page_num
    raise ValueError(f"Character offset {char_offset} is out of range.")
