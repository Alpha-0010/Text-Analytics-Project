"""
extract_text.py
---------------
Extracts plain text from all PDFs in data/raw_pdfs/ and saves each as a
.txt file in data/extracted_text/.

Usage:
    python src/extract_text.py

Requires: pdfplumber
"""

import re
from pathlib import Path

import pdfplumber

PDF_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"
TEXT_DIR = Path(__file__).parent.parent / "data" / "extracted_text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def is_boilerplate(line: str) -> bool:
    """
    Return True for lines that are likely headers, footers, or page numbers
    and should be discarded.
    """
    stripped = line.strip()
    # Pure page numbers
    if re.match(r"^\d{1,3}$", stripped):
        return True
    # Very short lines (< 3 chars)
    if len(stripped) < 3:
        return True
    return False


def clean_extracted_text(raw: str) -> str:
    """
    Post-process raw extracted text:
    - Remove boilerplate lines
    - Collapse excessive whitespace
    - Strip ligatures / smart quotes
    """
    # Normalise ligatures and smart quotes
    replacements = {
        "\ufb01": "fi", "\ufb02": "fl", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
        "\u00a0": " ",
    }
    for src, tgt in replacements.items():
        raw = raw.replace(src, tgt)

    lines = raw.splitlines()
    cleaned = [ln for ln in lines if not is_boilerplate(ln)]

    # Collapse multiple blank lines into one
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(pdf_path: Path) -> str:
    """
    Extract and clean all text from a PDF using pdfplumber.
    Returns the full cleaned text as a string.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                pages_text.append(page_text)

    raw = "\n".join(pages_text)
    return clean_extracted_text(raw)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        print("Run  python src/download_pdfs.py  first, or place PDFs manually.")
        return

    print(f"Found {len(pdf_files)} PDFs\n")
    success, failed = [], []

    for pdf_path in pdf_files:
        txt_path = TEXT_DIR / (pdf_path.stem + ".txt")

        if txt_path.exists():
            print(f"[SKIP] {pdf_path.name} (already extracted)")
            continue

        print(f"[....] {pdf_path.name}", end="", flush=True)
        try:
            text = extract_pdf(pdf_path)
            word_count = len(text.split())

            if word_count < 100:
                print(f" ->[WARN] Only {word_count} words extracted (may be scanned/image PDF)")
            else:
                print(f" ->{word_count:,} words")

            txt_path.write_text(text, encoding="utf-8")
            success.append(pdf_path.stem)

        except Exception as e:
            print(f" ->[FAIL] {e}")
            failed.append(pdf_path.stem)

    print(f"\n--- Summary ---")
    print(f"Extracted : {len(success)}")
    print(f"Failed    : {len(failed)}")
    if failed:
        print("Failed files:", failed)


if __name__ == "__main__":
    main()
